"""Stage-scoped checkpointing.

The project trains component by component, so a checkpoint is never "the model" --
it is *one component at one stage*, saved so a later stage can load it frozen.

Layout on disk:

    checkpoints/
      world_model/
        stage_a_tokenizer/
          <run_name>/
            config.yaml          full resolved config for this run
            manifest.json        run metadata + every checkpoint written
            last.pt              latest
            best.pt              best by the stage's monitored metric
            step_010000.pt       periodic
      vla/
        stage_bc/<run_name>/...

Every `.pt` carries `component`, `stage`, `config_hash` and `parent` (the
checkpoint this stage was initialised from). `resolve_parent()` walks that chain,
so a dynamics checkpoint can always answer "which tokenizer produced my tokens?" --
the failure mode that silently ruins staged training.
"""

from __future__ import annotations

import json
import shutil
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml

from .config import REPO_ROOT, config_hash

CHECKPOINT_ROOT = REPO_ROOT / "checkpoints"


@dataclass
class CheckpointMeta:
    """Everything needed to know what a weight file is, without loading it."""

    component: str          # "tokenizer", "latent_action", "dynamics", "octo", ...
    stage: str              # "stage_a_tokenizer", "stage_bc", ...
    step: int = 0
    epoch: int = 0
    metrics: Dict[str, float] = field(default_factory=dict)
    config_hash: str = ""
    parent: Optional[str] = None      # path to the checkpoint this run started from
    frozen_parents: Dict[str, str] = field(default_factory=dict)  # role -> ckpt path
    created_at: float = field(default_factory=time.time)
    notes: str = ""


class CheckpointManager:
    """Owns one run directory. One instance per training stage invocation."""

    def __init__(
        self,
        stage: str,
        run_name: str,
        project: str = "world_model",
        root: Path | str = CHECKPOINT_ROOT,
        monitor: str = "loss",
        mode: str = "min",
    ) -> None:
        self.stage = stage
        self.run_name = run_name
        self.project = project
        self.dir = Path(root) / project / stage / run_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self.monitor = monitor
        self.mode = mode
        self._best = float("inf") if mode == "min" else float("-inf")
        self.manifest_path = self.dir / "manifest.json"
        self._entries: list[Dict[str, Any]] = []
        if self.manifest_path.exists():
            data = json.loads(self.manifest_path.read_text())
            self._entries = data.get("checkpoints", [])
            self._best = data.get("best_value", self._best)

    # ------------------------------------------------------------------ writing

    def save_config(self, cfg: Dict[str, Any]) -> Path:
        path = self.dir / "config.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        return path

    def save(
        self,
        state: Dict[str, Any],
        meta: CheckpointMeta,
        tag: str | None = None,
        is_best: bool | None = None,
    ) -> Path:
        """Write `state` (usually `{"model": sd, "optimizer": sd, ...}`) + metadata.

        Always refreshes `last.pt`. Copies to `best.pt` when the monitored metric
        improves (or when `is_best` is passed explicitly).
        """
        payload = {"state": state, "meta": asdict(meta)}
        tag = tag or f"step_{meta.step:08d}"
        path = self.dir / f"{tag}.pt"
        torch.save(payload, path)
        shutil.copyfile(path, self.dir / "last.pt")

        if is_best is None:
            value = meta.metrics.get(self.monitor)
            is_best = value is not None and (
                value < self._best if self.mode == "min" else value > self._best
            )
            if is_best:
                self._best = float(value)  # type: ignore[arg-type]
            elif not self._entries:
                # Always produce a `best.pt` on the first save. Without this, a
                # stage that ran with no validation set leaves `stage:best`
                # unresolvable and the next stage fails with a confusing
                # file-not-found rather than "you had no val data".
                is_best = True
        if is_best:
            shutil.copyfile(path, self.dir / "best.pt")

        self._entries.append(
            {"tag": tag, "path": str(path), "is_best": bool(is_best), **asdict(meta)}
        )
        self._write_manifest()
        return path

    def _write_manifest(self) -> None:
        self.manifest_path.write_text(
            json.dumps(
                {
                    "project": self.project,
                    "stage": self.stage,
                    "run_name": self.run_name,
                    "monitor": self.monitor,
                    "mode": self.mode,
                    "best_value": self._best,
                    "checkpoints": self._entries,
                },
                indent=2,
                default=str,
            )
        )

    def prune(self, keep_last: int = 3) -> None:
        """Delete periodic checkpoints beyond `keep_last`; never touches best/last."""
        periodic = [e for e in self._entries if e["tag"].startswith("step_")]
        for entry in periodic[:-keep_last] if keep_last else periodic:
            p = Path(entry["path"])
            if p.exists() and not entry.get("is_best"):
                p.unlink()


# ---------------------------------------------------------------------- reading


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> tuple[Dict[str, Any], CheckpointMeta]:
    """Load a checkpoint written by `CheckpointManager.save`."""
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    if "state" not in payload or "meta" not in payload:
        # Tolerate a bare state_dict from an external source (e.g. HuggingFace).
        return payload, CheckpointMeta(component="unknown", stage="external")
    return payload["state"], CheckpointMeta(**payload["meta"])


def load_component(
    module: torch.nn.Module,
    path: str | Path,
    key: str = "model",
    strict: bool = True,
    freeze: bool = False,
    expect_component: str | None = None,
) -> CheckpointMeta:
    """Load one component's weights into `module`, optionally freezing it.

    `expect_component` guards against the classic staged-training mistake of
    loading a decoder checkpoint into the tokenizer slot.
    """
    state, meta = load_checkpoint(path)
    if expect_component and meta.component not in (expect_component, "unknown"):
        raise ValueError(
            f"checkpoint {path} holds component '{meta.component}', "
            f"expected '{expect_component}'"
        )
    module.load_state_dict(state[key] if key in state else state, strict=strict)
    if freeze:
        module.eval()
        for p in module.parameters():
            p.requires_grad_(False)
    return meta


def resolve_lineage(path: str | Path) -> list[Dict[str, Any]]:
    """Walk `parent` links back to the first stage. Returns newest-first."""
    chain: list[Dict[str, Any]] = []
    seen: set[str] = set()
    current: Optional[str] = str(path)
    while current and current not in seen and Path(current).exists():
        seen.add(current)
        _, meta = load_checkpoint(current)
        chain.append({"path": current, **asdict(meta)})
        current = meta.parent
    return chain


def latest_run(stage: str, project: str = "world_model", root: Path | str = CHECKPOINT_ROOT) -> Optional[Path]:
    """Most recently modified run directory for a stage -- convenience for CLIs."""
    stage_dir = Path(root) / project / stage
    if not stage_dir.exists():
        return None
    runs = [d for d in stage_dir.iterdir() if d.is_dir()]
    return max(runs, key=lambda d: d.stat().st_mtime) if runs else None


def resolve_ckpt(
    spec: str | Path, project: str = "world_model", root: Path | str = CHECKPOINT_ROOT
) -> Path:
    """Accept either an explicit path or `stage:best` / `stage:last` shorthand.

    `--tokenizer_ckpt stage_a_tokenizer:best` resolves to the newest run's best.pt,
    so day-to-day commands do not carry timestamped paths.
    """
    spec = str(spec)
    if ":" in spec and not Path(spec).exists():
        stage, tag = spec.split(":", 1)
        run = latest_run(stage, project=project, root=root)
        if run is None:
            raise FileNotFoundError(f"no runs found for stage '{stage}' in {project}")
        resolved = run / f"{tag}.pt"
        if not resolved.exists():
            # The usual cause: the newest run crashed before its first save, so
            # its directory exists but holds no weights. Point at that rather
            # than at a bare missing-file error, and list what *is* available.
            available = sorted(p.name for p in run.glob("*.pt"))
            contents = (
                ", ".join(available)
                if available
                else "nothing -- did this run crash before finishing an epoch?"
            )
            others = [
                d.name
                for d in sorted((Path(root) / project / stage).iterdir())
                if d.is_dir() and any(d.glob("*.pt"))
            ]
            raise FileNotFoundError(
                f"'{spec}' resolved to the newest run '{run.name}', which has no "
                f"{tag}.pt.\n"
                f"  that run contains: {contents}\n"
                f"  runs with checkpoints: {', '.join(others) if others else 'none'}\n"
                f"Pass an explicit path to pick a specific run."
            )
        return resolved
    path = Path(spec)
    if not path.exists():
        raise FileNotFoundError(f"checkpoint not found: {path}")
    return path


__all__ = [
    "CHECKPOINT_ROOT",
    "CheckpointMeta",
    "CheckpointManager",
    "load_checkpoint",
    "load_component",
    "resolve_lineage",
    "resolve_ckpt",
    "latest_run",
    "config_hash",
]
