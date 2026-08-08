"""Run logging: stdout + a JSONL metric stream + optional TensorBoard.

JSONL is the primary record because it survives without extra dependencies and
`scripts/tools/plot_run.py` can read it. TensorBoard is used when installed.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from .config import flatten


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
        force=True,
    )


class RunLogger:
    """Writes to stdout, `metrics.jsonl`, and TensorBoard if available."""

    def __init__(self, run_dir: Path | str, cfg: Optional[Dict[str, Any]] = None) -> None:
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        setup_logging()
        self.log = logging.getLogger(self.dir.name)
        self._jsonl = (self.dir / "metrics.jsonl").open("a")

        self._tb = None
        try:  # optional
            from torch.utils.tensorboard import SummaryWriter

            self._tb = SummaryWriter(str(self.dir / "tb"))
            if cfg:
                self._tb.add_text("config", f"```\n{json.dumps(cfg, indent=2, default=str)}\n```")
        except Exception:  # pragma: no cover - tensorboard is optional
            pass

        if cfg:
            (self.dir / "config_flat.json").write_text(
                json.dumps(flatten(cfg), indent=2, default=str)
            )

    def info(self, msg: str) -> None:
        self.log.info(msg)

    def log_scalars(self, prefix: str, metrics: Mapping[str, float], step: int) -> None:
        record = {"step": step}
        for key, value in metrics.items():
            full = f"{prefix}/{key}" if prefix and not key.startswith(("train/", "val/")) else key
            record[full] = float(value)
            if self._tb is not None:
                self._tb.add_scalar(full, float(value), step)
        self._jsonl.write(json.dumps(record) + "\n")
        self._jsonl.flush()

    def log_images(self, tag: str, images, step: int) -> None:
        """`images`: (N, C, H, W) float tensor in [0, 1]."""
        if self._tb is not None:
            import torchvision

            grid = torchvision.utils.make_grid(images.clamp(0, 1), nrow=4)
            self._tb.add_image(tag, grid, step)

    def log_video(self, tag: str, video, step: int, fps: int = 10) -> None:
        """`video`: (N, T, C, H, W) float tensor in [0, 1]."""
        if self._tb is not None:
            self._tb.add_video(tag, video.clamp(0, 1), step, fps=fps)

    def close(self) -> None:
        self._jsonl.close()
        if self._tb is not None:
            self._tb.close()
