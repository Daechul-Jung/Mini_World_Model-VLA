"""Layer 0 -- the framework itself: registry, config, checkpoints.

If these fail, nothing else in the repo can be trusted, because every component
is reached through them.
"""

from __future__ import annotations

import pytest
import torch

from common.checkpoint import CheckpointManager, CheckpointMeta, load_checkpoint, resolve_lineage
from common.config import apply_overrides, config_hash, deep_merge
from common.registry import Registry


class TestRegistry:
    def test_register_and_build(self):
        reg = Registry("thing")

        @reg.register("widget", status="baseline")
        class Widget:
            def __init__(self, size=1):
                self.size = size

        assert "widget" in reg
        assert reg.build({"name": "widget", "size": 3}).size == 3
        assert reg.describe()["widget"]["status"] == "baseline"

    def test_unknown_name_lists_alternatives(self):
        reg = Registry("thing")
        reg.register("a")(lambda: None)
        with pytest.raises(KeyError, match="registered.*a"):
            reg.build({"name": "nope"})

    def test_duplicate_registration_rejected(self):
        reg = Registry("thing")
        reg.register("a")(lambda: 1)
        with pytest.raises(KeyError, match="already registered"):
            reg.register("a")(lambda: 2)

    def test_overrides_beat_config(self):
        reg = Registry("thing")

        @reg.register("w")
        class W:
            def __init__(self, size):
                self.size = size

        assert reg.build({"name": "w", "size": 1}, size=9).size == 9


class TestConfig:
    def test_deep_merge_is_recursive(self):
        base = {"a": {"b": 1, "c": 2}, "d": 3}
        assert deep_merge(base, {"a": {"b": 9}}) == {"a": {"b": 9, "c": 2}, "d": 3}

    def test_replace_directive_substitutes_wholesale(self):
        """A child switching component types must not inherit the parent's args."""
        base = {"data": {"name": "video_folder", "clip_len": 8, "stride": 4}}
        merged = deep_merge(base, {"data": {"_replace_": True, "name": "images", "limit": 10}})
        assert merged["data"] == {"name": "images", "limit": 10}
        assert "clip_len" not in merged["data"]

    def test_overrides_parse_as_yaml(self):
        cfg = apply_overrides({"a": {"b": 1}}, ["a.b=2", "a.c=[1,2]", "d=true"])
        assert cfg["a"]["b"] == 2
        assert cfg["a"]["c"] == [1, 2]
        assert cfg["d"] is True

    def test_hash_is_order_independent(self):
        assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
        assert config_hash({"a": 1}) != config_hash({"a": 2})


class TestCheckpoints:
    def test_save_load_roundtrip_and_best(self, tmp_path):
        mgr = CheckpointManager("stage_x", "run", project="test", root=tmp_path, monitor="loss")
        model = torch.nn.Linear(4, 4)

        mgr.save({"model": model.state_dict()}, CheckpointMeta("comp", "stage_x", step=1,
                                                              metrics={"loss": 1.0}))
        mgr.save({"model": model.state_dict()}, CheckpointMeta("comp", "stage_x", step=2,
                                                              metrics={"loss": 0.5}))
        mgr.save({"model": model.state_dict()}, CheckpointMeta("comp", "stage_x", step=3,
                                                              metrics={"loss": 2.0}))

        _, best = load_checkpoint(mgr.dir / "best.pt")
        _, last = load_checkpoint(mgr.dir / "last.pt")
        assert best.step == 2, "best.pt should track the monitored metric, not the last write"
        assert last.step == 3

    def test_first_save_always_produces_best(self, tmp_path):
        """A stage with no validation set must still leave `stage:best` resolvable."""
        mgr = CheckpointManager("stage_x", "run", project="test", root=tmp_path)
        mgr.save({"model": {}}, CheckpointMeta("comp", "stage_x", step=1))
        assert (mgr.dir / "best.pt").exists()

    def test_component_mismatch_is_rejected(self, tmp_path):
        """Loading a decoder checkpoint into the tokenizer slot must fail loudly."""
        from common.checkpoint import load_component

        mgr = CheckpointManager("stage_d", "run", project="test", root=tmp_path)
        model = torch.nn.Linear(4, 4)
        path = mgr.save({"model": model.state_dict()},
                        CheckpointMeta("decoder", "stage_d", step=1))

        with pytest.raises(ValueError, match="holds component 'decoder'"):
            load_component(torch.nn.Linear(4, 4), path, expect_component="tokenizer")

    def test_lineage_walks_parents(self, tmp_path):
        a = CheckpointManager("stage_a", "r", project="test", root=tmp_path)
        p1 = a.save({"model": {}}, CheckpointMeta("tokenizer", "stage_a", step=1))
        b = CheckpointManager("stage_c", "r", project="test", root=tmp_path)
        p2 = b.save({"model": {}}, CheckpointMeta("dynamics", "stage_c", step=1, parent=str(p1)))

        chain = resolve_lineage(p2)
        assert [c["component"] for c in chain] == ["dynamics", "tokenizer"]

    def test_freeze_removes_gradients(self, tmp_path):
        from common.checkpoint import load_component

        mgr = CheckpointManager("stage_a", "r", project="test", root=tmp_path)
        model = torch.nn.Linear(4, 4)
        path = mgr.save({"model": model.state_dict()},
                        CheckpointMeta("tokenizer", "stage_a", step=1))

        target = torch.nn.Linear(4, 4)
        load_component(target, path, freeze=True, expect_component="tokenizer")
        assert not any(p.requires_grad for p in target.parameters())
