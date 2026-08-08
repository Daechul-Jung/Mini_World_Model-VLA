# Convenience targets. Every one is a thin wrapper over a script -- run the
# scripts directly when you need flags these do not expose.

PY ?= python3

.PHONY: help test components vram-vla vram-wm wm-a wm-b wm-c wm-d vla-bc clean-pycache

help:
	@grep -E '^[a-z-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

test:            ## contract tests (CPU, seconds)
	$(PY) -m pytest tests/ -q

components:      ## list every registered swappable component
	$(PY) scripts/tools/list_components.py

vram-vla:        ## measure VRAM/step-time for a VLA config
	$(PY) scripts/tools/vram_probe.py --project vla --config $(CONFIG)

vram-wm:         ## measure VRAM/step-time for a world-model stage
	$(PY) scripts/tools/vram_probe.py --project world_model --config $(CONFIG) --stage $(STAGE)

# ---------------------------------------------------------------- world model
# Stages are independent runs. Each loads its predecessors frozen.

wm-a:            ## stage A: video tokenizer (LSUN stills + TUM frames)
	$(PY) scripts/train/train_world_model.py --stage a --config genie_small_lsun.yaml

wm-b:            ## stage B: latent action model (VIDEO only -- not LSUN)
	$(PY) scripts/train/train_world_model.py --stage b --config genie_small.yaml

wm-c:            ## stage C: dynamics (needs A and B)
	$(PY) scripts/train/train_world_model.py --stage c --config genie_small.yaml \
		--tokenizer_ckpt stage_a_tokenizer:best \
		--latent_action_ckpt stage_b_latent_action:best

wm-d:            ## stage D: diffusion decoder (optional, do last)
	$(PY) scripts/train/train_world_model.py --stage d --config genie_small.yaml \
		--tokenizer_ckpt stage_a_tokenizer:best

# ------------------------------------------------------------------------ VLA

vla-bc:          ## behaviour cloning, Octo-small from scratch
	$(PY) scripts/train/train_vla.py --stage bc --config octo_small.yaml

clean-pycache:
	find . -name '__pycache__' -type d -not -path './venv*' -exec rm -rf {} + 2>/dev/null || true
