"""Shared pytest fixtures.

Tests here are **contract tests**, not accuracy tests. They answer "does this
component honour its interface", which is the property that makes swapping
components safe. They run on CPU in seconds with tiny models and synthetic data,
so they can be run on every change.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))


@pytest.fixture(scope="session")
def device() -> torch.device:
    return torch.device("cpu")


@pytest.fixture
def tiny_frames() -> torch.Tensor:
    """(B=2, T=4, 3, 64, 64) in [-1, 1]."""
    torch.manual_seed(0)
    return torch.randn(2, 4, 3, 64, 64).clamp(-1, 1)


@pytest.fixture
def tiny_wm_config() -> dict:
    """A world model small enough to build and run in a second."""
    return {
        "tokenizer": {
            "name": "conv_vqvae",
            "base_channels": 16,
            "channel_mults": [1, 2, 4],
            "latent_dim": 32,
            "n_res_blocks": 1,
            "image_size": 64,
            "codebook_size": 128,
        },
        "latent_action": {
            "name": "vq_lam",
            "image_size": 64,
            "patch_size": 16,
            "num_actions": 8,
            "dim": 64,
            "depth": 1,
            "num_heads": 2,
            "action_dim": 32,
        },
        "dynamics": {
            "name": "causal_gpt",
            "latent_grid": [8, 8],
            "vocab_size": 128,
            "latent_dim": 32,
            "action_kind": "latent",
            "num_actions": 8,
            "n_layers": 1,
            "dim": 64,
            "num_heads": 2,
            "max_frames": 8,
        },
        "decoder": {
            "name": "diffusion_unet",
            "base_channels": 16,
            "channel_mults": [1, 2],
            "attn_at_levels": [False, True],
            "context_dim": 32,
            "n_res_blocks": 1,
        },
    }


@pytest.fixture
def tiny_policy_config() -> dict:
    return {
        "name": "octo_torch",
        "action_dim": 4,
        "action_chunk": 2,
        "obs_horizon": 2,
        "image_size": 64,
        "patch": 16,
        "dim": 64,
        "depth": 2,
        "num_heads": 2,
    }


@pytest.fixture
def tiny_observation():
    from common.types import Observation

    torch.manual_seed(0)
    return Observation(
        image=torch.randn(2, 2, 3, 64, 64).clamp(-1, 1),
        instruction=["pick up the red object", "place the pot in the sink"],
    )
