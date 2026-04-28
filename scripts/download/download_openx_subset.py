"""
Download a small subset of Open X-Embodiment (bridge_data_v2) for VLA training.

Bridge Dataset v2 contains tabletop manipulation tasks (pick & place, stacking,
pushing) with a WidowX robot arm. It's one of the most commonly used open-source
robot manipulation datasets.

Usage:
    # Download 500 episodes of bridge_data_v2 (~5GB)
    python scripts/download/download_openx_subset.py \
        --dataset bridge_data_v2 \
        --n_episodes 500 \
        --output_dir data/openx

Requirements:
    pip install tensorflow tensorflow_datasets rlds

Notes:
    - The first run downloads metadata and streams episodes from Google Cloud.
    - Subsequent runs use the local cache (set TFDS_DATA_DIR).
    - For pick-and-place only, filter by task_language below.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np


# Map dataset name to TFDS identifier
DATASET_REGISTRY = {
    "bridge_data_v2":     "bridge_dataset",          # Tabletop manipulation, ~60K episodes
    "fractal":            "fractal20220817_data",    # Google RT-1 data, diverse manipulation
    "kuka":               "kuka",                    # DeepMind Kuka manipulation
    "taco_play":          "taco_play",               # Bimanual tasks
}

# Keywords to filter for pick-and-place tasks (applied to language_instruction)
PICK_PLACE_KEYWORDS = [
    "pick", "place", "put", "move", "grab", "lift", "stack",
    "push", "slide", "pull",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="bridge_data_v2", choices=list(DATASET_REGISTRY))
    p.add_argument("--n_episodes", type=int, default=500, help="Max episodes to download")
    p.add_argument("--output_dir", default="data/openx")
    p.add_argument("--split", default="train", help="TFDS split string e.g. 'train[:1000]'")
    p.add_argument("--filter_pick_place", action="store_true", default=True,
                   help="Only keep episodes with pick/place instructions")
    p.add_argument("--image_size", type=int, default=256,
                   help="Resize images to this square size")
    p.add_argument("--cache_dir", default=None,
                   help="TFDS cache dir (default: ~/tensorflow_datasets)")
    return p.parse_args()


def _is_pick_place(instruction: str) -> bool:
    instr_lower = instruction.lower()
    return any(kw in instr_lower for kw in PICK_PLACE_KEYWORDS)


def download_bridge(
    n_episodes: int,
    output_dir: Path,
    split: str,
    filter_pick_place: bool,
    image_size: int,
    cache_dir: str | None,
) -> None:
    try:
        import tensorflow_datasets as tfds
        import tensorflow as tf
    except ImportError:
        raise SystemExit(
            "Missing dependencies. Run:\n"
            "  pip install tensorflow tensorflow_datasets"
        )

    tfds_name = DATASET_REGISTRY["bridge_data_v2"]
    print(f"Loading {tfds_name} (split={split}) from TFDS...")
    print("Note: First run streams from Google Cloud and may be slow.")

    builder = tfds.builder(tfds_name, data_dir=cache_dir)
    builder.download_and_prepare()

    ds = builder.as_dataset(split=split, shuffle_files=False)
    # RLDS format: each element is an episode dict with 'steps'

    output_dir.mkdir(parents=True, exist_ok=True)
    ep_count = 0
    meta = []

    for episode in ds:
        if ep_count >= n_episodes:
            break

        steps = list(episode["steps"].as_numpy_iterator())
        if not steps:
            continue

        # Get language instruction from first step
        instruction = steps[0]["observation"].get(
            "natural_language_instruction", b""
        )
        if isinstance(instruction, bytes):
            instruction = instruction.decode("utf-8", errors="ignore")

        if filter_pick_place and not _is_pick_place(instruction):
            continue

        # Extract frames, actions, rewards
        images, actions, rewards = [], [], []
        for step in steps:
            obs = step["observation"]
            img = obs.get("image_primary", obs.get("image", None))
            if img is None:
                continue

            # Resize image
            img_t = tf.image.resize(img, [image_size, image_size]).numpy().astype(np.uint8)
            images.append(img_t)

            actions.append(step["action"].numpy().astype(np.float32))
            rewards.append(float(step.get("reward", 0.0)))

        if len(images) < 5:
            continue

        # Save as compressed numpy
        ep_path = output_dir / f"episode_{ep_count:05d}.npz"
        np.savez_compressed(
            ep_path,
            images=np.stack(images),          # (T, H, W, 3)
            actions=np.stack(actions),        # (T, action_dim)
            rewards=np.array(rewards),        # (T,)
            instruction=np.array(instruction),
        )
        meta.append({"path": str(ep_path), "instruction": instruction, "length": len(images)})
        ep_count += 1

        if ep_count % 50 == 0:
            print(f"  Saved {ep_count} / {n_episodes} episodes")

    # Save metadata index
    meta_path = output_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDone. Saved {ep_count} episodes to {output_dir}")
    print(f"Metadata: {meta_path}")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    download_bridge(
        n_episodes=args.n_episodes,
        output_dir=output_dir,
        split=args.split,
        filter_pick_place=args.filter_pick_place,
        image_size=args.image_size,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
