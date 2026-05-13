"""
Download a small subset of Open X-Embodiment pick-and-place data for VLA training.

The default dataset is the UCSD pick-and-place RLDS conversion exposed through
TensorFlow Datasets.

Usage:
    # Download 500 episodes of pick-and-place data
    python3 scripts/download/download_openx_subset.py \
        --dataset ucsd_pick_place \
        --n_episodes 500 \
        --output_dir data/openx

Requirements:
    python3 -m pip install --user tensorflow tensorflow_datasets rlds

Notes:
    - The first run downloads metadata and streams episodes from Google Cloud.
    - Subsequent runs use the local cache (set TFDS_DATA_DIR or --cache_dir).
    - Use --no-filter-pick-place if you want to keep every instruction.
"""
import sys
sys.setrecursionlimit(50000)
import argparse
import json
import os
from importlib import metadata
from pathlib import Path
from textwrap import dedent

import numpy as np


# Map dataset name to TFDS identifier
# DATASET_REGISTRY = {
#     "bridge_data_v2":     "bridge_dataset",          # Tabletop manipulation, ~60K episodes
#     "fractal":            "fractal20220817_data",    # Google RT-1 data, diverse manipulation
#     "kuka":               "kuka",                    # DeepMind Kuka manipulation
#     "taco_play":          "taco_play",               # Bimanual tasks
# }

DATASET_REGISTRY = {
    "ucsd_pick_place": "ucsd_pick_and_place_dataset_converted_externally_to_rlds",
}

# Keywords to filter for pick-and-place tasks (applied to language_instruction)
PICK_PLACE_KEYWORDS = [
    "pick", "place", "put", "move", "grab", "lift", "stack",
    "push", "slide", "pull",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="ucsd_pick_place", choices=list(DATASET_REGISTRY))
    p.add_argument("--n_episodes", type=int, default=500, help="Max episodes to download")
    p.add_argument("--output_dir", default="data/openx")
    p.add_argument("--split", default="train", help="TFDS split string e.g. 'train[:1000]'")
    p.add_argument(
        "--filter-pick-place",
        dest="filter_pick_place",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Only keep episodes with pick/place instructions",
    )
    p.add_argument("--image_size", type=int, default=256,
                   help="Resize images to this square size")
    p.add_argument("--cache_dir", default=None,
                   help="TFDS cache dir (default: ~/tensorflow_datasets)")
    return p.parse_args()


def _is_pick_place(instruction: str) -> bool:
    instr_lower = instruction.lower()
    return any(kw in instr_lower for kw in PICK_PLACE_KEYWORDS)


def _decode_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return _decode_text(value.item())
    if isinstance(value, str):
        return value
    return str(value)


def _first_available(mapping: dict, *keys):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _extract_instruction(step: dict) -> str:
    """Read language instruction from common RLDS step or observation locations."""
    value = _first_available(
        step,
        "natural_language_instruction",
        "language_instruction",
        "task_language_instruction",
    )
    if value is not None:
        return _decode_text(value)

    obs = step.get("observation", {})
    value = _first_available(
        obs,
        "natural_language_instruction",
        "language_instruction",
        "task_language_instruction",
    )
    return _decode_text(value)


def _check_openx_environment() -> None:
    """
    Catch the common NumPy 2.x / old pyarrow crash before TFDS reaches Beam.

    The failure usually appears as:
      AttributeError: _ARRAY_API not found
      ImportError: numpy.core.multiarray failed to import
    """
    numpy_major = int(np.__version__.split(".", 1)[0])
    try:
        pyarrow_version = metadata.version("pyarrow")
    except metadata.PackageNotFoundError:
        pyarrow_version = None

    if numpy_major >= 2 and pyarrow_version is not None:
        pyarrow_major = int(pyarrow_version.split(".", 1)[0])
        if pyarrow_major < 14:
            raise SystemExit(
                dedent(
                    f"""
                    OpenX download environment is not compatible right now.

                    Installed versions:
                      numpy=={np.__version__}
                      pyarrow=={pyarrow_version}

                    pyarrow {pyarrow_version} was built for the NumPy 1.x ABI, but
                    this interpreter is loading NumPy {np.__version__}.

                    Fix option A, recommended for this downloader:
                      sudo apt install python3.10-venv   # if venv/pip is missing
                      python3 -m venv venv_openx_download
                      ./venv_openx_download/bin/python -m pip install -r requirements-openx.txt
                      ./venv_openx_download/bin/python scripts/download/download_openx_subset.py \\
                          --dataset ucsd_pick_place --n_episodes 100 --output_dir data/openx

                    Fix option B, if using your user Python:
                      python3 -m pip install --user 'numpy<2' 'pyarrow>=14' --upgrade
                    """
                ).strip()
            )

    try:
        import pyarrow  # noqa: F401
    except Exception as exc:
        raise SystemExit(
            dedent(
                f"""
                OpenX download environment is not compatible right now.

                pyarrow failed to import under NumPy {np.__version__}:
                  {type(exc).__name__}: {exc}

                Fix option A, recommended for this downloader:
                  sudo apt install python3.10-venv   # if venv/pip is missing
                  python3 -m venv venv_openx_download
                  ./venv_openx_download/bin/python -m pip install -r requirements-openx.txt
                  ./venv_openx_download/bin/python scripts/download/download_openx_subset.py \\
                      --dataset ucsd_pick_place --n_episodes 100 --output_dir data/openx

                Fix option B, if using your user Python:
                  python3 -m pip install --user 'numpy<2' 'pyarrow>=14' --upgrade

                The root cause is that apache-beam imports pyarrow, and the installed
                pyarrow wheel was compiled against NumPy 1.x while this interpreter is
                loading NumPy 2.x.
                """
            ).strip()
        ) from exc


def download_bridge(
    args,
    n_episodes: int,
    output_dir: Path,
    split: str,
    filter_pick_place: bool,
    image_size: int,
    cache_dir: str | None,
) -> None:
    _check_openx_environment()
    try:
        import tensorflow_datasets as tfds
        import tensorflow as tf
    except ImportError:
        raise SystemExit(
            "Missing dependencies. Run:\n"
            "  python3 -m pip install --user tensorflow tensorflow_datasets rlds"
        )
    tfds_name = DATASET_REGISTRY[args.dataset]
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

        # Get language instruction from first step. Different RLDS datasets put
        # this at either the step level or inside observation.
        instruction = _extract_instruction(steps[0])

        if filter_pick_place and not _is_pick_place(instruction):
            continue

        # Extract frames, actions, rewards
        images, actions, rewards = [], [], []
        for step in steps:
            obs = step["observation"]
            img = _first_available(
                obs,
                "image_primary",
                "image",
                "rgb",
                "image_0",
                "agentview_image",
            )
            if img is None:
                continue

            # Resize image
            img_t = tf.image.resize(img, [image_size, image_size]).numpy().astype(np.uint8)
            images.append(img_t)

            action = step.get("action")
            if hasattr(action, "numpy"):
                action = action.numpy()
            actions.append(np.asarray(action, dtype=np.float32))

            reward = step.get("reward", 0.0)
            if hasattr(reward, "numpy"):
                reward = reward.numpy()
            rewards.append(float(reward))

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
        args,
        n_episodes=args.n_episodes,
        output_dir=output_dir,
        split=args.split,
        filter_pick_place=args.filter_pick_place,
        image_size=args.image_size,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()
