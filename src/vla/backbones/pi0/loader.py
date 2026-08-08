"""HuggingFace loading for pi0 PyTorch ports."""

from __future__ import annotations

from typing import Any, Tuple

import torch

DEFAULT_MODEL = "lerobot/pi0"


def load_pi0(
    model_name: str = DEFAULT_MODEL,
    load_in_4bit: bool = True,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[Any, Any, int]:
    """Return `(processor, model, hidden_dim)`.

    pi0's original release is JAX (openpi). This loads a PyTorch port from the
    Hub via `AutoModel` with `trust_remote_code`. If the checkpoint you pick is
    packaged as a LeRobot policy rather than a bare HF model, import it from
    `lerobot` instead and return the same triple -- everything downstream depends
    only on this signature.
    """
    try:
        from transformers import AutoModel, AutoProcessor
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pi0 needs `transformers`. pip install -r requirements-vla.txt") from exc

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": torch_dtype,
        "device_map": device_map,
    }
    if load_in_4bit:
        from transformers import BitsAndBytesConfig

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_name, **kwargs)

    config = model.config
    hidden_dim = getattr(config, "hidden_size", None) or getattr(
        getattr(config, "text_config", config), "hidden_size", 2048
    )
    return processor, model, int(hidden_dim)
