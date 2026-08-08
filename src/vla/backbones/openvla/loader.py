"""HuggingFace loading for OpenVLA, with the quantisation options that matter.

Kept separate from `policy.py` so the download/quantisation logic can be reused
by an eval script that never builds a policy, and so a missing optional
dependency produces one clear error instead of an import chain.
"""

from __future__ import annotations

from typing import Any, Tuple

import torch

DEFAULT_MODEL = "openvla/openvla-7b"


def load_openvla(
    model_name: str = DEFAULT_MODEL,
    load_in_4bit: bool = True,
    device_map: str = "auto",
    torch_dtype: torch.dtype = torch.bfloat16,
) -> Tuple[Any, Any, int]:
    """Return `(processor, model, hidden_dim)`.

    `trust_remote_code=True` is required: OpenVLA ships custom modelling code
    (the Prismatic VLM wrapper and `predict_action`) rather than using a stock
    HF architecture.

    Download size is ~15 GB. It caches under `HF_HOME`; set that to a disk with
    room before the first call, given this project's storage constraints.
    """
    try:
        from transformers import AutoModelForVision2Seq, AutoProcessor
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "OpenVLA needs `transformers`. pip install -r requirements-vla.txt"
        ) from exc

    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "low_cpu_mem_usage": True,
        "torch_dtype": torch_dtype,
        "device_map": device_map,
    }

    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "4-bit loading needs `bitsandbytes`. pip install bitsandbytes"
            ) from exc
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(model_name, **kwargs)

    config = model.config
    hidden_dim = getattr(config, "hidden_size", None) or getattr(
        getattr(config, "text_config", config), "hidden_size", 4096
    )
    return processor, model, int(hidden_dim)


def attach_lora(model: Any, r: int = 32, alpha: int = 16, dropout: float = 0.0) -> Any:
    """Wrap the backbone in LoRA adapters.

    Provided for completeness and for use on a larger GPU. On 8 GB this will
    OOM -- OpenVLA LoRA at r=32 is reported at roughly 27 GB. Prefer the frozen
    backbone + `PolicyModule` route, which trains a comparable number of
    parameters without holding optimiser state for the 7B trunk.
    """
    try:
        from peft import LoraConfig, get_peft_model
    except ImportError as exc:  # pragma: no cover
        raise ImportError("LoRA needs `peft`. pip install peft") from exc

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules="all-linear",
        init_lora_weights="gaussian",
    )
    peft_model = get_peft_model(model, config)
    peft_model.print_trainable_parameters()
    return peft_model
