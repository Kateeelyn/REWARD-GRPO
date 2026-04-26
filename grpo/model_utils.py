"""Model and tokenizer helpers for the GRPO pipeline.

The goal is to stay stylistically close to the existing DPO repo while adding
what GRPO and reward modeling need:

- causal-LM policy model for GRPO
- sequence-classification reward model for RewardTrainer
- LoRA wrappers for both stages
- explicit tokenizer padding-side control
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
)

try:
    from peft import AutoPeftModelForSequenceClassification
except Exception:  # pragma: no cover - older PEFT installs
    AutoPeftModelForSequenceClassification = None  # type: ignore[assignment]

TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def is_bf16_available() -> bool:
    """Return True when bfloat16 is available on the current machine."""
    return bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())


def resolve_torch_dtype(use_bf16: bool = True) -> torch.dtype:
    """Choose a safe torch dtype for model loading."""
    if use_bf16 and is_bf16_available():
        return torch.bfloat16
    return torch.float32


def maybe_build_bnb_config(
    use_4bit: bool,
    torch_dtype: torch.dtype,
) -> Optional[BitsAndBytesConfig]:
    """Return a 4-bit quantization config when requested."""
    if not use_4bit:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_use_double_quant=True,
    )


def auto_device_map() -> Optional[str]:
    """Use automatic device placement on GPU; stay on CPU otherwise."""
    return "auto" if torch.cuda.is_available() else None


def disable_dropout(model: nn.Module) -> None:
    """Set all dropout probabilities to zero for more stable RL training."""
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0


def get_policy_lora_config() -> LoraConfig:
    """LoRA configuration for the causal-LM policy model."""
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=TARGET_MODULES,
        task_type=TaskType.CAUSAL_LM,
        bias="none",
    )


def get_reward_lora_config() -> LoraConfig:
    """LoRA configuration for the reward model.

    `modules_to_save=["score"]` is important so the scalar reward head is kept
    trainable and saved together with the adapter.
    """
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.0,
        target_modules=TARGET_MODULES,
        task_type=TaskType.SEQ_CLS,
        bias="none",
        modules_to_save=["score"],
    )


def load_tokenizer(
    model_name_or_path: str,
    *,
    padding_side: str = "left",
):
    """Load a tokenizer and ensure pad token + padding side are configured.

    GRPO requires left padding for the policy tokenizer. Reward models can use
    either side; right padding is often more natural there.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = padding_side
    return tokenizer


def _wrap_with_lora(model, lora_config: LoraConfig, use_4bit: bool):
    if use_4bit:
        model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    disable_dropout(model)
    model.train()
    return model


def load_policy_model(
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    *,
    use_bf16: bool = True,
    use_4bit: bool = False,
):
    """Load the causal-LM policy model and wrap it with LoRA."""
    torch_dtype = resolve_torch_dtype(use_bf16)
    quantization_config = maybe_build_bnb_config(use_4bit, torch_dtype)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = _wrap_with_lora(model, get_policy_lora_config(), use_4bit=use_4bit)
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model


def load_reward_model_for_training(
    model_name: str = "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    *,
    use_bf16: bool = True,
    use_4bit: bool = False,
    pad_token_id: Optional[int] = None,
):
    """Load the sequence-classification reward model and wrap it with LoRA."""
    torch_dtype = resolve_torch_dtype(use_bf16)
    quantization_config = maybe_build_bnb_config(use_4bit, torch_dtype)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=1,
        torch_dtype=torch_dtype,
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if pad_token_id is not None:
        model.config.pad_token_id = pad_token_id
    model = _wrap_with_lora(model, get_reward_lora_config(), use_4bit=use_4bit)
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model


def load_reward_model_for_inference(
    model_name_or_path: str,
    *,
    use_bf16: bool = True,
):
    """Load a saved reward model or adapter for inference inside GRPO.

    This first tries the PEFT auto-loader because RewardTrainer often saves a
    LoRA adapter directory. If that is unavailable or fails, it falls back to a
    standard sequence-classification load.
    """
    torch_dtype = resolve_torch_dtype(use_bf16)

    model = None
    if AutoPeftModelForSequenceClassification is not None:
        try:
            model = AutoPeftModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                device_map=auto_device_map(),
                trust_remote_code=True,
                is_trainable=False,
            )
        except Exception:
            model = None

    if model is None:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=auto_device_map(),
            trust_remote_code=True,
        )

    disable_dropout(model)
    for param in model.parameters():
        param.requires_grad = False
    model.eval()
    return model
