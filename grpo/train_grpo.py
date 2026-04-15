"""GRPO training entry point for code generation.

Supported reward modes
----------------------
auto   - choose a sensible default based on available metadata
rm     - use a trained reward model only
code   - use executable code rewards only
hybrid - combine reward-model reward with code rewards
"""

from __future__ import annotations

import argparse
import os
from typing import Any, List, Optional, Sequence, Tuple

import torch

from .data_utils import (
    SAMPLE_CODE_TASKS,
    build_grpo_dataset,
    prompt_truncation_mode,
    truncate_prompt_dataset,
)
from .model_utils import (
    load_policy_model,
    load_reward_model_for_inference,
    load_tokenizer,
)
from .reward_utils import build_code_reward_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a GRPO policy model.")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--output_dir", default="qwen-coder-grpo")
    parser.add_argument(
        "--dataset_name",
        default=None,
        help=(
            "Use 'hh' or 'shp' for built-in preference loaders, any Hugging Face "
            "dataset name with either prompt-only rows or code metadata, or omit "
            "to run the built-in code-task smoke test dataset."
        ),
    )
    parser.add_argument(
        "--dataset_path",
        default=None,
        help="Optional local JSONL path for prompt-only/code/preference data.",
    )
    parser.add_argument("--eval_split", default="test")
    parser.add_argument(
        "--reward_mode",
        choices=["auto", "rm", "code", "hybrid"],
        default="auto",
        help="How rewards are computed during GRPO.",
    )
    parser.add_argument(
        "--reward_model_path",
        default=None,
        help="Path to a trained reward model or adapter (required for rm/hybrid).",
    )
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_prompt_tokens", type=int, default=256)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument(
        "--loss_type",
        default="dapo",
        choices=["grpo", "dapo", "dr_grpo"],
        help="GRPO loss formulation.",
    )
    parser.add_argument(
        "--reward_weights",
        default=None,
        help="Comma-separated weights overriding the mode defaults.",
    )
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Enable 4-bit quantization for the policy model.",
    )
    return parser.parse_args()


def parse_reward_weights(text: Optional[str]) -> Optional[List[float]]:
    if text is None:
        return None
    pieces = [piece.strip() for piece in text.split(",") if piece.strip()]
    if not pieces:
        return None
    return [float(piece) for piece in pieces]


def _load_dataset(args: argparse.Namespace):
    if args.dataset_path is not None:
        return build_grpo_dataset(path=args.dataset_path)
    if args.dataset_name is not None:
        return build_grpo_dataset(dataset_name=args.dataset_name, split="train")
    return build_grpo_dataset(raw_data=SAMPLE_CODE_TASKS)


def _load_eval_dataset(args: argparse.Namespace):
    if args.dataset_path is not None:
        return None
    if args.dataset_name is None:
        return None
    try:
        return build_grpo_dataset(dataset_name=args.dataset_name, split=args.eval_split)
    except Exception as exc:
        print(f"[WARN] Could not load eval split '{args.eval_split}': {exc}")
        return None


def resolve_reward_mode(
    requested: str,
    *,
    has_code_metadata: bool,
    has_reward_model: bool,
) -> str:
    if requested != "auto":
        return requested

    if has_code_metadata and has_reward_model:
        return "hybrid"
    if has_code_metadata:
        return "code"
    if has_reward_model:
        return "rm"
    raise ValueError(
        "Could not infer reward mode automatically. Provide either a reward model "
        "path for RM-style GRPO or a dataset with entry_point/test_cases for code rewards."
    )


def validate_generation_batch(args: argparse.Namespace) -> None:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    effective_batch = world_size * args.batch_size * args.grad_accum
    if effective_batch % args.num_generations != 0:
        raise ValueError(
            "num_generations must divide WORLD_SIZE * batch_size * grad_accum. "
            f"Got WORLD_SIZE={world_size}, batch_size={args.batch_size}, "
            f"grad_accum={args.grad_accum}, num_generations={args.num_generations}."
        )


def build_reward_stack(
    mode: str,
    *,
    reward_model_path: Optional[str],
) -> Tuple[Any, Optional[Sequence[Any]], Optional[List[float]]]:
    code_reward_funcs, code_default_weights = build_code_reward_bundle()

    if mode == "code":
        return code_reward_funcs, None, code_default_weights

    if reward_model_path is None:
        raise ValueError(f"reward_model_path is required when reward_mode='{mode}'.")

    reward_model = load_reward_model_for_inference(reward_model_path)
    reward_tokenizer = load_tokenizer(reward_model_path, padding_side="right")

    if mode == "rm":
        return reward_model, reward_tokenizer, [1.0]

    if mode == "hybrid":
        funcs: List[Any] = [reward_model] + list(code_reward_funcs)
        processing_classes: List[Any] = [reward_tokenizer, None, None, None]
        weights = [1.0] + code_default_weights
        return funcs, processing_classes, weights

    raise ValueError(f"Unsupported reward mode: {mode}")


def main() -> None:
    args = parse_args()
    validate_generation_batch(args)

    train_dataset = _load_dataset(args)
    eval_dataset = _load_eval_dataset(args)

    has_code_metadata = {"entry_point", "test_cases"}.issubset(set(train_dataset.column_names))
    mode = resolve_reward_mode(
        args.reward_mode,
        has_code_metadata=has_code_metadata,
        has_reward_model=args.reward_model_path is not None,
    )
    if mode in {"code", "hybrid"} and not has_code_metadata:
        raise ValueError(
            f"reward_mode='{mode}' requires dataset columns 'entry_point' and 'test_cases'."
        )

    print(f"Using reward mode: {mode}")
    print(f"Train examples: {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"Eval examples: {len(eval_dataset)}")

    tokenizer = load_tokenizer(args.model_name, padding_side="left")
    truncation_strategy = prompt_truncation_mode(args.dataset_name, has_code_metadata)
    train_dataset = truncate_prompt_dataset(
        train_dataset,
        tokenizer,
        args.max_prompt_tokens,
        strategy=truncation_strategy,
    )
    if eval_dataset is not None:
        eval_dataset = truncate_prompt_dataset(
            eval_dataset,
            tokenizer,
            args.max_prompt_tokens,
            strategy=truncation_strategy,
        )

    model = load_policy_model(
        args.model_name,
        use_bf16=True,
        use_4bit=args.use_4bit,
    )

    reward_funcs, reward_processing_classes, default_weights = build_reward_stack(
        mode,
        reward_model_path=args.reward_model_path,
    )
    reward_weights = parse_reward_weights(args.reward_weights) or default_weights

    from trl import GRPOConfig, GRPOTrainer

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) and torch.cuda.is_available(),
        logging_steps=10,
        save_steps=100,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        report_to=["tensorboard"],
        remove_unused_columns=False,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.beta,
        loss_type=args.loss_type,
        reward_weights=reward_weights,
        log_completions=False,
        disable_dropout=True,
    )

    trainer_kwargs = dict(
        model=model,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    if reward_processing_classes is not None:
        trainer_kwargs["reward_processing_classes"] = reward_processing_classes

    try:
        trainer = GRPOTrainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:  # fallback for older TRL installs
        trainer = GRPOTrainer(**trainer_kwargs, tokenizer=tokenizer)

    print("Starting GRPO training ...")
    print(f"  lr                  = {args.lr}")
    print(f"  epochs              = {args.epochs}")
    print(f"  num_generations     = {args.num_generations}")
    print(f"  max_prompt_tokens   = {args.max_prompt_tokens}")
    print(f"  max_completion_len  = {args.max_completion_length}")
    print(f"  beta                = {args.beta}")
    print(f"  loss_type           = {args.loss_type}")
    print(f"  reward_weights      = {reward_weights}")
    if args.use_4bit:
        print("  quant               = 4-bit")
    print()

    trainer.train()

    print(f"\nTraining complete. Saving to {args.output_dir} ...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
