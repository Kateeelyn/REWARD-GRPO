"""Train a reward model on pairwise preference data.

This script is intentionally parallel to the existing DPO repo's `train_trl.py`:
- same base model by default
- same idea of built-in sample data vs real dataset
- single entry point with argparse
"""

from __future__ import annotations

import argparse

import torch

from .data_utils import SAMPLE_PREFERENCE_DATA, build_preference_dataset
from .model_utils import load_reward_model_for_training, load_tokenizer

BUILTIN_DATASETS = ("hh", "shp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a reward model for GRPO.")
    parser.add_argument("--model_name", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--output_dir", default="qwen-coder-rm")
    parser.add_argument(
        "--dataset_name",
        default=None,
        help=(
            "Use 'hh' or 'shp' for built-in loaders, any Hugging Face dataset name "
            "that already contains prompt/chosen/rejected, or omit for the built-in "
            "3-example smoke-test dataset."
        ),
    )
    parser.add_argument(
        "--dataset_path",
        default=None,
        help="Optional local JSONL path with prompt/chosen/rejected rows.",
    )
    parser.add_argument(
        "--eval_split",
        default="test",
        help="Evaluation split name. If unavailable, evaluation is skipped.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument(
        "--use_4bit",
        action="store_true",
        help="Enable 4-bit quantization for smaller GPUs.",
    )
    return parser.parse_args()


def _load_split(dataset_name: str, split: str):
    try:
        return build_preference_dataset(dataset_name=dataset_name, split=split)
    except Exception as exc:
        print(f"[WARN] Could not load split '{split}' for '{dataset_name}': {exc}")
        return None


def main() -> None:
    args = parse_args()

    if args.dataset_path is not None:
        train_dataset = build_preference_dataset(path=args.dataset_path)
        eval_dataset = None
    elif args.dataset_name is not None:
        train_dataset = _load_split(args.dataset_name, "train")
        if train_dataset is None:
            raise RuntimeError(f"Could not load training split for '{args.dataset_name}'.")
        eval_dataset = _load_split(args.dataset_name, args.eval_split)
    else:
        print(
            "[INFO] No dataset provided. Using the built-in 3-example preference set.\n"
            "       This is a smoke test only."
        )
        train_dataset = build_preference_dataset(raw_data=SAMPLE_PREFERENCE_DATA)
        eval_dataset = None

    print(f"Train examples: {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"Eval examples: {len(eval_dataset)}")

    tokenizer = load_tokenizer(args.model_name, padding_side="right")
    model = load_reward_model_for_training(
        args.model_name,
        use_bf16=True,
        use_4bit=args.use_4bit,
        pad_token_id=tokenizer.pad_token_id,
    )

    from trl import RewardConfig, RewardTrainer

    training_args = RewardConfig(
        output_dir=args.output_dir,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_length=args.max_length,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) and torch.cuda.is_available(),
        logging_steps=10,
        save_steps=100,
        eval_strategy="epoch" if eval_dataset is not None else "no",
        report_to=["tensorboard"],
        remove_unused_columns=False,
        disable_dropout=True,
    )

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    try:
        trainer = RewardTrainer(**trainer_kwargs, processing_class=tokenizer)
    except TypeError:  # older TRL versions
        trainer = RewardTrainer(**trainer_kwargs, tokenizer=tokenizer)

    print("Starting reward-model training ...")
    print(f"  lr      = {args.lr}")
    print(f"  epochs  = {args.epochs}")
    print(f"  max_len = {args.max_length}")
    if args.use_4bit:
        print("  quant   = 4-bit")
    print()

    trainer.train()

    print(f"\nTraining complete. Saving to {args.output_dir} ...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
