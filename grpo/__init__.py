"""GRPO pipeline package for code-generation experiments.

This package is intentionally small and mirrors the structure of the user's
existing DPO repository so it can be dropped into the same project with
minimal edits.
"""

__all__ = [
    "data_utils",
    "model_utils",
    "reward_model",
    "reward_utils",
    "smoke_test",
    "train_grpo",
]

__version__ = "0.1.0"
