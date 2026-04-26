"""Small CPU-only verification for the GRPO package.

This file avoids model downloads. It checks:
1. preference dataset loading
2. code dataset loading
3. reward functions on known-good reference solutions
"""

from __future__ import annotations

from .data_utils import SAMPLE_CODE_TASKS, SAMPLE_PREFERENCE_DATA, build_code_dataset, build_preference_dataset
from .reward_utils import compile_reward_func, format_reward_func, unit_test_reward_func


def run_all_checks() -> None:
    pref_ds = build_preference_dataset(raw_data=SAMPLE_PREFERENCE_DATA)
    assert len(pref_ds) == 3
    assert set(pref_ds.column_names) == {"prompt", "chosen", "rejected"}

    code_ds = build_code_dataset(raw_data=SAMPLE_CODE_TASKS)
    assert len(code_ds) == 3
    assert {"prompt", "entry_point", "test_cases"}.issubset(set(code_ds.column_names))

    good_completions = [row["reference_solution"] for row in SAMPLE_CODE_TASKS]
    entry_points = [row["entry_point"] for row in SAMPLE_CODE_TASKS]
    test_cases = [row["test_cases"] for row in SAMPLE_CODE_TASKS]

    compile_scores = compile_reward_func(good_completions)
    format_scores = format_reward_func(good_completions)
    test_scores = unit_test_reward_func(
        good_completions,
        entry_point=entry_points,
        test_cases=test_cases,
    )

    assert compile_scores == [1.0, 1.0, 1.0]
    assert format_scores == [1.0, 1.0, 1.0]
    assert test_scores == [1.0, 1.0, 1.0]

    bad_completions = [
        "def add_one(x):\n    return x - 1\n",
        "def is_even(n):\n    return True\n",
        "def reverse_string(s):\n    return s\n",
    ]
    bad_scores = unit_test_reward_func(
        bad_completions,
        entry_point=entry_points,
        test_cases=test_cases,
    )
    assert all(score < 1.0 for score in bad_scores)

    try:
        from trl import GRPOTrainer, RewardTrainer  # noqa: F401
    except Exception:
        # Importing TRL is optional for this pure-Python smoke test.
        pass

    print("All smoke checks passed.")


if __name__ == "__main__":
    run_all_checks()
