"""Dataset utilities for reward-model training and GRPO.

This module keeps the data flow parallel to the user's DPO repo:

1. reward-model stage uses preference data with columns:
   {"prompt", "chosen", "rejected"}
2. GRPO stage uses prompt-only data, optionally with extra columns such as:
   {"prompt", "entry_point", "test_cases", "task"}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from datasets import Dataset as HFDataset
except Exception:  # pragma: no cover - lightweight fallback for smoke tests
    class HFDataset(list):
        @classmethod
        def from_list(cls, rows):
            return cls(rows)

        @property
        def column_names(self):
            return list(self[0].keys()) if self else []

        def map(self, fn):
            return HFDataset.from_list([fn(row) for row in self])

SUPPORTED_DATASETS = ("hh", "shp")
DATASET_TRUNCATION_MODE = {
    "hh": "keep_end",
    "shp": "keep_start",
    "code": "keep_start",
}

SAMPLE_PREFERENCE_DATA: List[Dict[str, str]] = [
    {
        "prompt": (
            "\n\nHuman: Write a Python function `second_largest(nums)` that returns "
            "the second largest unique element in a list of integers. Return "
            "None if fewer than 2 unique elements exist.\n\nAssistant:"
        ),
        "chosen": (
            "def second_largest(nums):\n"
            "    unique = sorted(set(nums), reverse=True)\n"
            "    if len(unique) < 2:\n"
            "        return None\n"
            "    return unique[1]\n"
        ),
        "rejected": (
            "def second_largest(nums):\n"
            "    nums.sort()\n"
            "    return nums[-2]\n"
        ),
    },
    {
        "prompt": (
            "\n\nHuman: Write a Python function `is_palindrome(s)` that returns "
            "True if the string is a palindrome when ignoring case and spaces."
            "\n\nAssistant:"
        ),
        "chosen": (
            "def is_palindrome(s):\n"
            "    cleaned = ''.join(ch.lower() for ch in s if ch != ' ')\n"
            "    return cleaned == cleaned[::-1]\n"
        ),
        "rejected": (
            "def is_palindrome(s):\n"
            "    return s == s[::-1]\n"
        ),
    },
    {
        "prompt": (
            "\n\nHuman: Write a Python function `flatten(lst)` that recursively "
            "flattens a nested list of integers into a single list.\n\nAssistant:"
        ),
        "chosen": (
            "def flatten(lst):\n"
            "    out = []\n"
            "    for item in lst:\n"
            "        if isinstance(item, list):\n"
            "            out.extend(flatten(item))\n"
            "        else:\n"
            "            out.append(item)\n"
            "    return out\n"
        ),
        "rejected": (
            "def flatten(lst):\n"
            "    return [x for x in lst if not isinstance(x, list)]\n"
        ),
    },
]

SAMPLE_CODE_TASKS: List[Dict[str, Any]] = [
    {
        "prompt": (
            "Write a Python function `add_one(x)` that returns x + 1. "
            "Only output Python code."
        ),
        "entry_point": "add_one",
        "reference_solution": "def add_one(x):\n    return x + 1\n",
        "test_cases": [
            {"input": [1], "expected": 2},
            {"input": [-3], "expected": -2},
            {"input": [0], "expected": 1},
        ],
        "task": "coding",
    },
    {
        "prompt": (
            "Write a Python function `is_even(n)` that returns True if n is even "
            "and False otherwise. Only output Python code."
        ),
        "entry_point": "is_even",
        "reference_solution": "def is_even(n):\n    return n % 2 == 0\n",
        "test_cases": [
            {"input": [2], "expected": True},
            {"input": [5], "expected": False},
            {"input": [0], "expected": True},
        ],
        "task": "coding",
    },
    {
        "prompt": (
            "Write a Python function `reverse_string(s)` that returns the reversed "
            "string. Only output Python code."
        ),
        "entry_point": "reverse_string",
        "reference_solution": (
            "def reverse_string(s):\n"
            "    return s[::-1]\n"
        ),
        "test_cases": [
            {"input": ["abc"], "expected": "cba"},
            {"input": [""], "expected": ""},
            {"input": ["racecar"], "expected": "racecar"},
        ],
        "task": "coding",
    },
]




def _is_code_like_row(row: Dict[str, Any]) -> bool:
    return {"prompt", "entry_point", "test_cases"}.issubset(row.keys())


def _is_preference_row(row: Dict[str, Any]) -> bool:
    return {"prompt", "chosen", "rejected"}.issubset(row.keys())


def _is_prompt_only_row(row: Dict[str, Any]) -> bool:
    return "prompt" in row and not _is_code_like_row(row) and not _is_preference_row(row)


def _validate_string_preference_record(record: Dict[str, Any]) -> Dict[str, str]:
    required = ("prompt", "chosen", "rejected")
    missing = [key for key in required if key not in record]
    if missing:
        raise ValueError(f"Preference row missing required keys: {missing}")
    cleaned: Dict[str, str] = {}
    for key in required:
        value = record[key]
        if not isinstance(value, str):
            raise TypeError(f"Preference field '{key}' must be a string, got {type(value)!r}")
        cleaned[key] = value
    return cleaned


def _validate_code_record(record: Dict[str, Any]) -> Dict[str, Any]:
    if "prompt" not in record:
        raise ValueError("Code RL row must include a 'prompt' field.")
    if not isinstance(record["prompt"], str):
        raise TypeError("Code RL field 'prompt' must be a string.")

    normalized = dict(record)
    if "test_cases" in normalized and isinstance(normalized["test_cases"], str):
        normalized["test_cases"] = json.loads(normalized["test_cases"])

    if "test_cases" not in normalized:
        raise ValueError("Code RL row must include 'test_cases'.")
    if not isinstance(normalized["test_cases"], list):
        raise TypeError("Field 'test_cases' must be a list or a JSON string.")

    if "entry_point" not in normalized or not isinstance(normalized["entry_point"], str):
        raise ValueError("Code RL row must include string field 'entry_point'.")

    if "task" not in normalized:
        normalized["task"] = "coding"

    return normalized


def _load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} of {path}: {exc}") from exc
            if not isinstance(row, dict):
                raise TypeError(f"JSONL row {line_no} in {path} must be an object.")
            records.append(row)
    return records


def _extract_anthropic_prompt(text: str) -> str:
    marker = "\n\nAssistant:"
    idx = text.rfind(marker)
    if idx < 0:
        raise ValueError("Anthropic HH row does not contain the expected assistant marker.")
    return text[: idx + len(marker)]


def load_hh(split: str = "train", silent: bool = False) -> List[Dict[str, str]]:
    from datasets import load_dataset

    dataset = load_dataset("Anthropic/hh-rlhf", split=split)
    data: List[Dict[str, str]] = []
    skipped = 0
    for row in dataset:
        try:
            prompt = _extract_anthropic_prompt(row["chosen"])
            rejected_prompt = _extract_anthropic_prompt(row["rejected"])
            if rejected_prompt != prompt:
                skipped += 1
                continue
            chosen = row["chosen"][len(prompt) :]
            rejected = row["rejected"][len(prompt) :]
            data.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
        except Exception:
            skipped += 1
    if not silent:
        print(f"Loaded HH split='{split}' with {len(data)} usable rows ({skipped} skipped).")
    return data


def load_shp(
    split: str = "train",
    *,
    min_score_ratio: float = 2.0,
    silent: bool = False,
) -> List[Dict[str, str]]:
    from datasets import load_dataset

    dataset = load_dataset("stanfordnlp/SHP", split=split)
    data: List[Dict[str, str]] = []
    skipped = 0
    for row in dataset:
        score_a = row.get("score_A", 0)
        score_b = row.get("score_B", 0)
        if score_a <= 0 or score_b <= 0:
            skipped += 1
            continue
        ratio = max(score_a, score_b) / min(score_a, score_b)
        if ratio < min_score_ratio:
            skipped += 1
            continue

        prompt = "\n\nHuman: " + row["history"] + "\n\nAssistant:"
        if row.get("labels") == 1:
            chosen = " " + row["human_ref_A"]
            rejected = " " + row["human_ref_B"]
        else:
            chosen = " " + row["human_ref_B"]
            rejected = " " + row["human_ref_A"]
        data.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})

    if not silent:
        print(f"Loaded SHP split='{split}' with {len(data)} usable rows ({skipped} skipped).")
    return data


def _load_preference_records(
    *,
    dataset_name: Optional[str] = None,
    split: str = "train",
    raw_data: Optional[Iterable[Dict[str, Any]]] = None,
    path: Optional[str] = None,
) -> List[Dict[str, str]]:
    if raw_data is not None:
        return [_validate_string_preference_record(dict(row)) for row in raw_data]

    if path is not None:
        rows = _load_jsonl(path)
        return [_validate_string_preference_record(row) for row in rows]

    if dataset_name is None:
        return [_validate_string_preference_record(row) for row in SAMPLE_PREFERENCE_DATA]

    if dataset_name == "hh":
        return load_hh(split=split)
    if dataset_name == "shp":
        return load_shp(split=split)

    from datasets import load_dataset

    dataset = load_dataset(dataset_name, split=split)
    return [_validate_string_preference_record(dict(row)) for row in dataset]


def build_preference_dataset(
    *,
    dataset_name: Optional[str] = None,
    split: str = "train",
    raw_data: Optional[Iterable[Dict[str, Any]]] = None,
    path: Optional[str] = None,
) -> HFDataset:
    """Build a preference dataset for RewardTrainer.

    Accepted sources:
    - built-in sample data
    - `hh` or `shp`
    - any HF dataset that already exposes prompt/chosen/rejected
    - local JSONL file with prompt/chosen/rejected
    """
    records = _load_preference_records(
        dataset_name=dataset_name,
        split=split,
        raw_data=raw_data,
        path=path,
    )
    return HFDataset.from_list(records)


def preference_to_prompt_records(
    preference_rows: Iterable[Dict[str, Any]],
    *,
    keep_gold_completion: bool = True,
) -> List[Dict[str, Any]]:
    """Convert preference rows into prompt-only rows for GRPO.

    `gold_completion` is optional but useful for debugging and future reward
    design. GRPO itself only requires the `prompt` column.
    """
    prompt_rows: List[Dict[str, Any]] = []
    for row in preference_rows:
        cleaned = _validate_string_preference_record(dict(row))
        prompt_row: Dict[str, Any] = {"prompt": cleaned["prompt"]}
        if keep_gold_completion:
            prompt_row["gold_completion"] = cleaned["chosen"]
        prompt_rows.append(prompt_row)
    return prompt_rows


def build_code_dataset(
    *,
    raw_data: Optional[Iterable[Dict[str, Any]]] = None,
    path: Optional[str] = None,
    dataset_name: Optional[str] = None,
    split: str = "train",
) -> HFDataset:
    """Build a prompt-only coding dataset with executable metadata."""
    if raw_data is not None:
        rows = [dict(row) for row in raw_data]
    elif path is not None:
        rows = _load_jsonl(path)
    elif dataset_name is not None:
        from datasets import load_dataset

        dataset = load_dataset(dataset_name, split=split)
        rows = [dict(row) for row in dataset]
    else:
        rows = [dict(row) for row in SAMPLE_CODE_TASKS]

    normalized = [_validate_code_record(row) for row in rows]
    return HFDataset.from_list(normalized)


def build_grpo_dataset(
    *,
    dataset_name: Optional[str] = None,
    split: str = "train",
    raw_data: Optional[Iterable[Dict[str, Any]]] = None,
    path: Optional[str] = None,
    require_code_metadata: bool = False,
) -> HFDataset:
    """Build a GRPO dataset.

    When `require_code_metadata=False`, the function accepts three source styles:

    1. preference rows: {prompt, chosen, rejected} -> converted to prompt-only
    2. prompt-only rows: {prompt, ...} -> passed through
    3. code rows: {prompt, entry_point, test_cases, ...} -> passed through

    When `require_code_metadata=True`, rows must include `prompt`, `entry_point`,
    and `test_cases`, which are required by the code-execution reward functions.
    """
    if require_code_metadata:
        return build_code_dataset(
            raw_data=raw_data,
            path=path,
            dataset_name=dataset_name,
            split=split,
        )

    if path is not None:
        rows = _load_jsonl(path)
    elif raw_data is not None:
        rows = [dict(row) for row in raw_data]
    elif dataset_name is None:
        rows = [dict(row) for row in SAMPLE_CODE_TASKS]
    elif dataset_name in SUPPORTED_DATASETS:
        rows = _load_preference_records(dataset_name=dataset_name, split=split)
    else:
        from datasets import load_dataset

        dataset = load_dataset(dataset_name, split=split)
        rows = [dict(row) for row in dataset]

    if not rows:
        return HFDataset.from_list([])

    first = rows[0]
    if _is_code_like_row(first):
        return HFDataset.from_list([_validate_code_record(row) for row in rows])
    if _is_preference_row(first):
        return HFDataset.from_list(preference_to_prompt_records(rows))
    if _is_prompt_only_row(first):
        for row in rows:
            if "prompt" not in row or not isinstance(row["prompt"], str):
                raise TypeError("Prompt-only GRPO rows must contain a string 'prompt' field.")
        return HFDataset.from_list(rows)

    raise ValueError(
        "Unsupported GRPO dataset format. Expected prompt-only rows, preference rows, "
        "or code rows with entry_point/test_cases."
    )


def prompt_truncation_mode(dataset_name: Optional[str], has_code_metadata: bool) -> str:
    if has_code_metadata:
        return DATASET_TRUNCATION_MODE["code"]
    if dataset_name in DATASET_TRUNCATION_MODE:
        return DATASET_TRUNCATION_MODE[dataset_name]
    return "keep_start"


def truncate_prompt_text(
    prompt: str,
    tokenizer,
    max_prompt_tokens: Optional[int],
    *,
    strategy: str = "keep_start",
) -> str:
    """Token-level prompt truncation with either keep_start or keep_end."""
    if max_prompt_tokens is None:
        return prompt
    encoded = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if len(encoded) <= max_prompt_tokens:
        return prompt
    if strategy == "keep_end":
        encoded = encoded[-max_prompt_tokens:]
    else:
        encoded = encoded[:max_prompt_tokens]
    return tokenizer.decode(encoded, skip_special_tokens=True)


def truncate_prompt_dataset(
    dataset: HFDataset,
    tokenizer,
    max_prompt_tokens: Optional[int],
    *,
    strategy: str = "keep_start",
) -> HFDataset:
    """Apply token-level prompt truncation to a Hugging Face Dataset."""
    if max_prompt_tokens is None:
        return dataset

    def _map(row: Dict[str, Any]) -> Dict[str, Any]:
        updated = dict(row)
        updated["prompt"] = truncate_prompt_text(
            row["prompt"],
            tokenizer,
            max_prompt_tokens,
            strategy=strategy,
        )
        return updated

    return dataset.map(_map)
