from grpo.data_utils import (
    SAMPLE_CODE_TASKS,
    SAMPLE_PREFERENCE_DATA,
    build_code_dataset,
    build_grpo_dataset,
    build_preference_dataset,
    preference_to_prompt_records,
    prompt_truncation_mode,
    truncate_prompt_text,
)


class DummyTokenizer:
    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": text.split()}

    def decode(self, tokens, skip_special_tokens=True):
        return " ".join(tokens)


def test_build_preference_dataset_from_sample_rows():
    ds = build_preference_dataset(raw_data=SAMPLE_PREFERENCE_DATA)
    assert len(ds) == 3
    assert set(ds.column_names) == {"prompt", "chosen", "rejected"}


def test_preference_to_prompt_records_keeps_prompt_and_gold_completion():
    rows = preference_to_prompt_records(SAMPLE_PREFERENCE_DATA)
    assert len(rows) == 3
    assert set(rows[0].keys()) == {"prompt", "gold_completion"}


def test_build_code_dataset_from_sample_rows():
    ds = build_code_dataset(raw_data=SAMPLE_CODE_TASKS)
    assert len(ds) == 3
    assert {"prompt", "entry_point", "test_cases"}.issubset(set(ds.column_names))


def test_build_grpo_dataset_accepts_preference_rows():
    ds = build_grpo_dataset(raw_data=SAMPLE_PREFERENCE_DATA)
    assert len(ds) == 3
    assert set(ds.column_names) == {"prompt", "gold_completion"}


def test_build_grpo_dataset_accepts_code_rows():
    ds = build_grpo_dataset(raw_data=SAMPLE_CODE_TASKS)
    assert len(ds) == 3
    assert {"prompt", "entry_point", "test_cases"}.issubset(set(ds.column_names))


def test_prompt_truncation_mode_matches_dataset_type():
    assert prompt_truncation_mode("hh", False) == "keep_end"
    assert prompt_truncation_mode("shp", False) == "keep_start"
    assert prompt_truncation_mode(None, True) == "keep_start"


def test_truncate_prompt_text_keep_start_and_keep_end():
    tokenizer = DummyTokenizer()
    prompt = "a b c d e"
    assert truncate_prompt_text(prompt, tokenizer, 3, strategy="keep_start") == "a b c"
    assert truncate_prompt_text(prompt, tokenizer, 3, strategy="keep_end") == "c d e"
