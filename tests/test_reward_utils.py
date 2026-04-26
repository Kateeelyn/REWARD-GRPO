from grpo.reward_utils import (
    build_code_reward_bundle,
    compile_reward_func,
    format_reward_func,
    is_valid_python,
    looks_like_code,
    run_unit_tests,
    strip_markdown_fences,
    unit_test_reward_func,
)


def test_strip_markdown_fences_extracts_python_block():
    text = """Here is code:\n```python\ndef add_one(x):\n    return x + 1\n```"""
    assert strip_markdown_fences(text) == "def add_one(x):\n    return x + 1"


def test_is_valid_python_and_looks_like_code():
    code = "def add_one(x):\n    return x + 1\n"
    assert is_valid_python(code)
    assert looks_like_code(code)
    assert not is_valid_python("def broken(:")


def test_run_unit_tests_scores_good_solution():
    code = "def add_one(x):\n    return x + 1\n"
    score = run_unit_tests(
        code,
        "add_one",
        [{"input": [1], "expected": 2}, {"input": [0], "expected": 1}],
    )
    assert score == 1.0


def test_run_unit_tests_scores_bad_solution_lower():
    code = "def add_one(x):\n    return x - 1\n"
    score = run_unit_tests(
        code,
        "add_one",
        [{"input": [1], "expected": 2}, {"input": [0], "expected": 1}],
    )
    assert score < 1.0


def test_reward_functions_work_on_standard_format_strings():
    completions = ["def add_one(x):\n    return x + 1\n"]
    assert compile_reward_func(completions) == [1.0]
    assert format_reward_func(completions) == [1.0]
    assert unit_test_reward_func(
        completions,
        entry_point=["add_one"],
        test_cases=[[{"input": [1], "expected": 2}]],
    ) == [1.0]


def test_non_coding_tasks_return_none_for_unit_test_reward():
    rewards = unit_test_reward_func(
        ["def foo(x):\n    return x\n"],
        entry_point=["foo"],
        test_cases=[[{"input": [1], "expected": 1}]],
        task=["math"],
    )
    assert rewards == [None]


def test_default_code_reward_bundle_has_three_functions_and_weights():
    funcs, weights = build_code_reward_bundle()
    assert len(funcs) == 3
    assert weights == [1.0, 0.2, 0.1]
