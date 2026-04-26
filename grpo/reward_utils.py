"""Custom reward functions for code-generation GRPO.

Important note:
This is a lightweight local validator intended to help the team run the
pipeline end-to-end. It is *not* a hardened sandbox for arbitrary untrusted
code. For a serious benchmark or open-ended generation setting, run tests in an
isolated subprocess/container with resource limits.
"""

from __future__ import annotations

import ast
import builtins
import contextlib
import io
import re
import signal
import textwrap
from typing import Any, Dict, Iterable, List, Mapping, Optional

CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)

_ALLOWED_BUILTINS = {
    name: getattr(builtins, name)
    for name in [
        "abs",
        "all",
        "any",
        "bool",
        "dict",
        "enumerate",
        "float",
        "int",
        "isinstance",
        "len",
        "list",
        "map",
        "max",
        "min",
        "range",
        "set",
        "sorted",
        "str",
        "sum",
        "tuple",
        "zip",
    ]
}


class TimeoutErrorForReward(RuntimeError):
    """Raised when execution exceeds the configured time limit."""


@contextlib.contextmanager
def time_limit(seconds: Optional[int]):
    """Apply a soft time limit on Unix; no-op elsewhere."""
    if seconds is None or seconds <= 0:
        yield
        return

    if not hasattr(signal, "SIGALRM"):
        yield
        return

    def _handler(signum, frame):  # pragma: no cover - OS/timing dependent
        raise TimeoutErrorForReward(f"Execution exceeded {seconds} seconds.")

    previous = signal.signal(signal.SIGALRM, _handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


@contextlib.contextmanager
def suppress_io():
    """Silence stdout/stderr during candidate execution."""
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        yield


def extract_completion_text(completion: Any) -> str:
    """Normalize a GRPO completion into plain text.

    TRL custom rewards can receive either strings (standard format) or lists of
    messages (conversational format).
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        parts: List[str] = []
        for item in completion:
            if isinstance(item, dict) and "content" in item:
                parts.append(str(item["content"]))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return str(completion)


def strip_markdown_fences(text: str) -> str:
    """Extract the first fenced code block when present."""
    match = CODE_BLOCK_RE.search(text)
    if match:
        return textwrap.dedent(match.group(1)).strip()
    return text.strip()


def looks_like_code(text: str) -> bool:
    """Heuristic check for Python-looking outputs."""
    cleaned = strip_markdown_fences(text)
    patterns = [r"\bdef\b", r"\breturn\b", r":\n", r"lambda "]
    return any(re.search(pattern, cleaned) for pattern in patterns)


def is_valid_python(text: str) -> bool:
    """Return True when the completion parses as Python."""
    cleaned = strip_markdown_fences(text)
    try:
        ast.parse(cleaned)
        return True
    except SyntaxError:
        return False


def _safe_exec(code: str, entry_point: str) -> Any:
    """Compile code in a restricted global namespace and return the entry function."""
    cleaned = strip_markdown_fences(code)
    parsed = ast.parse(cleaned)

    for node in ast.walk(parsed):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            raise ValueError("Imports are disabled in the lightweight reward sandbox.")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"eval", "exec", "open", "compile", "input", "__import__"}:
                raise ValueError(f"Disallowed function used in candidate code: {node.func.id}")

    namespace: Dict[str, Any] = {"__builtins__": _ALLOWED_BUILTINS}
    with suppress_io():
        exec(compile(parsed, filename="<candidate>", mode="exec"), namespace, namespace)
    if entry_point not in namespace or not callable(namespace[entry_point]):
        raise ValueError(f"Entry point '{entry_point}' was not defined by the candidate.")
    return namespace[entry_point]


def run_unit_tests(
    code: str,
    entry_point: str,
    test_cases: Iterable[Mapping[str, Any]],
    *,
    timeout_seconds: Optional[int] = 2,
) -> float:
    """Run simple function-call tests and return the pass rate in [0, 1].

    Supported case formats:
    - {"input": [...], "expected": ...}
    - {"args": [...], "kwargs": {...}, "expected": ...}
    """
    cases = list(test_cases)
    if not cases:
        return 0.0

    with time_limit(timeout_seconds):
        fn = _safe_exec(code, entry_point)

    passed = 0
    for case in cases:
        args = case.get("args", case.get("input", []))
        kwargs = case.get("kwargs", {})
        if not isinstance(args, list):
            raise TypeError("Each test case 'args'/'input' field must be a list.")
        if not isinstance(kwargs, dict):
            raise TypeError("Each test case 'kwargs' field must be a dict.")
        expected = case.get("expected")
        try:
            with time_limit(timeout_seconds), suppress_io():
                result = fn(*args, **kwargs)
            if result == expected:
                passed += 1
        except Exception:
            continue
    return float(passed) / float(len(cases))


def _mask_non_coding_samples(
    rewards: List[Optional[float]],
    task: Optional[Iterable[Any]],
) -> List[Optional[float]]:
    if task is None:
        return rewards
    task_list = list(task)
    masked: List[Optional[float]] = []
    for reward, task_name in zip(rewards, task_list):
        if task_name is None or str(task_name).lower() == "coding":
            masked.append(reward)
        else:
            masked.append(None)
    return masked


def compile_reward_func(completions, task=None, **kwargs):
    """Reward syntactically valid Python completions with 1.0, else 0.0."""
    rewards = [1.0 if is_valid_python(extract_completion_text(c)) else 0.0 for c in completions]
    return _mask_non_coding_samples(rewards, task)


def format_reward_func(completions, task=None, **kwargs):
    """Reward outputs that at least look like Python code."""
    rewards = [1.0 if looks_like_code(extract_completion_text(c)) else 0.0 for c in completions]
    return _mask_non_coding_samples(rewards, task)


def unit_test_reward_func(
    completions,
    entry_point,
    test_cases,
    task=None,
    timeout_seconds: int = 2,
    **kwargs,
):
    """Reward candidates by the fraction of hidden tests they pass.

    Missing metadata returns `None` per example so the trainer can ignore those
    samples, which is useful for mixed-task or partially curated datasets.
    """
    rewards: List[Optional[float]] = []
    task_list = list(task) if task is not None else ["coding"] * len(completions)

    for completion, fn_name, tests, task_name in zip(completions, entry_point, test_cases, task_list):
        if task_name is not None and str(task_name).lower() != "coding":
            rewards.append(None)
            continue
        if fn_name is None or tests is None:
            rewards.append(None)
            continue
        try:
            score = run_unit_tests(
                extract_completion_text(completion),
                str(fn_name),
                tests,
                timeout_seconds=timeout_seconds,
            )
            rewards.append(score)
        except Exception:
            rewards.append(0.0)
    return rewards


DEFAULT_CODE_REWARD_WEIGHTS = [1.0, 0.2, 0.1]


def build_code_reward_bundle():
    """Return the default code reward functions and their suggested weights."""
    return [unit_test_reward_func, compile_reward_func, format_reward_func], list(
        DEFAULT_CODE_REWARD_WEIGHTS
    )
