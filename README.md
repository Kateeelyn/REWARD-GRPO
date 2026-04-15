# GRPO Training for Code Generation

Base model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`  
Comparison: This repo implements the **GRPO arm** of a two-method RLHF study (**Reward Model + GRPO vs. DPO**) for a course project.

* * *

## Repository Structure

```text
GRPO/
├── grpo/
│   ├── __init__.py          # Package metadata
│   ├── data_utils.py        # Dataset loading and normalization
│   ├── model_utils.py       # Policy / reward model loading (LoRA, GPU placement)
│   ├── reward_model.py      # Reward-model training entry point
│   ├── reward_utils.py      # Reward functions for code tasks
│   ├── train_grpo.py        # GRPO training entry point
│   └── smoke_test.py        # CPU-only smoke test
├── tests/
│   ├── test_data_utils.py   # Dataset utility tests
│   └── test_reward_utils.py # Reward utility tests
├── README.md
├── requirements.txt
└── setup_env.sh
```

* * *

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU recommended for full training
- Internet access to download models and datasets from Hugging Face

### Install

```bash
bash setup_env.sh
source venv/bin/activate
```

`setup_env.sh` creates a virtual environment and installs all dependencies from `requirements.txt`.

### Manual install

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Recommended for long runs

For cloud training, use `tmux` so training continues even if your laptop sleeps or your SSH session disconnects:

```bash
tmux new -s grpo
```

Detach without stopping the job:

```text
Ctrl+b, then d
```

Re-attach later:

```bash
tmux attach -t grpo
```

* * *

## Verification

### Smoke test (CPU only)

Run this first to verify the basic pipeline before touching a GPU:

```bash
python -m grpo.smoke_test
```

This checks that:

1. data loading works
2. reward utilities do not crash
3. sample code tasks can be normalized and scored

### Test suite (CPU only)

```bash
pytest tests -v
```

* * *

## How This Differs from DPO

This GRPO package follows a **two-stage pipeline**:

1. train a **reward model** from preference pairs (`prompt`, `chosen`, `rejected`)
2. use that reward model during **GRPO policy optimization**

This is different from DPO, which directly optimizes the policy from preference pairs without first training a separate reward model.

In this repo, `grpo/train_grpo.py` supports multiple reward modes:

- `rm` — reward-model reward only
- `code` — executable code rewards only
- `hybrid` — reward-model reward + code rewards
- `auto` — choose a default based on available dataset columns

If you run `reward_mode rm` or `reward_mode hybrid`, you should train a reward model first.

* * *

## Quick Start

This repository includes a lightweight quick-start path so that the full GRPO pipeline can be validated before scaling to larger datasets.

### 1. Create a tiny preference dataset for reward-model training

```bash
cat > quickstart_prefs.jsonl <<'EOF'
{"prompt": "Write a Python function add_one(x).", "chosen": "def add_one(x):\n    return x + 1", "rejected": "def add_one(x):\n    return x - 1"}
{"prompt": "Write a Python function square(x).", "chosen": "def square(x):\n    return x * x", "rejected": "def square(x):\n    return x + x"}
{"prompt": "Write a Python function is_even(x).", "chosen": "def is_even(x):\n    return x % 2 == 0", "rejected": "def is_even(x):\n    return x % 2 == 1"}
EOF
```

### 2. Create a tiny prompt-only dataset for GRPO

```bash
cat > quickstart_prompts.jsonl <<'EOF'
{"prompt": "Write a Python function add_one(x). Only output Python code."}
{"prompt": "Write a Python function square(x). Only output Python code."}
{"prompt": "Write a Python function is_even(x). Only output Python code."}
EOF
```

### 3. Train a quick-start reward model

```bash
python -u -m grpo.reward_model \
  --dataset_path quickstart_prefs.jsonl \
  --output_dir qwen-coder-rm-quickstart \
  --epochs 1
```

### 4. Run quick-start GRPO

```bash
python -u -m grpo.train_grpo \
  --dataset_path quickstart_prompts.jsonl \
  --reward_mode rm \
  --reward_model_path qwen-coder-rm-quickstart \
  --output_dir qwen-coder-grpo-quickstart \
  --epochs 1 \
  --num_generations 4 \
  --max_steps 5
```

The quick-start path is intended for **end-to-end validation**, not for strong final performance.

* * *

## Training

### Train a reward model on Anthropic Helpful-Harmless (HH)

```bash
python -u -m grpo.reward_model \
  --dataset_name hh \
  --output_dir qwen-coder-rm-hh \
  --epochs 1
```

### Train GRPO on HH using reward-model rewards

Train the reward model first, then run GRPO:

```bash
python -u -m grpo.train_grpo \
  --dataset_name hh \
  --reward_mode rm \
  --reward_model_path qwen-coder-rm-hh \
  --output_dir qwen-coder-grpo-hh \
  --epochs 1 \
  --num_generations 4 \
  --max_steps 100
```

`--max_steps 100` is a good first sanity run on a cloud GPU. Increase it later if the run is stable and the runtime is acceptable.

### Train GRPO with code-only rewards

```bash
python -u -m grpo.train_grpo \
  --dataset_path code_tasks.jsonl \
  --reward_mode code \
  --output_dir qwen-coder-grpo-code \
  --epochs 1 \
  --num_generations 4 \
  --max_steps 100
```

### Train GRPO with hybrid rewards

```bash
python -u -m grpo.train_grpo \
  --dataset_path code_tasks.jsonl \
  --reward_mode hybrid \
  --reward_model_path qwen-coder-rm-hh \
  --output_dir qwen-coder-grpo-hybrid \
  --epochs 1 \
  --num_generations 4 \
  --max_steps 100
```

* * *

## Supported Data Formats

### Preference data for reward-model training

```json
{"prompt": "Write a Python function add_one(x).", "chosen": "def add_one(x):\n    return x + 1", "rejected": "def add_one(x):\n    return x - 1"}
```

### Prompt-only data for GRPO

```json
{"prompt": "Write a Python function add_one(x). Only output Python code."}
```

### Code-task data for code rewards

```json
{
  "prompt": "Write a Python function add_one(x) that returns x + 1. Only output Python code.",
  "entry_point": "add_one",
  "test_cases": [
    {"input": [1], "expected": 2},
    {"input": [0], "expected": 1}
  ],
  "task": "coding"
}
```

* * *

## Full Pipeline Recommendation

For a clean end-to-end demonstration:

```bash
bash setup_env.sh
source venv/bin/activate
python -m grpo.smoke_test
python -u -m grpo.reward_model --dataset_path quickstart_prefs.jsonl --output_dir qwen-coder-rm-quickstart --epochs 1
python -u -m grpo.train_grpo --dataset_path quickstart_prompts.jsonl --reward_mode rm --reward_model_path qwen-coder-rm-quickstart --output_dir qwen-coder-grpo-quickstart --epochs 1 --num_generations 4 --max_steps 5
```

For a more realistic experiment:

```bash
python -u -m grpo.reward_model --dataset_name hh --output_dir qwen-coder-rm-hh --epochs 1
python -u -m grpo.train_grpo --dataset_name hh --reward_mode rm --reward_model_path qwen-coder-rm-hh --output_dir qwen-coder-grpo-hh --epochs 1 --num_generations 4 --max_steps 100
```

* * *

## Common Pitfalls

### 1. GRPO tokenizer padding differs from DPO

This implementation uses:

- policy tokenizer: left padding
- reward tokenizer: right padding

Do not blindly copy DPO tokenizer settings into GRPO.

### 2. `num_generations` must align with the effective batch

The effective batch size

```text
WORLD_SIZE * batch_size * grad_accum
```

should be divisible by `num_generations`.

### 3. Reward-related dataset columns must not be dropped

GRPO training may need fields such as `test_cases` and `entry_point`, so the training pipeline keeps extra columns instead of aggressively pruning them.

### 4. The code executor is lightweight

`reward_utils.py` currently uses a lightweight local execution path for experimentation. It is suitable for small-scale testing, but not a hardened sandbox for unrestricted large-scale code execution.

* * *

## Suggested Project Summary

A concise way to describe this repository in a report or presentation:

> We implement a two-stage GRPO pipeline for code generation: first a reward model is trained from preference data, then a policy model is optimized with GRPO using reward-model rewards, code-based rewards, or a hybrid of both. A lightweight quick-start path is included to validate the pipeline end-to-end before scaling to larger datasets such as HH or curated code tasks.
