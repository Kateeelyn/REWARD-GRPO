# GRPO 部分单独文件包说明

这份文件包是按照你现在的 DPO 仓库风格单独拆出来的 **GRPO 部分**，可以直接作为一个独立小包使用，也可以把里面的 `grpo/` 和 `tests/` 合并进你们现有仓库。

我做这份包时，重点是满足你现在这几个需求：

1. **先把 GRPO pipeline 跑通**，哪怕先用很小的样例。
2. **结构尽量贴近你现有 DPO repo**，方便老师/组员对照看。
3. **后面能继续扩展到 dataset curation 和代码任务奖励**，不是只写一段空壳说明。

---

## 1. 文件结构

```text
GRPO/
├── grpo/
│   ├── __init__.py
│   ├── data_utils.py
│   ├── model_utils.py
│   ├── reward_model.py
│   ├── reward_utils.py
│   ├── train_grpo.py
│   └── smoke_test.py
├── tests/
│   ├── test_data_utils.py
│   └── test_reward_utils.py
├── README_CN.md
├── requirements.txt
└── setup_env.sh
```

---

## 2. 每个文件是干什么的

### `grpo/reward_model.py`
训练 **Reward Model**。

输入数据格式是：

```json
{"prompt": "...", "chosen": "...", "rejected": "..."}
```

这一步是为了支持你们项目里 **RM + GRPO vs DPO** 的对比逻辑。

### `grpo/train_grpo.py`
训练 **GRPO policy model**。

我给你做了 4 种 reward 模式：

- `auto`：自动判断
- `rm`：只用 reward model 打分
- `code`：只用代码执行奖励
- `hybrid`：reward model + 代码奖励一起用

这样你现在可以先跑通 pipeline，后面再接真正的 code dataset。

### `grpo/reward_utils.py`
放自定义 reward function。

默认有三种代码奖励：

- `unit_test_reward_func`：按测试通过率给分
- `compile_reward_func`：Python 语法能否通过
- `format_reward_func`：输出看起来像不像代码

默认建议权重是：

```text
[1.0, 0.2, 0.1]
```

也就是：**单元测试 > 语法 > 格式**。

### `grpo/data_utils.py`
数据加载和格式转换。

支持三类数据源：

1. **偏好数据**：`prompt/chosen/rejected`
2. **prompt-only 数据**：只有 `prompt`
3. **代码任务数据**：`prompt/entry_point/test_cases`

这意味着：

- 你现在可以直接拿偏好数据先训练 reward model
- 之后可以把 dataset curation 出来的代码任务数据直接接到 GRPO

### `grpo/smoke_test.py`
CPU 就能跑的小测试，不下模型也能测。

用途是先确认：

- 数据加载没问题
- reward 函数没写崩
- 样例代码能拿到合理 reward

---

## 3. 你现在最推荐的使用顺序

### 第一步：先做纯 Python 冒烟测试

```bash
python -m grpo.smoke_test
```

### 第二步：训练 reward model

如果你要和 DPO 做最标准的对照，先跑这个：

```bash
python -m grpo.reward_model \
  --dataset_name hh \
  --output_dir qwen-coder-rm-hh \
  --epochs 1
```

如果你已经有本地 JSONL：

```bash
python -m grpo.reward_model \
  --dataset_path your_preference_data.jsonl \
  --output_dir qwen-coder-rm-local \
  --epochs 1
```

### 第三步：跑 GRPO

#### 方案 A：只用 reward model

适合你现在想先把 **RM + GRPO** 路线完整跑通。

```bash
python -m grpo.train_grpo \
  --dataset_name hh \
  --reward_mode rm \
  --reward_model_path qwen-coder-rm-hh \
  --output_dir qwen-coder-grpo-hh \
  --epochs 1 \
  --num_generations 4
```

#### 方案 B：只用代码奖励

适合你后面做完 code dataset curation 之后直接接上。

```bash
python -m grpo.train_grpo \
  --dataset_path code_tasks.jsonl \
  --reward_mode code \
  --output_dir qwen-coder-grpo-code \
  --epochs 1 \
  --num_generations 4
```

#### 方案 C：hybrid

如果你的数据里既有 `test_cases`，你又已经有 reward model，可以一起用：

```bash
python -m grpo.train_grpo \
  --dataset_path code_tasks.jsonl \
  --reward_mode hybrid \
  --reward_model_path qwen-coder-rm-hh \
  --output_dir qwen-coder-grpo-hybrid \
  --epochs 1 \
  --num_generations 4
```

---

## 4. 本地 JSONL 数据格式示例

### 4.1 reward model 用的数据

```json
{"prompt": "Write a Python function add_one(x)", "chosen": "def add_one(x):\n    return x + 1", "rejected": "def add_one(x):\n    return x - 1"}
```

### 4.2 GRPO 代码任务数据

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

---

## 5. 这份包和你们现有 DPO repo 的对应关系

你们现在的 DPO 仓库大概是这个风格：

- 一个主入口脚本
- `data_utils.py`
- `model_utils.py`
- `tests/`
- `requirements.txt`
- `setup_env.sh`

我这次给你的 GRPO 部分也按这个风格写了，所以你可以：

### 方式 1：单独作为新目录

把整个 `GRPO/` 放在仓库旁边单独跑。

### 方式 2：并进现有仓库

把下面这些合并进去：

- `grpo/`
- `tests/test_data_utils.py`
- `tests/test_reward_utils.py`
- 视情况更新 `requirements.txt`

---

## 6. 和 DPO 最不一样、最容易踩坑的地方

### 坑 1：GRPO 的 tokenizer 要用 left padding

我已经在 `train_grpo.py` 里固定成：

- policy tokenizer：`padding_side="left"`
- reward tokenizer：`padding_side="right"`

不要直接把 DPO 里 tokenizer 的 right padding 整段照搬过来。

### 坑 2：`num_generations` 要和有效 batch 对齐

脚本里加了检查：

```text
WORLD_SIZE * batch_size * grad_accum
```

必须能被 `num_generations` 整除，不然 GRPOTrainer 会很容易出问题。

### 坑 3：如果 reward function 需要额外列，不能把列裁掉

我在 GRPO 训练里把：

```python
remove_unused_columns=False
```

直接固定了。因为 `test_cases`、`entry_point` 这些列都要传进 reward function。

### 坑 4：现在这个代码执行器只是轻量版，不是强安全沙箱

`reward_utils.py` 里我做的是 **轻量本地执行器**，适合先跑通 pipeline、做小规模实验。

如果你们后面真要在开放生成场景里跑大量代码样本，建议把执行放到：

- 独立 subprocess
- Docker / sandbox
- 带 CPU / 内存 / 时间限制的环境

---

## 7. 你现在最适合怎么汇报/写到实验计划里

你可以直接把 GRPO 部分概括成：

> 我们实现了一个两阶段 GRPO pipeline：先用偏好数据训练 reward model，再对 policy model 进行在线 GRPO 优化；同时预留了基于 hidden unit tests 的代码奖励接口，便于后续 dataset curation 完成后切换到代码执行型 reward。

这样写的好处是：

- 和你们的 DPO baseline 对齐
- 和后续 dataset curation 衔接自然
- 说明你现在写的不是一次性脚本，而是能继续扩展的 pipeline

---

## 8. 推荐你先跑的最小命令

如果你现在就是想“先有一个能交代、能展示、能继续改”的版本，我建议顺序就是：

```bash
bash setup_env.sh
source venv/bin/activate
python -m grpo.smoke_test
python -m grpo.reward_model --dataset_name hh --output_dir qwen-coder-rm-hh --epochs 1
python -m grpo.train_grpo --dataset_name hh --reward_mode rm --reward_model_path qwen-coder-rm-hh --output_dir qwen-coder-grpo-hh --epochs 1 --num_generations 4
```

如果你后面已经把代码题数据整理成 JSONL，再把最后一条换成：

```bash
python -m grpo.train_grpo --dataset_path code_tasks.jsonl --reward_mode code --output_dir qwen-coder-grpo-code --epochs 1 --num_generations 4
```

---

## 9. 额外说明

这份包里我已经放了：

- 可运行的代码骨架
- 冒烟测试
- pytest 测试
- 中文使用说明

所以你现在可以直接把它当成你负责的 **GRPO pipeline 单独部分** 提出去。
