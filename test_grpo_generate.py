import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
ADAPTER_PATH = "/home/kashengping/GRPO/qwen-coder-grpo-hh"

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    trust_remote_code=True,
).to(device)

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

prompts = [
    "Write a Python function add_one(x) that returns x + 1. Only output Python code.",
    "Write a Python function is_even(x) that returns True if x is even. Only output Python code.",
    "Write a Python function factorial(n) using recursion. Only output Python code."
]

for i, prompt in enumerate(prompts, 1):
    print("=" * 80)
    print(f"Prompt {i}: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nGenerated:\n")
    print(text)
    print()
