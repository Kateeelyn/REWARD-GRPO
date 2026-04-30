import json
import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import PeftModel

BASE = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
RM_PATH = sys.argv[1] if len(sys.argv) > 1 else "./qwen-coder-rm-hh"
EVAL_FILE = sys.argv[2] if len(sys.argv) > 2 else "hh_eval_500.jsonl"
N_GEN = int(sys.argv[3]) if len(sys.argv) > 3 else 50

dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_rm():
    tok = AutoTokenizer.from_pretrained(RM_PATH if os.path.exists(os.path.join(RM_PATH, "tokenizer_config.json")) else BASE, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    if os.path.exists(os.path.join(RM_PATH, "adapter_config.json")):
        base = AutoModelForSequenceClassification.from_pretrained(
            BASE, num_labels=1, torch_dtype=dtype, trust_remote_code=True
        ).to(device).eval()
        base.config.pad_token_id = tok.pad_token_id
        rm = PeftModel.from_pretrained(base, RM_PATH).to(device).eval()
    else:
        rm = AutoModelForSequenceClassification.from_pretrained(
            RM_PATH, torch_dtype=dtype, trust_remote_code=True
        ).to(device).eval()
    return rm, tok

rm, rm_tok = load_rm()
pairs = [json.loads(line) for line in open(EVAL_FILE)]

correct = 0
chosen_scores = []
rejected_scores = []

for i, p in enumerate(pairs):
    c = rm_tok(p["prompt"] + p["chosen"], return_tensors="pt", truncation=True, max_length=1024).to(device)
    r = rm_tok(p["prompt"] + p["rejected"], return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        sc = rm(**c).logits.squeeze().item()
        sr = rm(**r).logits.squeeze().item()
    chosen_scores.append(sc)
    rejected_scores.append(sr)
    correct += int(sc > sr)
    if (i + 1) % 100 == 0:
        print(f"{i+1}/{len(pairs)} pairwise acc so far: {correct/(i+1):.3f}")

pairwise_acc = correct / len(pairs)
chosen_mean = sum(chosen_scores) / len(chosen_scores)
rejected_mean = sum(rejected_scores) / len(rejected_scores)

print(f"RM pairwise accuracy: {pairwise_acc:.4f}")
print(f"chosen mean:   {chosen_mean:+.4f}")
print(f"rejected mean: {rejected_mean:+.4f}")
print(f"margin mean:   {chosen_mean - rejected_mean:+.4f}")

pol_tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
if pol_tok.pad_token is None:
    pol_tok.pad_token = pol_tok.eos_token
policy = AutoModelForCausalLM.from_pretrained(
    BASE, torch_dtype=dtype, trust_remote_code=True
).to(device).eval()

gen_scores = []
gen_lengths = []
for p in pairs[:N_GEN]:
    ids = pol_tok(p["prompt"], return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        out = policy.generate(**ids, max_new_tokens=256, do_sample=False, pad_token_id=pol_tok.eos_token_id)
    new_tokens = out[0, ids.input_ids.shape[1]:]
    gen_lengths.append(new_tokens.shape[0])
    text = pol_tok.decode(new_tokens, skip_special_tokens=True)
    rm_in = rm_tok(p["prompt"] + text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        gen_scores.append(rm(**rm_in).logits.squeeze().item())

base_gen_mean = sum(gen_scores) / len(gen_scores)
base_gen_median = sorted(gen_scores)[len(gen_scores) // 2]

print(f"base generation RM mean:   {base_gen_mean:+.4f}")
print(f"base generation RM median: {base_gen_median:+.4f}")
print(f"chosen mean - base gen mean: {chosen_mean - base_gen_mean:+.4f}")
print(f"base generation mean length: {sum(gen_lengths)/len(gen_lengths):.1f}")

os.makedirs("results", exist_ok=True)
with open("results/rm_accuracy.json", "w") as f:
    json.dump({
        "rm_path": RM_PATH,
        "eval_file": EVAL_FILE,
        "n_pairs": len(pairs),
        "rm_accuracy_pairwise": pairwise_acc,
        "chosen_score_mean": chosen_mean,
        "rejected_score_mean": rejected_mean,
        "chosen_minus_rejected_mean": chosen_mean - rejected_mean,
        "base_policy_gen_score_mean": base_gen_mean,
        "base_policy_gen_score_median": base_gen_median,
        "chosen_minus_base_gen_mean": chosen_mean - base_gen_mean,
        "base_policy_gen_mean_length": sum(gen_lengths) / len(gen_lengths),
        "n_base_generation_probes": N_GEN,
    }, f, indent=2)
