import json
import os
import statistics
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


EVAL_FILE = "hh_eval_500.jsonl"

MODELS = {
    "Base": "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    "lr1e6": "qwen-coder-grpo-hh-v2",
    "lr2e5": "qwen-coder-grpo-hh-lr2e5",
}


def load_policy(model_path, dtype):
    tok_path = model_path
    adapter_cfg = os.path.join(model_path, "adapter_config.json") if os.path.isdir(model_path) else ""
    is_peft = os.path.exists(adapter_cfg)

    if is_peft:
        with open(adapter_cfg, "r", encoding="utf-8") as f:
            base_name = json.load(f)["base_model_name_or_path"]

        tok_path = base_name

        base = AutoModelForCausalLM.from_pretrained(
            base_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).cuda().eval()

        model = PeftModel.from_pretrained(base, model_path)
        model = model.merge_and_unload().eval()

    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).cuda().eval()

    tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    return model, tok


@torch.no_grad()
def sequence_logprob(model, tokenizer, prompt, response):
    # EXACT SAME LOGIC AS eval_margin_delta.py
    full = prompt + response
    full_ids = tokenizer(full, return_tensors="pt").input_ids.to(model.device)
    prompt_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    prompt_len = prompt_ids.shape[1]

    logits = model(full_ids).logits
    log_probs = F.log_softmax(logits[:, :-1].float(), dim=-1)
    targets = full_ids[:, 1:]

    token_lp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    response_lp = token_lp[:, prompt_len - 1:]

    return response_lp.sum().item()


def summarize(xs):
    return {
        "mean": sum(xs) / len(xs),
        "median": statistics.median(xs),
        "std": statistics.pstdev(xs),
        "min": min(xs),
        "max": max(xs),
    }


def eval_model(name, model_path, pairs, dtype):
    print(f"\nLoading {name}: {model_path}", flush=True)

    model, tok = load_policy(model_path, dtype)

    chosen_logps = []
    rejected_logps = []
    margins = []

    for i, p in enumerate(pairs):
        lp_c = sequence_logprob(model, tok, p["prompt"], p["chosen"])
        lp_r = sequence_logprob(model, tok, p["prompt"], p["rejected"])

        chosen_logps.append(lp_c)
        rejected_logps.append(lp_r)
        margins.append(lp_c - lp_r)

        if (i + 1) % 50 == 0 or (i + 1) == len(pairs):
            print(f"{name}: {i + 1}/{len(pairs)} pairs done", flush=True)

    # 这句非常重要：如果没跑满 500，会直接报错，不会假装成功
    assert len(margins) == len(pairs), f"{name} only processed {len(margins)} / {len(pairs)}"

    del model
    torch.cuda.empty_cache()

    return {
        "avg_logp_chosen": sum(chosen_logps) / len(chosen_logps),
        "avg_logp_rejected": sum(rejected_logps) / len(rejected_logps),
        "avg_margin": sum(margins) / len(margins),
        "preference_accuracy": sum(x > 0 for x in margins) / len(margins),
        "chosen_logp": summarize(chosen_logps),
        "rejected_logp": summarize(rejected_logps),
        "margin": summarize(margins),
        "n": len(margins),
    }, margins


def print_row(metric, base, lr1e6, lr2e5):
    print(f"| {metric:<20} | {base:>10} | {lr1e6:>10} | {lr2e5:>10} |")


def main():
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    with open(EVAL_FILE, "r", encoding="utf-8") as f:
        pairs = [json.loads(line) for line in f]

    print(f"Loaded {len(pairs)} pairs from {EVAL_FILE}", flush=True)

    if len(pairs) != 500:
        print(f"WARNING: eval file has {len(pairs)} pairs, not 500", flush=True)

    results = {}
    all_margins = {}

    for name, model_path in MODELS.items():
        results[name], all_margins[name] = eval_model(name, model_path, pairs, dtype)

    for name in ["lr1e6", "lr2e5"]:
        deltas = [t - b for t, b in zip(all_margins[name], all_margins["Base"])]
        results[name]["delta_vs_base"] = summarize(deltas)
        results[name]["fraction_delta_positive"] = sum(x > 0 for x in deltas) / len(deltas)

    results["Base"]["delta_vs_base"] = None
    results["Base"]["fraction_delta_positive"] = None

    print("\nAverage log-prob comparison")
    print("+----------------------+------------+------------+------------+")
    print("| Metric               |       Base |      lr1e6 |      lr2e5 |")
    print("+----------------------+------------+------------+------------+")

    print_row(
        "avg log P(chosen)",
        f"{results['Base']['avg_logp_chosen']:.2f}",
        f"{results['lr1e6']['avg_logp_chosen']:.2f}",
        f"{results['lr2e5']['avg_logp_chosen']:.2f}",
    )

    print_row(
        "avg log P(rejected)",
        f"{results['Base']['avg_logp_rejected']:.2f}",
        f"{results['lr1e6']['avg_logp_rejected']:.2f}",
        f"{results['lr2e5']['avg_logp_rejected']:.2f}",
    )

    print_row(
        "avg margin",
        f"{results['Base']['avg_margin']:.2f}",
        f"{results['lr1e6']['avg_margin']:.2f}",
        f"{results['lr2e5']['avg_margin']:.2f}",
    )

    print_row(
        "delta vs base",
        "—",
        f"{results['lr1e6']['delta_vs_base']['mean']:+.3f}",
        f"{results['lr2e5']['delta_vs_base']['mean']:+.3f}",
    )

    print_row(
        "pref accuracy",
        f"{results['Base']['preference_accuracy']:.3f}",
        f"{results['lr1e6']['preference_accuracy']:.3f}",
        f"{results['lr2e5']['preference_accuracy']:.3f}",
    )

    print("+----------------------+------------+------------+------------+")

    os.makedirs("results", exist_ok=True)

    with open("results/avg_logprob_comparison_STRICT.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open("results/avg_logprob_comparison_STRICT.txt", "w", encoding="utf-8") as f:
        f.write("Average log-prob comparison\n")
        f.write("+----------------------+------------+------------+------------+\n")
        f.write("| Metric               |       Base |      lr1e6 |      lr2e5 |\n")
        f.write("+----------------------+------------+------------+------------+\n")
        f.write(f"| {'avg log P(chosen)':<20} | {results['Base']['avg_logp_chosen']:>10.2f} | {results['lr1e6']['avg_logp_chosen']:>10.2f} | {results['lr2e5']['avg_logp_chosen']:>10.2f} |\n")
        f.write(f"| {'avg log P(rejected)':<20} | {results['Base']['avg_logp_rejected']:>10.2f} | {results['lr1e6']['avg_logp_rejected']:>10.2f} | {results['lr2e5']['avg_logp_rejected']:>10.2f} |\n")
        f.write(f"| {'avg margin':<20} | {results['Base']['avg_margin']:>10.2f} | {results['lr1e6']['avg_margin']:>10.2f} | {results['lr2e5']['avg_margin']:>10.2f} |\n")
        f.write(f"| {'delta vs base':<20} | {'—':>10} | {results['lr1e6']['delta_vs_base']['mean']:>+10.3f} | {results['lr2e5']['delta_vs_base']['mean']:>+10.3f} |\n")
        f.write(f"| {'pref accuracy':<20} | {results['Base']['preference_accuracy']:>10.3f} | {results['lr1e6']['preference_accuracy']:>10.3f} | {results['lr2e5']['preference_accuracy']:>10.3f} |\n")
        f.write("+----------------------+------------+------------+------------+\n")

    print("\nSaved to:")
    print("  results/avg_logprob_comparison_STRICT.json")
    print("  results/avg_logprob_comparison_STRICT.txt")


if __name__ == "__main__":
    main()
