import argparse
import json
import statistics
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import PeftModel

def load_pairs(path):
    pairs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pairs.append(json.loads(line))
    return pairs

@torch.no_grad()
def sequence_logprob(model, tokenizer, prompt: str, response: str) -> float:
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

@torch.no_grad()
def preference_accuracy(model, tokenizer, pairs):
    correct = 0
    for p in pairs:
        lp_c = sequence_logprob(model, tokenizer, p["prompt"], p["chosen"])
        lp_r = sequence_logprob(model, tokenizer, p["prompt"], p["rejected"])
        if lp_c > lp_r:
            correct += 1
    return correct / len(pairs)

def load_policy(model_path, dtype):
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).cuda().eval()
    return model, tok

def load_rm(base_model_name, rm_path, dtype):
    tok = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    base = AutoModelForSequenceClassification.from_pretrained(
        base_model_name,
        num_labels=1,
        torch_dtype=dtype,
        trust_remote_code=True,
    ).cuda().eval()

    # Support either full checkpoint or LoRA adapter checkpoint
    try:
        rm = PeftModel.from_pretrained(base, rm_path).eval()
    except Exception:
        rm = AutoModelForSequenceClassification.from_pretrained(
            rm_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).cuda().eval()

    return rm, tok

@torch.no_grad()
def rm_score_of_generations(policy, policy_tok, rm, rm_tok, prompts):
    scores = []
    lengths = []

    for prompt in prompts:
        ids = policy_tok(prompt, return_tensors="pt", truncation=True, max_length=256).to("cuda")
        gen = policy.generate(
            **ids,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=policy_tok.eos_token_id,
        )
        new_tokens = gen[0, ids.input_ids.shape[1]:]
        lengths.append(new_tokens.shape[0])

        text = policy_tok.decode(new_tokens, skip_special_tokens=True)
        rm_in = rm_tok(prompt + text, return_tensors="pt", truncation=True, max_length=1024).to("cuda")
        scores.append(rm(**rm_in).logits.squeeze().item())

    return {
        "mean": sum(scores) / len(scores),
        "median": statistics.median(scores),
        "mean_length": sum(lengths) / len(lengths),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--eval_file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rm_path", default=None)
    ap.add_argument("--rm_base_model", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    args = ap.parse_args()

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    pairs = load_pairs(args.eval_file)
    policy, policy_tok = load_policy(args.model, dtype)

    results = {
        "model": args.model,
        "eval_file": args.eval_file,
        "num_pairs": len(pairs),
        "preference_accuracy": preference_accuracy(policy, policy_tok, pairs),
    }

    if args.rm_path:
        rm, rm_tok = load_rm(args.rm_base_model, args.rm_path, dtype)
        prompts = [p["prompt"] for p in pairs]
        rm_stats = rm_score_of_generations(policy, policy_tok, rm, rm_tok, prompts)
        results["rm_score_mean"] = rm_stats["mean"]
        results["rm_score_median"] = rm_stats["median"]
        results["mean_response_length"] = rm_stats["mean_length"]

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
