import argparse
import json
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


def load_model(model_path, base_model_name=None):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    tok = AutoTokenizer.from_pretrained(
        base_model_name if base_model_name else model_path,
        trust_remote_code=True,
    )
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if base_model_name:
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).cuda().eval()
        model = PeftModel.from_pretrained(base, model_path).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            trust_remote_code=True,
        ).cuda().eval()

    return model, tok


def clean_completion(text: str) -> str:
    text = text.replace("```python", "").replace("```", "").strip()

    prefixes = [
        "Here is the code:",
        "Here’s the code:",
        "Sure, here's the code:",
        "Sure, here is the code:",
    ]
    for p in prefixes:
        if text.startswith(p):
            text = text[len(p):].strip()

    for marker in ["\nif __name__ ==", "\nif __name__=="]:
        if marker in text:
            text = text.split(marker)[0].rstrip()

    matches = list(re.finditer(r"(?m)^def |^class ", text))
    if len(matches) >= 2:
        text = text[:matches[1].start()].rstrip()

    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base_model", default=None)
    args = ap.parse_args()

    from human_eval.data import read_problems

    model, tok = load_model(args.model, args.base_model)
    problems = read_problems()

    with open(args.out, "w", encoding="utf-8") as f:
        for task_id, prob in problems.items():
            prompt = prob["prompt"] + "\n\nOnly output Python code, no explanation.\n"
            inputs = tok(prompt, return_tensors="pt").to("cuda")

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=tok.eos_token_id,
                    eos_token_id=tok.eos_token_id,
                    use_cache=True,
                )

            completion = tok.decode(
                outputs[0, inputs.input_ids.shape[1]:],
                skip_special_tokens=True,
            )
            completion = clean_completion(completion)

            f.write(json.dumps({
                "task_id": task_id,
                "completion": completion
            }) + "\n")

    print(f"Wrote samples to {args.out}")


if __name__ == "__main__":
    main()
