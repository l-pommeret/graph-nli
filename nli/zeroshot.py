from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


MODEL = "Qwen/Qwen3.5-0.8B-Base"
LABELS = ["entailment", "neutral", "contradiction"]
TEMPLATE = ("{ctx}\n\nQuestion: is the hypothesis entailed by the premise, "
            "neutral, or a contradiction?\nAnswer: {label}")


def main() -> None:
    ap = argparse.ArgumentParser(description='zero-shot label-likelihood baseline')
    ap.add_argument("--inputs", nargs="+", required=True, help="jsonl {text,label}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    print(f"[zeroshot] {args.model}", flush=True)

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model, dtype="auto", max_model_len=2048,
              gpu_memory_utilization=0.90, enforce_eager=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=0)

    results = {}
    for path in args.inputs:
        rows = [json.loads(l) for l in open(path)][: args.limit]
        prompts, n_label_toks = [], []
        for r in rows:
            ctx = r["text"][:6000]
            base_len = len(tok(TEMPLATE.format(ctx=ctx, label="")).input_ids)
            for lab in LABELS:
                full = TEMPLATE.format(ctx=ctx, label=lab)
                prompts.append(full)
                n_label_toks.append(len(tok(full).input_ids) - base_len)
        outs = llm.generate(prompts, params)
        correct = 0
        for i, r in enumerate(rows):
            scores = []
            for j in range(3):
                o = outs[3 * i + j]
                k = max(n_label_toks[3 * i + j], 1)
                lps = [next(iter(d.values())).logprob
                       for d in o.prompt_logprobs[-k:] if d]
                scores.append(sum(lps) / len(lps))
            if scores.index(max(scores)) == r["label"]:
                correct += 1
        acc = correct / len(rows)
        results[Path(path).stem] = {"accuracy": round(acc, 4), "n": len(rows)}
        print(f"[zeroshot] {Path(path).stem}: {acc:.4f} ({len(rows)} ex.)", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"model": args.model, "method": "label log-likelihood", **results}, indent=2))


if __name__ == "__main__":
    main()
