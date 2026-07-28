from __future__ import annotations

import argparse
import json
from pathlib import Path

from nli.extract import load_llm

MODEL = "ANON/propositionneur-v2-large"
MAX_PROPS = 16

SCHEMA = {"type": "array", "items": {"type": "string", "maxLength": 300},
          "minItems": 1, "maxItems": 24}


def messages_for(text: str) -> list[dict]:
    return [{"role": "user", "content": f"Atomize: {text.strip()}"}]


def sampling_params():
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams
    return SamplingParams(temperature=0.0, max_tokens=1536,
                          structured_outputs=StructuredOutputsParams(json=SCHEMA))


def clean_props(sentence: str, props: list[str]) -> list[str]:
    max_len = 1.6 * len(sentence) + 20
    seen, out = set(), []
    for p in props:
        p = " ".join(p.split())
        key = p.lower().rstrip(".")
        if p and len(p) <= max_len and key not in seen:
            seen.add(key)
            out.append(p)
        if len(out) >= MAX_PROPS:
            break
    return out or [sentence.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description='sentences -> atomic propositions')
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="jsonl {premise, hypothesis, label}")
    ap.add_argument("--out", nargs="+", required=True, help="un --out par --pairs")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--chunk", type=int, default=2048)
    args = ap.parse_args()
    assert len(args.pairs) == len(args.out)


    print(f"[atomize] model {args.model}", flush=True)

    rows_per_file = []
    sentences, seen = [], set()
    for path in args.pairs:
        rows = [json.loads(l) for l in open(path)]
        rows_per_file.append(rows)
        for row in rows:
            for s in (row["premise"], row["hypothesis"]):
                s = s.strip()
                if s and s not in seen:
                    seen.add(s)
                    sentences.append(s)

    cache_path = Path(args.out[0]).parent / ".atomize_cache.jsonl"
    cache: dict[str, list[str]] = {}
    if cache_path.exists():
        with open(cache_path) as fh:
            cache = {r["text"]: r["props"] for r in map(json.loads, fh)}
    todo = [s for s in sentences if s not in cache]
    print(f"[atomize] {len(sentences)} unique sentences, {len(todo)} to decompose", flush=True)

    if todo:
        llm = load_llm(args.model, eager=True)
        params = sampling_params()
        for i in range(0, len(todo), args.chunk):
            chunk = todo[i:i + args.chunk]
            outs = llm.chat([messages_for(s) for s in chunk], params,
                            chat_template_kwargs={"enable_thinking": False})
            with open(cache_path, "a") as fh:
                for s, o in zip(chunk, outs):
                    try:
                        props = [p for p in json.loads(o.outputs[0].text)
                                 if isinstance(p, str)]
                    except json.JSONDecodeError:
                        props = []
                    props = clean_props(s, props)
                    cache[s] = props
                    fh.write(json.dumps({"text": s, "props": props}) + "\n")
            print(f"[atomize] {min(i + args.chunk, len(todo))}/{len(todo)}", flush=True)

    for rows, out in zip(rows_per_file, args.out):
        with open(out, "w") as fh:
            for row in rows:
                fh.write(json.dumps({
                    "premise": row["premise"], "hypothesis": row["hypothesis"],
                    "label": row["label"],
                    "premise_propositions": cache.get(row["premise"].strip(),
                                                      [row["premise"]]),
                    "hypothesis_propositions": cache.get(row["hypothesis"].strip(),
                                                         [row["hypothesis"]]),
                }) + "\n")
        print(f"[atomize] {len(rows)} paires -> {out}", flush=True)


if __name__ == "__main__":
    main()
