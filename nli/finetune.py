from __future__ import annotations

import argparse
import json
from pathlib import Path

MODEL = "Qwen/Qwen3.5-0.8B-Base"


def pick_logits(logits, labels):
    if isinstance(logits, (tuple, list)):
        for t in logits:
            if getattr(t, "ndim", 0) == 2 and t.shape[-1] == 3:
                return t
        return logits[0]
    return logits


def metrics_fn(eval_pred):
    import numpy as np
    logits, labels = eval_pred
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    acc = float((preds == labels).mean())
    f1s = []
    for c in range(3):
        tp = int(((preds == c) & (labels == c)).sum())
        fp = int(((preds == c) & (labels != c)).sum())
        fn = int(((preds != c) & (labels == c)).sum())
        f1s.append(2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0)
    return {"accuracy": acc, "macro_f1": float(sum(f1s) / 3)}


def main() -> None:
    ap = argparse.ArgumentParser(description='fine-tune the classifier on serialised graphs')
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--test", nargs="+", default=[])
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--bs", type=int, default=16, help="par device")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--eval-steps", type=int, default=1000)
    ap.add_argument("--grad-ckpt", action=argparse.BooleanOptionalAction, default=True,
                    help="--no-grad-ckpt : plus rapide, plus de VRAM")
    ap.add_argument("--eval-only", action="store_true",
                    help="no training: evaluate --model on val and tests")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              Trainer, TrainingArguments, set_seed)

    set_seed(args.seed)
    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.truncation_side = "left"

    ds = load_dataset("json", data_files={
        "train": args.train, "val": args.val,
        **({"test": args.test} if args.test else {})})

    def tok_fn(batch):
        return tok(batch["text"], truncation=True, max_length=args.max_len)

    keep = [c for c in ds["train"].column_names if c != "label"]
    ds = ds.map(tok_fn, batched=True, remove_columns=keep, num_proc=4)

    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model, num_labels=3)

    for c in (cfg, getattr(cfg, "text_config", None)):
        if c is not None:
            c.pad_token_id = tok.pad_token_id
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, config=cfg, dtype=torch.bfloat16)

    targs = TrainingArguments(
        output_dir=args.out, seed=args.seed,
        learning_rate=args.lr, num_train_epochs=args.epochs,
        per_device_train_batch_size=args.bs,
        per_device_eval_batch_size=args.bs * 2,
        gradient_accumulation_steps=args.accum,
        lr_scheduler_type="cosine", warmup_ratio=0.03, weight_decay=0.01,
        bf16=True, gradient_checkpointing=args.grad_ckpt,
        logging_steps=100, eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="steps", save_steps=args.eval_steps, save_total_limit=1,
        load_best_model_at_end=True, metric_for_best_model="accuracy",
        greater_is_better=True, report_to=[],
        label_names=["labels"],
    )
    trainer = Trainer(model=model, args=targs, processing_class=tok,
                      train_dataset=ds["train"], eval_dataset=ds["val"],
                      compute_metrics=metrics_fn,
                      preprocess_logits_for_metrics=pick_logits)
    if not args.eval_only:


        ckpts = sorted(Path(args.out).glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[1]))
        trainer.train(resume_from_checkpoint=str(ckpts[-1]) if ckpts else None)
        final = Path(args.out) / "final"
        trainer.save_model(str(final))
        tok.save_pretrained(str(final))

    results = {"seed": args.seed, "model": args.model,
               "train": args.train, "val_best": None}
    val = trainer.evaluate(ds["val"], metric_key_prefix="val")
    train_sub = ds["train"].shuffle(seed=0).select(
        range(min(len(ds["val"]), len(ds["train"]))))
    tr = trainer.evaluate(train_sub, metric_key_prefix="trainsub")
    results["val_best"] = val["val_accuracy"]
    results["gap_train_val"] = tr["trainsub_accuracy"] - val["val_accuracy"]
    if args.test:
        for i, path in enumerate(args.test):
            one = load_dataset("json", data_files=path)["train"]
            one = one.map(tok_fn, batched=True,
                          remove_columns=[c for c in one.column_names
                                          if c != "label"])
            m = trainer.evaluate(one, metric_key_prefix=f"test{i}")
            results[f"test:{Path(path).parent.name}/{Path(path).stem}"] = {
                "accuracy": m[f"test{i}_accuracy"],
                "macro_f1": m[f"test{i}_macro_f1"]}
    print(json.dumps(results, indent=2))
    Path(args.out, "results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
