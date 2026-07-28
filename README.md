# Interpretable NLI with Graphs Based on Atomic Propositions

Code for the paper. The classifier reads serialised graphs only; no raw text
reaches the classification stage.

## Install

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

ConceptNet 5.7 English assertions are expected in `data/conceptnet/`.

## Pipeline

```bash
# 1. sentences -> atomic propositions
python -m nli.atomize --pairs data/pairs/snli/train.jsonl \
    --out data/atomic/snli/train.jsonl

# 2. atomic propositions -> ConceptNet triples (GPU, vLLM)
python -m nli.extract --inputs data/atomic/snli/train.jsonl \
    --out out/snli/triples.jsonl

# 3. ConceptNet base (once), then the three-graph datasets
python -m nli.conceptnet build
python -m nli.dataset --atomic data/atomic/snli/train.jsonl \
    --triples out/snli/triples.jsonl --out out/snli/train.dataset.jsonl

# 4. fine-tuning on SNLI, then on ANLI from the SNLI checkpoint
python -m nli.finetune --train out/snli/train.dataset.jsonl \
    --val out/snli/validation.dataset.jsonl --test out/snli/test.dataset.jsonl \
    --out out/models/snli-s0 --seed 0 --no-grad-ckpt --epochs 2 --max-len 1024

python -m nli.finetune --model out/models/snli-s0/checkpoint-8000 \
    --train out/anli/train_r{1,2,3}.dataset.jsonl \
    --val out/anli/dev_r{1,2,3}.dataset.jsonl \
    --test out/anli/test_r{1,2,3}.dataset.jsonl out/snli/test.dataset.jsonl \
    --out out/models/anli-phase2 --max-len 1536 --epochs 3

# zero-shot control (no fine-tuning)
python -m nli.zeroshot --inputs out/snli/test.dataset.jsonl \
    --out out/models/zeroshot/snli.json
```

Checkpoints are selected on the validation split, never on the test split.

## Modules

| | |
|---|---|
| `nli/atomize.py` | sentences to atomic propositions |
| `nli/extract.py` | propositions to ConceptNet triples, constrained JSON decoding |
| `nli/conceptnet.py` | ConceptNet base and retrieval of the subgraph `K` |
| `nli/dataset.py` | assembly of `P`, `H`, `K` and serialisation |
| `nli/finetune.py` | classifier fine-tuning |
| `nli/zeroshot.py` | zero-shot label-likelihood baseline |
