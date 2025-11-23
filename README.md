# PII NER Assignment Skeleton

PII token-level NER on noisy STT-style transcripts with a synthetic dataset, Hugging Face token classifier, and span decoding helpers (regex + filters).

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python src/train.py \
  --model_name distilbert-base-uncased \
  --train data/train.jsonl \
  --dev data/dev.jsonl \
  --out_dir out
```

## Predict

```bash
python src/predict.py \
  --model_dir out \
  --input data/dev.jsonl \
  --output out/dev_pred.json
```

## Evaluate

```bash
python src/eval_span_f1.py \
  --gold data/dev.jsonl \
  --pred out/dev_pred.json
```

## Measure latency

```bash
python src/measure_latency.py \
  --model_dir out \
  --input data/dev.jsonl \
  --runs 50
```

## Current baseline
- Model: distilbert-base-uncased, 3 epochs, batch_size 16, lr 5e-5, max_len 256.
- Data: generated 800 train / 160 dev / 200 test synthetic noisy STT (see `src/generate_data.py`), with numeric/locational decoys to stress precision.
- Dev metrics:
  - Macro-F1: 0.810
  - PII-only: P=0.689 R=0.627 F1=0.656
  - Per-entity: CITY 1.00/1.00/1.00, CREDIT_CARD 1.00/0.27/0.42, DATE 0.81/1.00/0.90, EMAIL 1.00/1.00/1.00, LOCATION 1.00/1.00/1.00, PERSON_NAME 1.00/1.00/1.00, PHONE 0.37/0.34/0.35
- Latency (CPU, batch=1, 50 runs): p50 8.78 ms, p95 10.29 ms.
- Outputs: `out/dev_pred.json`, `out/test_pred.json`.

## Decoding notes
- `predict.py` applies regex/length checks for numeric entities and a light confidence gate (16-digit CC, 10-digit phone, email must contain “at” and “dot” or “@”) to favor precision.

Your task in the assignment is to modify the model and training code to improve entity and PII detection quality while keeping **p95 latency below ~20 ms** per utterance (batch size 1, on a reasonably modern CPU).
