import json
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from labels import ID2LABEL, label_is_pii
import os
import re


def bio_to_spans(text, offsets, label_ids):
    spans = []
    current_label = None
    current_start = None
    current_end = None

    for (start, end), lid in zip(offsets, label_ids):
        if start == 0 and end == 0:
            continue
        label = ID2LABEL.get(int(lid), "O")
        if label == "O":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
            continue

        prefix, ent_type = label.split("-", 1)
        if prefix == "B":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
            current_label = ent_type
            current_start = start
            current_end = end
        elif prefix == "I":
            if current_label == ent_type:
                current_end = end
            else:
                if current_label is not None:
                    spans.append((current_start, current_end, current_label))
                current_label = ent_type
                current_start = start
                current_end = end

    if current_label is not None:
        spans.append((current_start, current_end, current_label))

    return spans


def apply_regex_overrides(text, spans):
    """Use lightweight regex helpers to fix obvious numeric entity types."""
    def upsert(target_label, start, end):
        # If a span with identical offsets exists, relabel it; else append.
        for i, (s, e, lab) in enumerate(spans):
            if s == start and e == end:
                spans[i] = (s, e, target_label)
                return
        spans.append((start, end, target_label))

    phone_re = re.compile(r"\b\d{10}\b")
    cc_re = re.compile(r"\b(?:\d{4}\s?){4}\b")
    date_re = re.compile(r"\b\d{2}\s\d{2}\s\d{4}\b")

    for m in phone_re.finditer(text):
        upsert("PHONE", m.start(), m.end())
    for m in cc_re.finditer(text):
        upsert("CREDIT_CARD", m.start(), m.end())
    for m in date_re.finditer(text):
        upsert("DATE", m.start(), m.end())

    # If something was tagged as CREDIT_CARD but is only 10 digits, flip to PHONE.
    for i, (s, e, lab) in enumerate(list(spans)):
        if lab == "CREDIT_CARD":
            span_txt = text[s:e]
            digits_only = re.sub(r"\D", "", span_txt)
            if len(digits_only) == 10:
                spans[i] = (s, e, "PHONE")
            elif re.fullmatch(r"\d{2}\s\d{2}\s\d{4}", span_txt):
                spans[i] = (s, e, "DATE")

    return spans


def filter_spans(text, spans, logit_scores=None, ids=None, min_conf: float = 0.0):
    """
    Drop low-confidence or implausible spans to bump precision.
    - CREDIT_CARD: require 16 digits (allow spaces/hyphens)
    - PHONE: require 10 digits (allow spaces)
    - EMAIL: must contain 'at' and 'dot' somewhere
    - DATE: keep as-is
    """
    filtered = []
    for span in spans:
        s, e, lab = span
        chunk = text[s:e]
        digits = re.sub(r"\\D", "", chunk)
        # label-specific confidence targets
        cc_conf = 0.6
        phone_conf = 0.4
        email_conf = 0.5
        label_min_conf = min_conf
        if lab == "CREDIT_CARD":
            if len(digits) != 16:
                continue
            label_min_conf = cc_conf
        elif lab == "PHONE":
            if len(digits) != 10:
                continue
            label_min_conf = phone_conf
        elif lab == "EMAIL":
            lowered = chunk.lower()
            if not (("at" in lowered and "dot" in lowered) or "@" in lowered):
                continue
            label_min_conf = email_conf
        # optional confidence gate (token-level max over span)
        if logit_scores and ids:
            span_scores = []
            for idx, (start, end) in enumerate(ids):
                if start >= s and end <= e and (end - start) > 0:
                    span_scores.append(max(logit_scores[idx]))
            if span_scores and max(span_scores) < label_min_conf:
                continue
        filtered.append(span)
    return filtered


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir if args.model_name is None else args.model_name)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()

    results = {}

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]

            enc = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            )
            offsets = enc["offset_mapping"][0].tolist()
            input_ids = enc["input_ids"].to(args.device)
            attention_mask = enc["attention_mask"].to(args.device)

            with torch.no_grad():
                out = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = out.logits[0]
                pred_ids = logits.argmax(dim=-1).cpu().tolist()
                scores = logits.softmax(dim=-1).cpu().tolist()

            spans = bio_to_spans(text, offsets, pred_ids)
            spans = apply_regex_overrides(text, spans)
            spans = filter_spans(text, spans, logit_scores=scores, ids=offsets, min_conf=0.6)
            ents = []
            for s, e, lab in spans:
                ents.append(
                    {
                        "start": int(s),
                        "end": int(e),
                        "label": lab,
                        "pii": bool(label_is_pii(lab)),
                    }
                )
            results[uid] = ents

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions for {len(results)} utterances to {args.output}")


if __name__ == "__main__":
    main()
