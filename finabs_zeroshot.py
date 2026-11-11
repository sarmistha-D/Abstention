#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import gc
import json
import random
import re
from collections import Counter
from pathlib import Path

MODEL_IDS = {
    "qwen25_vl_3b": "Qwen/Qwen2.5-VL-3B-Instruct",
    "qwen25_vl_7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "gemma3_4b": "google/gemma-3-4b-it",
    "gemma3_12b": "google/gemma-3-12b-it",
}


ZERO_SHOT_PROMPT = (
    "You are a financial multimodal assistant.\n"
    "Given the image and the user question, decide whether "
    "the available evidence is sufficient.\n"
    "If sufficient, answer the question. If not, abstain.\n"
    "Return a structured output with:\n"
    "ACTION: Answer or Abstain\n"
    "EVIDENCE_STATUS: present or absent\n"
    "EVIDENCE_MODALITY: visual, text, both, or none\n"
    "RESPONSE: concise financial answer or abstention response\n"
    "REASONING: short explanation\n"
    "JUSTIFICATION: evidence-grounded justification.\n"
)
FIELDS = ("ACTION", "EVIDENCE_STATUS", "EVIDENCE_MODALITY", "RESPONSE", "REASONING", "JUSTIFICATION")
FIELD_RE = re.compile(r"(?im)^\s*(ACTION|EVIDENCE_STATUS|EVIDENCE_MODALITY|RESPONSE|REASONING|JUSTIFICATION)\s*:\s*(.*)$")


def parse_fields(generation: str) -> dict:
    found = {}
    for match in FIELD_RE.finditer(generation):
        found.setdefault(match.group(1).upper(), match.group(2).strip())
    action = found.get("ACTION", "").strip().lower()
    if action not in ("answer", "abstain"):
        # Not scored as correct merely because the response contains 'abstain' somewhere.
        action = ""
    return {
        "pred_action": action,
        "evidence_status": found.get("EVIDENCE_STATUS", ""),
        "evidence_modality": found.get("EVIDENCE_MODALITY", ""),
        "response": found.get("RESPONSE", ""),
        "reasoning": found.get("REASONING", ""),
        "justification": found.get("JUSTIFICATION", ""),
        "format_complete": int(all(found.get(k) for k in FIELDS)),
    }


def read_rows(file: Path) -> list[dict]:
    if not file.is_file():
        raise FileNotFoundError(f"Data CSV not found: {file}")
    with file.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"Empty CSV: {file}")
    required = {"sample_id", "question", "expected_action"}
    if not required.issubset(rows[0]):
        raise ValueError(f"Missing CSV headers: {required - set(rows[0])}")
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate sample_id detected")
    return rows


def resolve_image(row: dict, root: Path) -> Path:
    val = row.get("v2_resolved_image") or row.get("resolved_image") or row.get("image")
    if not val:
        raise ValueError(f"Image field missing for {row['sample_id']}")
    candidate = Path(val)
    path = candidate if candidate.is_absolute() else root / candidate
    if not path.is_file():
        raise FileNotFoundError(f"No image for {row['sample_id']}: {path}")
    return path


def summarize(rows: list[dict]) -> dict:
    labeled = [r for r in rows if r["gold_action"] in ("answer", "abstain")]
    cm = Counter((r["gold_action"], r["pred_action"]) for r in labeled)
    n = len(labeled)
    f1s = {}
    for label in ("answer", "abstain"):
        tp = cm[label, label]
        fp = sum(cm[other, label] for other in ("answer", "abstain") if other != label)
        fn = sum(cm[label, other] for other in ("answer", "abstain", "") if other != label)
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s[label] = 2*p*r/(p+r) if p+r else 0.0
    abstain_n = sum(r["gold_action"] == "abstain" for r in labeled)
    answer_n = sum(r["gold_action"] == "answer" for r in labeled)
    return {
        "n": n,
        "correct_action_n": cm["answer", "answer"] + cm["abstain", "abstain"],
        "action_accuracy": (cm["answer", "answer"] + cm["abstain", "abstain"]) / n if n else None,
        "action_macro_f1": sum(f1s.values()) / 2,
        "answer_f1": f1s["answer"],
        "abstain_f1": f1s["abstain"],
        "unsafe_answer_rate": cm["abstain", "answer"] / abstain_n if abstain_n else None,
        "over_abstention_rate": cm["answer", "abstain"] / answer_n if answer_n else None,
        "unparsed_rate": sum(r["pred_action"] == "" for r in rows) / len(rows) if rows else None,
        "parse_failures_are_errors": True,
        "note": "Action agreement with supplied labels; not independently verified grounding or generation quality.",
    }


def write_csv(path: Path, rows: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_base_model(tag, checkpoint, quantize):
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    if not torch.cuda.is_available() and quantize:
        raise RuntimeError("--load-4bit needs CUDA; use --no-load-4bit for CPU (very high RAM use)")
    if tag.startswith("qwen"):
        processor = AutoProcessor.from_pretrained(checkpoint,
                                                   min_pixels=64*28*28, max_pixels=512*28*28)
    else:
        processor = AutoProcessor.from_pretrained(checkpoint)
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if torch.cuda.is_available() else torch.float32
    kwargs = {"dtype": dtype, "device_map": "auto" if torch.cuda.is_available() else "cpu"}
    if quantize:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True)
    model = AutoModelForImageTextToText.from_pretrained(checkpoint, **kwargs).eval()
    return model, processor


def predict(model, processor, row, root, max_image_edge, max_new_tokens):
    import torch
    from PIL import Image, ImageOps
    img_path = resolve_image(row, root)
    with Image.open(img_path) as source:
        img = ImageOps.exif_transpose(source).convert("RGB")
        if max(img.size) > max_image_edge:
            img.thumbnail((max_image_edge, max_image_edge), Image.Resampling.LANCZOS)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": img},
        {"type": "text", "text": ZERO_SHOT_PROMPT + "\nQUESTION: " + row["question"].strip()},
    ]}]
    enc = processor.apply_chat_template(messages, tokenize=True, return_dict=True,
                                        return_tensors="pt", add_generation_prompt=True)
    device = model.get_input_embeddings().weight.device
    enc = {k: v.to(device) if hasattr(v, "to") else v for k, v in enc.items()}
    with torch.inference_mode():
        out = model.generate(**enc, do_sample=False, max_new_tokens=max_new_tokens,
                             pad_token_id=processor.tokenizer.pad_token_id)
    generated = processor.batch_decode(out[:, enc["input_ids"].shape[1]:],
                                       skip_special_tokens=True,
                                       clean_up_tokenization_spaces=False)[0]
    return generated


def run_one(args, tag, file):
    import torch
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    checkpoint = args.checkpoint or MODEL_IDS[tag]
    model, processor = load_base_model(tag, checkpoint, args.load_4bit)
    try:
        rows = read_rows(file)
        rows = sorted(rows, key=lambda r: r["sample_id"])
        if args.max_samples > 0:
            rows = rows[:args.max_samples]
        path = args.out_dir / tag
        path.mkdir(parents=True, exist_ok=True)
        all_results = []
        for i, row in enumerate(rows, 1):
            try:
                gen = predict(model, processor, row, args.root, args.image_edge, args.max_new_tokens)
                parsed = parse_fields(gen)
                rec = {"sample_id": row["sample_id"], "question": row["question"],
                       "gold_action": row["expected_action"].lower().strip(),
                       "raw_generation": gen, **parsed, "status": "ok", "error": ""}
            except Exception as exc:
                rec = {"sample_id": row["sample_id"], "question": row["question"],
                       "gold_action": row["expected_action"].lower().strip(),
                       "pred_action": "", "status": "error", "error": str(exc)[:300]}
            all_results.append(rec)
            if i % 25 == 0 or i == len(rows):
                print(f"[ZERO-SHOT] {tag}: {i}/{len(rows)}", flush=True)
        write_csv(path / "predictions.csv", all_results)
        summary = {"model": tag, "checkpoint": checkpoint, "prompt_source": "paper Figure 6(a)",
                   "input_file_name": file.name, "n_success": sum(r["status"] == "ok" for r in all_results),
                   "metrics": summarize(all_results)}
        (path / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[DONE] {tag}: {summary['metrics']['action_accuracy']}; results in {path}")
    finally:
        del model, processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parent / "data")
    p.add_argument("--split", choices=["train", "validation", "test"], default="test")
    p.add_argument("--input-csv", type=Path, default=None, help="Override the split CSV")
    p.add_argument("--models", nargs="+", default=["qwen25_vl_7b"], help="Model keys or 'all'")
    p.add_argument("--checkpoint", default=None, help="Override checkpoint (single-model only)")
    p.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent / "runs" / "zero_shot")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--image-edge", type=int, default=896)
    p.add_argument("--max-new-tokens", type=int, default=384)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--load-4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--print-prompt", action="store_true")
    args = p.parse_args()
    if args.print_prompt:
        print(ZERO_SHOT_PROMPT)
        return
    chosen = list(MODEL_IDS) if args.models == ["all"] else args.models
    if any(m not in MODEL_IDS for m in chosen):
        p.error(f"Available models: {list(MODEL_IDS)}")
    if len(chosen) > 1 and args.checkpoint:
        p.error("--checkpoint override is available only for single-model runs")
    file = args.input_csv or (args.root / "finabs_step2_outputs" / f"{args.split}_v2_candidate.csv")
    for model_tag in chosen:
        run_one(args, model_tag, file)


if __name__ == "__main__":
    main()
