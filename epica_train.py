#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

RUN
  python epica_train.py --model qwen25_vl_7b                       # train
  python epica_train.py --model qwen25_vl_7b --pilot               # quick check
  python epica_train.py --model qwen25_vl_7b --mode evaluate --evaluate-test
  python epica_train.py --self-test                                # CPU, no model
"""

import argparse
import copy
import csv
import gc
import json
import math
import random
import re
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import nn
from torch.nn import functional as F

MODELS = {   # key -> Hugging Face checkpoint, family, text-decoder depth
    "qwen25_vl_3b": {"hf_id": "Qwen/Qwen2.5-VL-3B-Instruct", "family": "qwen", "n_layers": 36},
    "qwen25_vl_7b": {"hf_id": "Qwen/Qwen2.5-VL-7B-Instruct", "family": "qwen", "n_layers": 28},
    "gemma3_4b": {"hf_id": "google/gemma-3-4b-it", "family": "gemma", "n_layers": 34},
    "gemma3_12b": {"hf_id": "google/gemma-3-12b-it", "family": "gemma", "n_layers": 48},
}
MODEL_KEY = "qwen25_vl_7b"          # set by --model
MODEL_ID = MODELS[MODEL_KEY]["hf_id"]
REPO_DIR = Path(__file__).resolve().parent


def set_model(key):
    global MODEL_KEY, MODEL_ID
    if key not in MODELS:
        raise ValueError(f"--model must be one of {sorted(MODELS)}")
    MODEL_KEY, MODEL_ID = key, MODELS[key]["hf_id"]
TRAINING_EPOCHS = 3
ACTIONS = ("ANSWER", "ABSTAIN")

# ---------------------------------------------------------------------------
# Built-in layer knowledge for Qwen2.5-VL-7B (1-based decoder layers, 28 total),
# from the layer diagnostics of its structured-SFT model.
# ---------------------------------------------------------------------------
DEFAULT_KNOWLEDGE = {
    "model": "qwen25_vl_7b",
    "index_base": 1,
    "n_layers": 28,
    "probe_auc": {},
    "probe_acc": {},
    "patch_flips_validation": {},
    "test_auc_original": {},
    "test_auc_blank": {},
    "roles": {
        "onset": [12],
        "evidence_memory": [19, 20],
        "decision_gate": [27],
        "terminal": [28],
    },
    "source": "built-in defaults (Qwen2.5-VL-7B)",
}

# Structured-SFT / EPICA instruction (the adapters were trained on this exact text).
STAGE2_INSTRUCTION = (
    'Assess the image and financial question using ONLY supplied evidence. '
    'Respond first with exactly one decision code: A for ANSWER (supported) '
    'or B for ABSTAIN (insufficient evidence). Then write the requested fields. '
    'Never assume a question is unanswerable based only on phrasing. '
    'Do not invent visual facts or reporting periods.\n'
    'Use exactly this output layout:\n'
    'A\nEVIDENCE_STATUS: present\nEVIDENCE_MODALITY: chart_graph\n'
    'RESPONSE: supported answer\nDECISION_REASONING: cited values/calculation\n'
    'JUSTIFICATION: why answering or abstaining is warranted\n'
    'The first line is the decision code; use B when evidence is insufficient.\n'
    'QUESTION: '
)

FIELD_RE = re.compile(
    r"(?im)^\s*(EVIDENCE_STATUS|EVIDENCE_MODALITY|RESPONSE|"
    r"DECISION_REASONING|JUSTIFICATION|ACTION|REQUIRED_EVIDENCE_STATUS|"
    r"STATUS|REASONING)\s*:"
)
FIELD_WEIGHTS = {
    "ACTION": 3.0, "EVIDENCE_STATUS": 1.5, "EVIDENCE_MODALITY": 0.5,
    "DECISION_REASONING": 0.5, "REQUIRED_EVIDENCE_STATUS": 1.5, "STATUS": 1.5,
    "RESPONSE": 4.0, "REASONING": 0.5, "JUSTIFICATION": 0.5,
}
NUMERIC_TOKEN_WEIGHT = 2.0
NUMERIC_SPAN_RE = re.compile(r"[+-]?\$?\d[\d,]*(?:\.\d+)?\s?%?")


# ===========================================================================
# utilities
# ===========================================================================
def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)


def parse_action(text):
    text = str(text or "").strip()
    if not text:
        return None
    first = text.splitlines()[0].strip().strip("\"' ").upper()
    if first == "A":
        return "ANSWER"
    if first == "B":
        return "ABSTAIN"
    m = re.fullmatch(r"ACTION\s*:\s*(ANSWER|ABSTAIN)\s*", first, re.I)
    return m.group(1).upper() if m else None


def parse_response(text):
    m = re.search(r"(?im)^\s*RESPONSE\s*:\s*(.*)$", str(text or ""))
    return m.group(1).strip() if m else ""


def parse_complete(text):
    found = {m.group(1).upper() for m in FIELD_RE.finditer(str(text))}
    need = {"EVIDENCE_STATUS", "EVIDENCE_MODALITY", "RESPONSE",
            "DECISION_REASONING", "JUSTIFICATION"}
    return parse_action(text) in ACTIONS and bool(parse_response(text)) and need <= found


def normalized_text(text):
    return re.sub(r"\s+", " ", re.sub(r"[^\w.%+-]", " ", str(text).lower())).strip()


def response_correct(pred, gold, tol=Decimal("0.000001")):
    if not str(gold).strip() or not str(pred).strip():
        return False
    if normalized_text(pred) == normalized_text(gold):
        return True
    pattern = r"^[+-]?\d[\d,]*(?:\.\d+)?%?$"
    pp, gg = str(pred).strip(), str(gold).strip()
    if re.fullmatch(pattern, pp) and re.fullmatch(pattern, gg):
        if pp.endswith("%") != gg.endswith("%"):
            return False
        try:
            return abs(Decimal(pp.rstrip("%").replace(",", ""))
                       - Decimal(gg.rstrip("%").replace(",", ""))) <= tol
        except InvalidOperation:
            return False
    return False


def rouge_l(pred, gold):
    a, b = normalized_text(pred).split(), normalized_text(gold).split()
    if not a or not b:
        return 0.0
    dp = [0] * (len(b) + 1)
    for x in a:
        prev = 0
        for j, y in enumerate(b, 1):
            old = dp[j]
            dp[j] = prev + 1 if x == y else max(dp[j], dp[j - 1])
            prev = old
    p, r = dp[-1] / len(a), dp[-1] / len(b)
    return 2 * p * r / (p + r) if p + r else 0.0


def safe_auc(y, score):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, score))


# ===========================================================================
# Layer knowledge: parse a layer report (.md/.txt) or a .json
# ===========================================================================
def _md_rows(text):
    """Yield (cells, is_header); a header is the row right above a |---| line."""
    lines = [ln.strip() for ln in str(text).splitlines()]
    for k, line in enumerate(lines):
        if not (line.startswith("|") and line.endswith("|")):
            continue
        if re.fullmatch(r"[|\s:-]+", line):
            continue
        nxt = lines[k + 1] if k + 1 < len(lines) else ""
        is_header = bool(nxt.startswith("|") and re.fullmatch(r"[|\s:-]+", nxt))
        yield [c.strip() for c in line.strip("|").split("|")], is_header


def _num(cell):
    cell = str(cell).strip()
    m = re.fullmatch(r"(\d+)\s*/\s*(\d+)", cell)
    if m:
        return int(m.group(1)) / max(int(m.group(2)), 1)
    m = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\s*%", cell)
    if m:
        return float(m.group(1)) / 100.0
    try:
        return float(cell)
    except ValueError:
        return None


def parse_layer_knowledge(path):
    """Read a layer report (.md/.txt tables) or a .json into a knowledge dict.

    Recognised markdown tables:
      | Layer (1-based) | Validation AUC | Validation accuracy | ... |
      | Layer | Validation flips | Test flips |
      | Layer | Original test AUC | Blank-image AUC | Question-only AUC |
      | Component | Decoder layer (1-based) | Reason |
    Anything not found keeps the DEFAULT_KNOWLEDGE value.
    """
    know = copy.deepcopy(DEFAULT_KNOWLEDGE)
    if path is None:
        return know
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"--layer-knowledge file not found: {path}")
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".json":
        loaded = json.loads(text)
        for key, value in loaded.items():
            if key == "plan_weights":
                value = {name: {int(k): float(v) for k, v in w.items()} for name, w in value.items()}
            elif key == "roles":
                value = {role: [int(L) for L in layers] for role, layers in value.items()}
            elif isinstance(value, dict):
                value = {int(k): float(v) for k, v in value.items()}
            know[key] = value
        if "plan_weights" not in loaded:      # the built-in plan weights belong to Qwen2.5-VL-7B only
            know.pop("plan_weights", None)
        know["source"] = str(path)
        return know

    header = None
    found = {"probe_auc": {}, "probe_acc": {}, "patch_flips_validation": {},
             "test_auc_original": {}, "test_auc_blank": {}, "roles": {}}
    for cells, is_header in _md_rows(text):
        low = [c.lower() for c in cells]
        if is_header:
            header = low if ("layer" in low[0] or "component" in low[0]) else None
            continue
        if header is None:
            continue
        if "component" in header[0]:
            layers = [int(v) for v in re.findall(r"\d+", cells[1])] if len(cells) > 1 else []
            name = low[0]
            role = ("terminal" if "terminal" in name else
                    "decision_gate" if "gate" in name or "decision" in name else
                    "evidence_memory" if "evidence" in name or "memory" in name else
                    "onset" if "early" in name or "onset" in name else None)
            if role and layers:
                found["roles"].setdefault(role, [])
                found["roles"][role].extend(layers)
            continue
        if not re.fullmatch(r"\d+", cells[0]):
            continue
        layer = int(cells[0])
        values = [_num(c) for c in cells[1:]]
        joined = " ".join(header)
        if "validation auc" in joined:
            if values and values[0] is not None:
                found["probe_auc"][layer] = values[0]
            if len(values) > 1 and values[1] is not None:
                found["probe_acc"][layer] = values[1]
        elif "flips" in joined:
            if values and values[0] is not None:
                found["patch_flips_validation"][layer] = values[0]
        elif "blank" in joined:
            if values and values[0] is not None:
                found["test_auc_original"][layer] = values[0]
            if len(values) > 1 and values[1] is not None:
                found["test_auc_blank"][layer] = values[1]
    for key, value in found.items():
        if value:
            know[key] = ({k: sorted(set(v)) for k, v in value.items()}
                         if key == "roles" else value)
    know["source"] = str(path)
    return know


def knowledge_plan(know, n_decoder_layers):
    """Turn the knowledge into zero-based decoder indices and loss weights."""
    base = int(know.get("index_base", 1))
    roles = know["roles"]
    terminal = set(roles.get("terminal", [n_decoder_layers if base == 1 else n_decoder_layers - 1]))
    classify = sorted(set(roles.get("onset", [])) | set(roles.get("evidence_memory", [])))
    gate = sorted(set(roles.get("decision_gate", [])) - terminal)
    if not classify:
        raise ValueError("Layer knowledge has no onset/evidence_memory layers to classify at")
    if not gate:
        raise ValueError("Layer knowledge has no non-terminal decision_gate layer")
    gate_layer = gate[0]
    classify = [L for L in classify if L < gate_layer]
    if not classify:
        raise ValueError("Classification layers must come before the decision gate")
    auc = know.get("probe_auc", {})
    orig, blank = know.get("test_auc_original", {}), know.get("test_auc_blank", {})
    w_auc = {L: max(float(auc.get(L, 0.9)) - 0.5, 0.05) for L in classify}
    total = sum(w_auc.values())
    w_auc = {L: v / total for L, v in w_auc.items()}
    evidence = set(roles.get("evidence_memory", []))
    gap = {L: max(float(orig.get(L, 0.9)) - float(blank.get(L, 0.9)), 0.05) for L in classify}
    w_vis = {L: gap[L] * (2.0 if L in evidence else 1.0) for L in classify}
    total = sum(w_vis.values())
    w_vis = {L: v / total for L, v in w_vis.items()}
    fixed = know.get("plan_weights") or {}
    if fixed:   # exported from a trained controller.pt: reproduce its plan exactly
        if set(fixed.get("auc_weight", {})) != set(classify) or set(fixed.get("visual_weight", {})) != set(classify):
            raise ValueError(f"plan_weights must cover exactly the classification layers {classify}")
        w_auc, w_vis = dict(fixed["auc_weight"]), dict(fixed["visual_weight"])
    to0 = (lambda L: L - 1) if base == 1 else (lambda L: L)
    plan = {
        "classify_layers_report": classify,
        "gate_layer_report": gate_layer,
        "terminal_layers_report": sorted(terminal),
        "classify_layers": [to0(L) for L in classify],
        "gate_layer": to0(gate_layer),
        "auc_weight": [w_auc[L] for L in classify],
        "visual_weight": [w_vis[L] for L in classify],
        "roles_report": roles,
    }
    for L in plan["classify_layers"] + [plan["gate_layer"]]:
        if not 0 <= L < n_decoder_layers:
            raise ValueError(f"Knowledge layer {L} (0-based) outside decoder 0..{n_decoder_layers-1}")
    return plan


# ===========================================================================
# Layer-knowledge controller (action heads + fusion + gate-layer decision injection)
# ===========================================================================
def _hidden(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def _replace(out, h):
    if isinstance(out, tuple):
        return (h,) + tuple(out[1:])
    if isinstance(out, list):
        return [h] + list(out[1:])
    return h


class Runtime:
    def __init__(self):
        self.reset()

    def reset(self, read_pos=None, capture=False, inject=False, fixed_p=None):
        self.read_pos = read_pos
        self.capture = capture
        self.inject = inject
        self.fixed_p = fixed_p
        self.logits = {}
        self.fused = None


class LayerKnowledgeController(nn.Module):
    def __init__(self, hidden, plan, decision_direction):
        super().__init__()
        self.classify = list(plan["classify_layers"])
        self.gate_layer = int(plan["gate_layer"])
        self.heads = nn.ModuleDict({
            str(L): nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 1))
            for L in self.classify})
        for head in self.heads.values():
            nn.init.normal_(head[1].weight, std=1e-3)
            nn.init.zeros_(head[1].bias)
        self.fusion = nn.Parameter(torch.log(torch.tensor(plan["auc_weight"], dtype=torch.float32)))
        self.register_buffer("auc_weight", torch.tensor(plan["auc_weight"], dtype=torch.float32))
        self.register_buffer("visual_weight", torch.tensor(plan["visual_weight"], dtype=torch.float32))
        u = torch.as_tensor(decision_direction, dtype=torch.float32)
        self.register_buffer("u_dec", u / u.norm().clamp_min(1e-6))
        # Zero init: training starts exactly at the structured-SFT model.
        self.g_dec = nn.Parameter(torch.zeros(()))
        self.v_answer = nn.Parameter(torch.zeros(hidden))
        self.v_abstain = nn.Parameter(torch.zeros(hidden))
        self.rt = Runtime()
        self.handles = []
        self.enabled_injection = True

    @property
    def device(self):
        return self.fusion.device

    def head_logit(self, layer, vec):
        return self.heads[str(layer)](vec.to(self.device).float()).squeeze(-1)

    def fused_logit(self, logits):
        w = torch.softmax(self.fusion, dim=0)
        stack = torch.stack([logits[L] for L in self.classify], dim=-1)
        return (stack * w).sum(-1)

    # ---- hooks -----------------------------------------------------------
    def install(self, decoder_layers):
        self.remove()
        for L in self.classify:
            self.handles.append(decoder_layers[L].register_forward_hook(self._capture(L)))
        self.handles.append(decoder_layers[self.gate_layer].register_forward_hook(self._inject))

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

    def _capture(self, layer):
        def hook(_module, _inputs, out):
            rt = self.rt
            if not rt.capture or rt.read_pos is None:
                return None
            h = _hidden(out)
            if h.shape[1] <= rt.read_pos:
                return None          # cached decoding step
            rt.logits[layer] = self.head_logit(layer, h[:, rt.read_pos, :])
            return None
        return hook

    def _inject(self, _module, _inputs, out):
        rt = self.rt
        if not (rt.inject and self.enabled_injection) or rt.read_pos is None:
            return None
        h = _hidden(out)
        if rt.fixed_p is not None:
            p = rt.fixed_p
        else:
            if any(L not in rt.logits for L in self.classify):
                return None
            rt.fused = self.fused_logit(rt.logits)
            p = torch.sigmoid(rt.fused)
        p = p.to(self.device).float().reshape(-1, 1)
        ctx = p * self.v_abstain + (1.0 - p) * self.v_answer          # [B,D]
        dec = self.g_dec * (1.0 - 2.0 * p) * self.u_dec              # [B,D] (+A / -B)
        T = h.shape[1]
        rms = h.detach().float().pow(2).mean(-1, keepdim=True).sqrt()  # [B,T,1]
        if T == 1:
            delta = ctx[:, None, :].to(h.device) * rms
        else:
            if rt.read_pos >= T:
                return None
            pos = torch.arange(T, device=h.device)
            after = (pos >= rt.read_pos).float()[None, :, None]
            at = (pos == rt.read_pos).float()[None, :, None]
            delta = (after * ctx[:, None, :].to(h.device) + at * dec[:, None, :].to(h.device)) * rms
        return _replace(out, (h.float() + delta).to(h.dtype))


# ===========================================================================
# Data  (<data-dir>/<split>.csv, see data/README.md)
# ===========================================================================
def _read_csv_rows(path):
    if not Path(path).is_file():
        raise FileNotFoundError(f"Missing split CSV: {path}")
    with open(path, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"CSV has no rows: {path}")
    return rows


def _one_line(s):
    return re.sub(r"\s+", " ", str(s or "")).strip()


def _reference_files(root):
    refs = {}
    if root is None:
        return refs
    root = Path(root)
    for p in [root / "all_with_split.csv", root / "train.csv",
              root / "validation.csv", root / "test.csv"]:
        if not p.is_file():
            continue
        for r in _read_csv_rows(p):
            sid = str(r.get("sample_id", "")).strip()
            if not sid:
                continue
            ref = refs.setdefault(sid, {})
            for k in ("response", "reasoning", "required_evidence_status",
                      "evidence_modality", "reasoning_label_explanation"):
                value = str(r.get(k, "") or "").strip()
                if value and not ref.get(k):
                    ref[k] = value
    return refs


def _resolve_image(raw, *roots):
    candidate = Path(str(raw or "").strip())
    for p in (candidate, *[Path(r) / candidate for r in roots if r is not None]):
        if p.is_file():
            return str(p.resolve())
    return str(candidate)


def load_split(data_dir, split, refs, knowledge_prompt="", split_csv="{split}.csv", reference_dir=None):
    path = Path(data_dir) / split_csv.format(split=split)
    records = []
    for index, r in enumerate(_read_csv_rows(path)):
        sid = str(r.get("sample_id", "")).strip()
        act = str(r.get("expected_action", "")).strip().lower()
        if act not in ("answer", "abstain") or not sid:
            raise ValueError(f"{path}, row {index}: invalid sample_id or expected_action")
        image = _resolve_image(r.get("v2_resolved_image") or r.get("resolved_image")
                               or r.get("image") or "", data_dir, reference_dir)
        question = str(r.get("question", "")).strip()
        if not question:
            raise ValueError(f"{path}, row {index}: question missing")
        ref = refs.get(sid, {})
        response = _one_line(r.get("human_final_answer") or r.get("response")
                             or ref.get("response") or "") or "N/A"
        reasoning = _one_line(r.get("reasoning") or ref.get("reasoning") or "") or "N/A"
        human_just = (r.get("human_final_justification")
                      if r.get("adjudication_status") == "FINALIZED" else "")
        justification = _one_line(human_just or r.get("reasoning_label_explanation")
                                  or ref.get("reasoning_label_explanation") or r.get("justification")
                                  or reasoning) or "N/A"
        status = str(r.get("required_evidence_status") or ref.get("required_evidence_status")
                     or "unknown").strip().lower()
        status = {"sufficient": "present", "insufficient": "absent", "missing": "absent"}.get(status, status)
        modality = _one_line(r.get("evidence_modality") or ref.get("evidence_modality") or "") or "unspecified"
        code = "A" if act == "answer" else "B"
        target = (f"{code}\nEVIDENCE_STATUS: {status}\nEVIDENCE_MODALITY: {modality}\n"
                  f"RESPONSE: {response}\nDECISION_REASONING: {reasoning}\n"
                  f"JUSTIFICATION: {justification}")
        records.append({"sample_id": sid, "image": image,
                        "prompt": knowledge_prompt + STAGE2_INSTRUCTION + question,
                        "target": target, "gold_action": act.upper(),
                        "reference_response": response, "modality": modality})
    if len({x["sample_id"] for x in records}) != len(records):
        raise ValueError(f"Duplicate IDs in {path}")
    print(f"[DATA] {split}: {len(records)} from {path}")
    return sorted(records, key=lambda x: x["sample_id"])


def abstention_target(modality="unspecified"):
    return ("B\nEVIDENCE_STATUS: absent\nEVIDENCE_MODALITY: " + str(modality or "unspecified") +
            "\nRESPONSE: Insufficient reliable visual evidence to answer.\n"
            "DECISION_REASONING: The required visual evidence is not present in the image.\n"
            "JUSTIFICATION: Abstaining avoids an unsupported financial assertion.")


def knowledge_prompt_text(plan):
    return ("[Layer knowledge] Abstention evidence becomes decodable at decoder layer "
            f"{plan['classify_layers_report'][0]}, is visually grounded at layers "
            f"{', '.join(map(str, plan['classify_layers_report'][1:]))}, and the decision "
            f"is formed at layer {plan['gate_layer_report']}. Decide A only when the "
            "required visual evidence is present.\n")


# ===========================================================================
# Model / processor
# ===========================================================================
def hub_load(cls, name, **kwargs):
    """Local cache first; the Hub only if that fails. On an offline machine set
    HF_HUB_OFFLINE=1 (or pass --checkpoint /path/to/local/model)."""
    try:
        return cls.from_pretrained(name, local_files_only=True, **kwargs)
    except Exception as local_error:
        try:
            return cls.from_pretrained(name, **kwargs)
        except Exception as online_error:
            raise RuntimeError(
                f"Could not load {name!r} locally ({local_error}) or from the Hub "
                f"({online_error}). Pass --checkpoint /path/to/local/model.") from online_error


def find_decoder_layers(model, expected=None):
    found = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.ModuleList) or not len(module):
            continue
        if any(t in name.lower() for t in ("visual", "vision", "vit", "siglip")):
            continue
        if expected is not None and len(module) != expected:
            continue
        first = module[0]
        if hasattr(first, "mlp") and hasattr(first, "self_attn"):
            found.append((name, module))
    if not found:
        raise RuntimeError("Could not locate the language decoder layers")
    found.sort(key=lambda x: ("language_model" in x[0], len(x[1])), reverse=True)
    return found[0]


def decision_token_ids(tokenizer):
    a = tokenizer.encode("A", add_special_tokens=False)
    b = tokenizer.encode("B", add_special_tokens=False)
    if len(a) != 1 or len(b) != 1:
        raise RuntimeError(f"Decision codes must be single tokens: A={a} B={b}")
    return a[0], b[0]


def load_model(args, trainable):
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
    from peft import LoraConfig, PeftModel, get_peft_model

    family = MODELS[MODEL_KEY]["family"]
    processor = hub_load(AutoProcessor, args.checkpoint)
    ip = getattr(processor, "image_processor", None)
    if family == "qwen" and ip is not None and hasattr(ip, "max_pixels"):
        ip.max_pixels = 512 * 28 * 28            # same setting as the structured-SFT stage
    kwargs = {}
    if getattr(args, "attn_implementation", None):
        kwargs["attn_implementation"] = args.attn_implementation
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
        if family == "gemma" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("Gemma-3 needs a bf16-capable GPU (float16 overflows)")
        kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if args.load_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=kwargs["torch_dtype"], bnb_4bit_use_double_quant=True)
    else:
        if args.load_4bit:
            raise RuntimeError("4-bit loading needs CUDA; pass --no-load-4bit for CPU")
        kwargs["torch_dtype"] = torch.float32
    model = hub_load(AutoModelForImageTextToText, args.checkpoint, **kwargs)
    model.config.use_cache = False

    adapter = args.adapter_to_load
    if adapter is not None:
        print(f"[MODEL] loading adapter {adapter} (trainable={trainable})")
        model = PeftModel.from_pretrained(model, str(adapter), is_trainable=trainable)
    elif trainable:
        print(f"[MODEL] fresh LoRA r={args.lora_rank} on the LANGUAGE model only")
        config = LoraConfig(
            r=args.lora_rank, lora_alpha=2 * args.lora_rank, lora_dropout=0.05, bias="none",
            target_modules=r"^(?!.*(visual|vision)).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$",
            task_type="CAUSAL_LM")
        model = get_peft_model(model, config)
    if trainable:
        if args.grad_ckpt:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[MODEL] trainable LoRA parameters: {n_train/1e6:.2f}M")
    else:
        model.requires_grad_(False)
    model.eval()
    return model, processor


def model_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def unembedding_direction(model, a_id, b_id):
    weight = model.get_output_embeddings().weight
    return (weight[a_id].float() - weight[b_id].float()).detach().cpu()


# ===========================================================================
# Tokenisation, field weights
# ===========================================================================
def prompt_text(processor, row):
    messages = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": row["prompt"]}]}]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_image(row, blank=False):
    with Image.open(row["image"]) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    if max(image.size) > 896:
        image.thumbnail((896, 896), Image.Resampling.LANCZOS)
    if blank:
        image = Image.new("RGB", image.size, (255, 255, 255))
    return image


def move_inputs(inputs, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}


def token_field_weights(tokenizer, ids, prefix_length):
    ids = [int(v) for v in ids]
    completion = ids[prefix_length:]
    generated = tokenizer.decode(completion, skip_special_tokens=False)
    offsets = [0]
    for j in range(1, len(completion) + 1):
        offsets.append(len(tokenizer.decode(completion[:j], skip_special_tokens=False)))
    offsets[-1] = len(generated)
    if not re.match(r"^\s*[AB]\s*(?:\n|$)", generated):
        raise ValueError("Target must start with decision code A or B")
    markers = list(FIELD_RE.finditer(generated))
    resp = [m for m in markers if m.group(1).upper() == "RESPONSE"]
    if not resp:
        raise ValueError("Target requires a RESPONSE: field")
    resp = resp[0]
    resp_end = generated.find("\n", resp.end())
    resp_end = len(generated) if resp_end < 0 else resp_end
    numeric = [(m.start() + resp.end(), m.end() + resp.end())
               for m in NUMERIC_SPAN_RE.finditer(generated[resp.end():resp_end])]
    weights = np.zeros(len(ids), dtype=np.float32)
    for j in range(len(completion)):
        start, end = int(offsets[j]), int(offsets[j + 1])
        if not generated[:start].strip() or not markers or start < markers[0].start():
            field = "ACTION"
        else:
            prior = [m for m in markers if m.start() <= start]
            field = prior[-1].group(1).upper() if prior else "ACTION"
        w = FIELD_WEIGHTS.get(field, 0.5)
        if resp.end() <= start < resp_end and any(start < b and end > a for a, b in numeric):
            w *= NUMERIC_TOKEN_WEIGHT
        weights[prefix_length + j] = w
    return weights


def build_inputs(processor, row, target, device, max_seq_len, blank=False):
    image = load_image(row, blank=blank)
    prefix = prompt_text(processor, row)
    prompt = processor(text=[prefix], images=[image], return_tensors="pt")
    if target is None:
        return move_inputs(prompt, device), int(prompt["input_ids"].shape[1]), None
    eos = processor.tokenizer.eos_token or ""
    full = processor(text=[prefix + target + eos], images=[image], return_tensors="pt")
    p_ids, f_ids = prompt["input_ids"][0].tolist(), full["input_ids"][0].tolist()
    if f_ids[:len(p_ids)] != p_ids:
        raise RuntimeError("Prompt/full tokenisation boundary mismatch")
    if len(f_ids) > max_seq_len:
        raise ValueError(f"sequence {len(f_ids)} > --max-seq-len {max_seq_len}")
    weights = token_field_weights(processor.tokenizer, f_ids, len(p_ids))
    return (move_inputs(full, device), len(p_ids),
            torch.tensor(weights, device=device))


_LOGITS_TO_KEEP_OK = [True]


def forward_logits(model, inputs, positions):
    """Logits only at `positions` (memory-safe); falls back to full logits."""
    if _LOGITS_TO_KEEP_OK[0]:
        try:
            out = model(**inputs, use_cache=False, return_dict=True, logits_to_keep=positions)
            if out.logits.shape[1] == positions.numel():
                return out.logits[0]
        except TypeError:
            pass
        _LOGITS_TO_KEEP_OK[0] = False
        print("[INFO] logits_to_keep unsupported: computing full logits")
    out = model(**inputs, use_cache=False, return_dict=True)
    return out.logits[0, positions.to(out.logits.device)]


# ===========================================================================
# Training
# ===========================================================================
def train_step(model, processor, ctrl, row, args, a_id, b_id, blank=False):
    device = model_device(model)
    target = abstention_target(row.get("modality")) if blank else row["target"]
    y_abstain = 1.0 if (blank or row["gold_action"] == "ABSTAIN") else 0.0
    inputs, prompt_len, weights = build_inputs(processor, row, target, device,
                                               args.max_seq_len, blank=blank)
    positions = torch.nonzero(weights[1:] > 0, as_tuple=False).flatten()

    ctrl.rt.reset(read_pos=prompt_len - 1, capture=True, inject=True)
    try:
        logits = forward_logits(model, inputs, positions).float()
    except Exception:
        ctrl.rt.reset()
        raise
    heads = dict(ctrl.rt.logits)
    fused = ctrl.rt.fused
    targets = inputs["input_ids"][0].to(logits.device)[positions + 1]
    w = weights[positions + 1].to(logits.device)
    sft = (F.cross_entropy(logits, targets, reduction="none") * w).sum() / w.sum().clamp_min(1.0)

    y = torch.tensor([y_abstain], device=ctrl.device)
    layer_w = ctrl.visual_weight if blank else ctrl.auc_weight
    head_loss = sum(layer_w[k] * F.binary_cross_entropy_with_logits(heads[L], y)
                    for k, L in enumerate(ctrl.classify))
    fused_loss = F.binary_cross_entropy_with_logits(fused, y) if fused is not None else 0.0
    # Agreement: the LM's own A/B readout should follow the layer decision.
    first = int((positions == prompt_len - 1).nonzero().flatten()[0])
    lm_ab = torch.stack([logits[first, b_id], logits[first, a_id]]).to(ctrl.device)  # [abstain, answer]
    lm_logp = F.log_softmax(lm_ab, dim=0)
    if fused is not None:
        p = torch.sigmoid(fused.detach())[0]
        agree = -(p * lm_logp[0] + (1 - p) * lm_logp[1])
    else:
        agree = torch.zeros((), device=ctrl.device)
    loss = (sft.to(ctrl.device) + args.head_weight * (head_loss + fused_loss)
            + args.agree_weight * agree)
    scalar = lambda v: float(v.detach()) if torch.is_tensor(v) else float(v)
    return loss, {"sft": scalar(sft), "heads": scalar(head_loss), "fused": scalar(fused_loss)}


def train_epoch(model, processor, ctrl, rows, args, optimizer, scheduler, epoch, a_id, b_id):
    model.train() if args.dropout_train else model.eval()
    ctrl.train()
    rng = np.random.default_rng(args.seed + epoch)
    order = rng.permutation(len(rows)).tolist()
    jobs = [(i, False) for i in order]
    blanks = [i for i in order if rng.random() < args.blank_prob]
    jobs += [(i, True) for i in blanks]
    rng.shuffle(jobs)
    params = [p for g in optimizer.param_groups for p in g["params"]]
    stats, done, skipped = Counter(), 0, 0
    optimizer.zero_grad(set_to_none=True)
    for n, (i, blank) in enumerate(jobs, 1):
        try:
            loss, parts = train_step(model, processor, ctrl, rows[i], args, a_id, b_id, blank=blank)
        except ValueError as exc:
            ctrl.rt.reset()
            skipped += 1
            print(f"[SKIP] {rows[i]['sample_id']} blank={blank}: {exc}")
            continue
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at {rows[i]['sample_id']}")
        try:
            (loss / args.grad_accum).backward()
        finally:
            ctrl.rt.reset()
        done += 1
        for k, v in parts.items():
            stats[k] += v
        if done % args.grad_accum == 0 or n == len(jobs):
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
        if done % 50 == 0:
            print(f"[EPOCH {epoch}] {done}/{len(jobs)} sft={stats['sft']/done:.4f} "
                  f"heads={stats['heads']/done:.4f} fused={stats['fused']/done:.4f} "
                  f"g_dec={float(ctrl.g_dec):+.4f}", flush=True)
    print(f"[EPOCH {epoch}] done={done} skipped={skipped} blank_copies={len(blanks)} "
          f"mean sft={stats['sft']/max(done,1):.4f} g_dec={float(ctrl.g_dec):+.4f} "
          f"fusion={torch.softmax(ctrl.fusion,0).detach().cpu().numpy().round(3).tolist()}")
    model.eval()
    ctrl.eval()


# ===========================================================================
# Inference
# ===========================================================================
@torch.no_grad()
def prefill(model, processor, ctrl, row, args, a_id, b_id, blank=False, inject=True):
    """One forward on the prompt: per-layer head probs, fused p, LM A/B margin."""
    device = model_device(model)
    inputs, prompt_len, _ = build_inputs(processor, row, None, device, args.max_seq_len, blank=blank)
    ctrl.rt.reset(read_pos=prompt_len - 1, capture=True, inject=inject)
    try:
        logits = forward_logits(model, inputs, torch.tensor([prompt_len - 1], device=device))
        heads = {L: float(torch.sigmoid(v)[0]) for L, v in ctrl.rt.logits.items()}
        fused_logit = ctrl.fused_logit(ctrl.rt.logits)
    finally:
        ctrl.rt.reset()
    margin = float(logits[0, a_id].float() - logits[0, b_id].float())
    return {"heads": heads, "fused": float(torch.sigmoid(fused_logit)[0]), "lm_margin": margin,
            "inputs": inputs, "prompt_len": prompt_len}


def layer_decision(pre, classify, tau):
    """Early exit: the earliest confident knowledge layer commits A/B."""
    for L in classify:
        p = pre["heads"][L]
        if max(p, 1 - p) >= tau:
            return ("ABSTAIN" if p >= 0.5 else "ANSWER"), L
    return ("ABSTAIN" if pre["fused"] >= 0.5 else "ANSWER"), "fused"


@torch.no_grad()
def generate(model, processor, ctrl, pre, args, decision_code_id=None):
    inputs = dict(pre["inputs"])
    prompt_len = pre["prompt_len"]
    if decision_code_id is not None:
        ids = inputs["input_ids"]
        extra = torch.tensor([[decision_code_id]], device=ids.device, dtype=ids.dtype)
        for key, value in list(inputs.items()):
            if torch.is_tensor(value) and value.ndim == 2 and value.shape == ids.shape:
                if key == "input_ids":
                    inputs[key] = torch.cat([value, extra], dim=1)
                else:
                    fill = 1 if key == "attention_mask" else 0
                    inputs[key] = torch.cat([value, torch.full_like(value[:, :1], fill)], dim=1)
    p = torch.tensor([pre["fused"]], device=ctrl.device)
    ctrl.rt.reset(read_pos=prompt_len - 1, capture=False, inject=True, fixed_p=p)
    try:
        tok = processor.tokenizer
        out = model.generate(**inputs, do_sample=False, max_new_tokens=args.max_new_tokens,
                             use_cache=True,
                             pad_token_id=tok.pad_token_id if tok.pad_token_id is not None
                             else tok.eos_token_id)
    finally:
        ctrl.rt.reset()
    text = tok.decode(out[0, prompt_len:], skip_special_tokens=True,
                      clean_up_tokenization_spaces=False)
    return text


def prefill_all(model, processor, ctrl, rows, args, a_id, b_id, ablate=True):
    records = []
    for k, row in enumerate(rows, 1):
        pre = prefill(model, processor, ctrl, row, args, a_id, b_id)
        rec = {"sample_id": row["sample_id"], "gold": row["gold_action"],
               "heads": pre["heads"], "fused": pre["fused"], "lm_margin": pre["lm_margin"]}
        if ablate:
            rec["lm_margin_no_injection"] = prefill(model, processor, ctrl, row, args, a_id, b_id,
                                                    inject=False)["lm_margin"]
        records.append(rec)
        if k % 100 == 0:
            print(f"[PREFILL] {k}/{len(rows)}", flush=True)
    return records


def select_decision_policy(pre_records, classify):
    """VALIDATION only: tau for early exit and layer-vs-model decision mode."""
    gold = [r["gold"] for r in pre_records]
    best = None
    for tau in [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 1.01]:
        pred = [layer_decision(r, classify, tau)[0] for r in pre_records]
        acc = accuracy_score(gold, pred)
        if best is None or acc > best[0] + 1e-12:
            best = (acc, tau)
    lm = ["ANSWER" if r["lm_margin"] > 0 else "ABSTAIN" for r in pre_records]
    lm_acc = accuracy_score(gold, lm)
    mode = "layer" if best[0] > lm_acc else "model"
    return {"tau": best[1], "layer_mode_val_accuracy": best[0],
            "model_mode_val_accuracy": lm_acc, "mode": mode}


def mechanistic_report(pre_records, classify, plan, policy):
    y = [int(r["gold"] == "ABSTAIN") for r in pre_records]
    report = {"layers_report_1based": plan["classify_layers_report"],
              "gate_layer_report_1based": plan["gate_layer_report"], "per_layer": {}}
    for L, Lr in zip(classify, plan["classify_layers_report"]):
        s = [r["heads"][L] for r in pre_records]
        report["per_layer"][f"L{Lr}"] = {
            "head_auc": safe_auc(y, s),
            "head_accuracy": float(np.mean([(v >= 0.5) == bool(t) for v, t in zip(s, y)]))}
    report["fused_auc"] = safe_auc(y, [r["fused"] for r in pre_records])
    lm_on = [int(r["lm_margin"] < 0) for r in pre_records]
    report["lm_decision_accuracy_injection_on"] = float(np.mean([a == b for a, b in zip(lm_on, y)]))
    if "lm_margin_no_injection" in pre_records[0]:
        lm_off = [int(r["lm_margin_no_injection"] < 0) for r in pre_records]
        report["lm_decision_accuracy_injection_off"] = float(np.mean([a == b for a, b in zip(lm_off, y)]))
        report["injection_flip_rate"] = float(np.mean([a != b for a, b in zip(lm_on, lm_off)]))
    commits = Counter(str(layer_decision(r, classify, policy["tau"])[1]) for r in pre_records)
    names = {str(L): f"L{Lr}" for L, Lr in zip(classify, plan["classify_layers_report"])}
    report["commit_layer_histogram"] = {names.get(k, k): v for k, v in commits.items()}
    report["policy"] = policy
    return report


@torch.no_grad()
def blank_check(model, processor, ctrl, rows, args, a_id, b_id, plan, limit):
    rows = rows[:limit]
    if not rows:
        return {}
    clean, blank = [], []
    for row in rows:
        clean.append(prefill(model, processor, ctrl, row, args, a_id, b_id))
        blank.append(prefill(model, processor, ctrl, row, args, a_id, b_id, blank=True))
    out = {"n": len(rows), "per_layer_mean_p_abstain": {}}
    for L, Lr in zip(ctrl.classify, plan["classify_layers_report"]):
        out["per_layer_mean_p_abstain"][f"L{Lr}"] = {
            "original": float(np.mean([c["heads"][L] for c in clean])),
            "blank": float(np.mean([b["heads"][L] for b in blank]))}
    out["lm_abstain_rate_original"] = float(np.mean([c["lm_margin"] < 0 for c in clean]))
    out["lm_abstain_rate_blank"] = float(np.mean([b["lm_margin"] < 0 for b in blank]))
    return out


def evaluate_generation(model, processor, ctrl, rows, args, a_id, b_id, policy, limit=0,
                        tag="validation"):
    if limit and limit < len(rows):
        idx = np.linspace(0, len(rows) - 1, limit, dtype=int).tolist()
        rows = [rows[i] for i in idx]
    records = []
    for k, row in enumerate(rows, 1):
        pre = prefill(model, processor, ctrl, row, args, a_id, b_id)
        if policy["mode"] == "layer":
            decision, commit = layer_decision(pre, ctrl.classify, policy["tau"])
            code = a_id if decision == "ANSWER" else b_id
            # The decoded text already starts with the forced code token.
            text = generate(model, processor, ctrl, pre, args, decision_code_id=code)
        else:
            commit = "model"
            text = generate(model, processor, ctrl, pre, args)
        action = parse_action(text) or "PARSE_ERROR"
        resp = parse_response(text)
        right = action == row["gold_action"]
        joint = right if row["gold_action"] == "ABSTAIN" else (
            right and response_correct(resp, row["reference_response"]))
        records.append({
            "sample_id": row["sample_id"], "gold_action": row["gold_action"],
            "pred_action": action, "commit_layer": str(commit),
            "fused_p_abstain": pre["fused"], "lm_margin": pre["lm_margin"],
            **{f"p_abstain_L{L+1}": pre["heads"][L] for L in ctrl.classify},
            "action_correct": int(right), "joint_correct": int(joint),
            "parse_complete": int(parse_complete(text)),
            "rouge_l": rouge_l(resp, row["reference_response"]),
            "response": resp, "reference_response": row["reference_response"], "text": text})
        if k % 50 == 0:
            print(f"[GEN {tag}] {k}/{len(rows)}", flush=True)
    return records, summarize(records, tag)


def summarize(records, tag, prefix="lk_sft"):
    gold = [r["gold_action"] for r in records]
    pred = [r["pred_action"] for r in records]
    abst = [i for i, g in enumerate(gold) if g == "ABSTAIN"]
    ans = [i for i, g in enumerate(gold) if g == "ANSWER"]
    return {
        "split": tag, "n": len(records),
        f"{prefix}_action_accuracy": accuracy_score(gold, pred),
        f"{prefix}_action_macro_f1": f1_score(gold, pred, labels=list(ACTIONS), average="macro",
                                               zero_division=0),
        # UAR: gold ABSTAIN answered (unsafe); OAR: gold ANSWER abstained (over-abstention)
        f"{prefix}_UAR": float(np.mean([pred[i] == "ANSWER" for i in abst])) if abst else float("nan"),
        f"{prefix}_OAR": float(np.mean([pred[i] == "ABSTAIN" for i in ans])) if ans else float("nan"),
        f"{prefix}_response_rouge_l": float(np.mean([r["rouge_l"] for r in records])),
        f"{prefix}_joint_correctness": float(np.mean([r["joint_correct"] for r in records])),
        f"{prefix}_parse_complete": float(np.mean([r["parse_complete"] for r in records])),
    }


def sft_baseline_summary(path, rows):
    """Same metrics for cached structured-SFT validation outputs (optional)."""
    path = Path(path)
    if not path.is_file():
        return None
    cache = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            obj = json.loads(line)
            cache[str(obj.get("sample_id"))] = obj.get("baseline_text", "")
    records = []
    for row in rows:
        if row["sample_id"] not in cache:
            return None
        text = cache[row["sample_id"]]
        action = parse_action(text) or "PARSE_ERROR"
        resp = parse_response(text)
        right = action == row["gold_action"]
        records.append({"gold_action": row["gold_action"], "pred_action": action,
                        "rouge_l": rouge_l(resp, row["reference_response"]),
                        "joint_correct": int(right if row["gold_action"] == "ABSTAIN" else
                                             right and response_correct(resp, row["reference_response"])),
                        "parse_complete": int(parse_complete(text))})
    return summarize(records, "validation", prefix="sft")


def write_csv(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)


# ===========================================================================
# Main
# ===========================================================================
def build_parser():
    p = argparse.ArgumentParser(description="EPICA (SFT-LK) training / evaluation on Fin-Abs")
    p.add_argument("--model", choices=sorted(MODELS), default="qwen25_vl_7b")
    p.add_argument("--data-dir", type=Path, default=REPO_DIR / "data",
                   help="Folder with <split>.csv files (and the images they reference)")
    p.add_argument("--split-csv", default="{split}.csv", help="File-name pattern of the split CSVs")
    p.add_argument("--reference-dir", type=Path, default=None,
                   help="Optional folder with reference CSVs that fill empty fields by sample_id")
    p.add_argument("--checkpoint", default=None, help="HF id or local model directory (default: --model's)")
    p.add_argument("--attn-implementation", default=None, help="e.g. eager / sdpa (default: library default)")
    p.add_argument("--layer-knowledge", type=Path, default=None,
                   help="Layer knowledge (.json or .md); default layer_knowledge/<model>.json")
    p.add_argument("--stage2-adapter", type=Path, default=None,
                   help="Structured-SFT adapter to continue from (default outputs/sft/<model>/adapter)")
    p.add_argument("--fresh-lora", action="store_true",
                   help="Start a new LoRA from the base model instead of the structured-SFT adapter")
    p.add_argument("--output", type=Path, default=None, help="Default outputs/epica/<model>")
    p.add_argument("--mode", choices=["train", "evaluate"], default="train")
    p.add_argument("--epochs", type=int, default=TRAINING_EPOCHS)
    p.add_argument("--lora-rank", type=int, default=32, help="Only used with --fresh-lora")
    p.add_argument("--lr", type=float, default=None,
                   help="LoRA lr (default 5e-5 when continuing from the SFT adapter, 2e-4 fresh)")
    p.add_argument("--controller-lr", type=float, default=1e-3)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--head-weight", type=float, default=0.5)
    p.add_argument("--agree-weight", type=float, default=0.2)
    p.add_argument("--blank-prob", type=float, default=0.15,
                   help="Fraction of TRAIN items also seen with a blank image + abstention target")
    p.add_argument("--numeric-weight", type=float, default=2.0)
    p.add_argument("--knowledge-prompt", action="store_true",
                   help="Also prepend a fixed text summary of the layer knowledge to every prompt")
    p.add_argument("--max-seq-len", type=int, default=4096)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--val-gen-limit", type=int, default=300,
                   help="VALIDATION items generated after each epoch (0 = all)")
    p.add_argument("--blank-eval-limit", type=int, default=100)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--sft-baseline-cache", type=Path, default=None,
                   help="Optional JSONL {sample_id, baseline_text} of SFT validation outputs for comparison")
    p.add_argument("--load-4bit", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--grad-ckpt", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--dropout-train", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--evaluate-test", action="store_true")
    p.add_argument("--pilot", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--export-knowledge", type=Path, default=None, metavar="CONTROLLER_PT",
                   help="Write the layer plan of a trained controller.pt to layer_knowledge/<model>.json and exit")
    p.add_argument("--export-out", type=Path, default=None)
    p.add_argument("--self-test", action="store_true")
    return p


def resolve_knowledge_path(args):
    if args.layer_knowledge is not None:
        return args.layer_knowledge
    path = REPO_DIR / "layer_knowledge" / f"{MODEL_KEY}.json"
    if path.is_file():
        return path
    if MODEL_KEY == DEFAULT_KNOWLEDGE["model"]:
        return None                                   # built-in Qwen2.5-VL-7B values
    raise FileNotFoundError(
        f"No layer knowledge for {MODEL_KEY}: add layer_knowledge/{MODEL_KEY}.json (see "
        "layer_knowledge/README.md), pass --layer-knowledge, or export it from a trained "
        "controller.pt with --export-knowledge.")


def export_knowledge(controller_pt, out_path):
    """controller.pt (plan) -> layer_knowledge JSON that reproduces the plan exactly."""
    saved = torch.load(controller_pt, map_location="cpu", weights_only=False)
    plan = saved["plan"]
    layers = [int(L) for L in plan["classify_layers_report"]]
    roles = {k: [int(L) for L in v] for k, v in plan["roles_report"].items()}
    know = {"model": MODEL_KEY, "index_base": 1, "n_layers": MODELS[MODEL_KEY]["n_layers"],
            "roles": roles,
            "plan_weights": {"auc_weight": {str(L): float(w) for L, w in zip(layers, plan["auc_weight"])},
                             "visual_weight": {str(L): float(w) for L, w in zip(layers, plan["visual_weight"])}},
            "source": f"exported from a trained controller ({Path(controller_pt).name})"}
    save_json(out_path, know)
    check = knowledge_plan(parse_layer_knowledge(out_path), MODELS[MODEL_KEY]["n_layers"])
    for key in ("classify_layers", "gate_layer"):
        assert check[key] == plan[key], (key, check[key], plan[key])
    assert np.allclose(check["auc_weight"], plan["auc_weight"]) and np.allclose(check["visual_weight"], plan["visual_weight"])
    print(f"[EXPORT] {out_path}: heads at {layers}, gate L{plan['gate_layer_report']} (plan reproduced exactly)")
    return know


def trainable_state(model, ctrl):
    lora = {k: v.detach().cpu().clone() for k, v in model.named_parameters() if v.requires_grad}
    return {"lora": lora, "ctrl": {k: v.detach().cpu().clone() for k, v in ctrl.state_dict().items()}}


def restore_state(model, ctrl, state):
    params = dict(model.named_parameters())
    with torch.no_grad():
        for k, v in state["lora"].items():
            params[k].copy_(v.to(params[k].device))
    ctrl.load_state_dict(state["ctrl"])


def main():
    args = build_parser().parse_args()
    if args.self_test:
        _self_test()
        return
    set_model(args.model)
    if args.export_knowledge is not None:
        export_knowledge(args.export_knowledge,
                         args.export_out or REPO_DIR / "layer_knowledge" / f"{MODEL_KEY}.json")
        return
    args.checkpoint = args.checkpoint or MODEL_ID
    global NUMERIC_TOKEN_WEIGHT
    NUMERIC_TOKEN_WEIGHT = float(args.numeric_weight)
    seed_everything(args.seed)
    if args.pilot:
        args.max_train_samples = 16
        args.val_gen_limit = 8
        args.blank_eval_limit = 4
    args.output = args.output or (REPO_DIR / "outputs" / ("epica_pilot" if args.pilot else "epica") / MODEL_KEY)
    args.output.mkdir(parents=True, exist_ok=True)
    stage2 = args.stage2_adapter or (REPO_DIR / "outputs" / "sft" / MODEL_KEY / "adapter")
    best_dir = args.output / "best"

    know = parse_layer_knowledge(resolve_knowledge_path(args))
    print(f"[KNOWLEDGE] source={know['source']}")
    print(f"[KNOWLEDGE] roles(1-based)={know['roles']}  probe AUC={know['probe_auc']}")

    if args.mode == "train":
        args.adapter_to_load = None if args.fresh_lora else (stage2 if Path(stage2).is_dir() else None)
        if args.adapter_to_load is None and not args.fresh_lora:
            print(f"[WARNING] structured-SFT adapter not found at {stage2}; starting a fresh LoRA")
        if args.lr is None:
            args.lr = 2e-4 if args.adapter_to_load is None else 5e-5
        model, processor = load_model(args, trainable=True)
    else:
        if not (best_dir / "adapter").is_dir():
            raise FileNotFoundError(f"No trained model at {best_dir}; run --mode train first")
        args.adapter_to_load = best_dir / "adapter"
        model, processor = load_model(args, trainable=False)

    name, layers = find_decoder_layers(model)
    saved = None
    if args.mode == "evaluate":
        saved = torch.load(best_dir / "controller.pt", map_location="cpu", weights_only=False)
        plan = saved["plan"]
        print("[PLAN] evaluate mode: using the layer plan stored with the trained model")
    else:
        plan = knowledge_plan(know, len(layers))
    print(f"[MODEL] decoder={name} layers={len(layers)}")
    print(f"[PLAN] classify at 1-based {plan['classify_layers_report']} "
          f"(0-based {plan['classify_layers']}), AUC weights={np.round(plan['auc_weight'],3).tolist()}, "
          f"visual weights={np.round(plan['visual_weight'],3).tolist()}")
    print(f"[PLAN] decision injection at 1-based L{plan['gate_layer_report']} "
          f"(0-based {plan['gate_layer']}); terminal {plan['terminal_layers_report']} untouched")
    save_json(args.output / "layer_knowledge_used.json", {"knowledge": know, "plan": plan})

    a_id, b_id = decision_token_ids(processor.tokenizer)
    hidden = model.get_output_embeddings().weight.shape[1]
    ctrl = LayerKnowledgeController(hidden, plan, unembedding_direction(model, a_id, b_id))
    ctrl.to(model_device(model))
    ctrl.install(layers)

    kprompt = knowledge_prompt_text(plan) if args.knowledge_prompt else ""
    refs = _reference_files(args.reference_dir)
    load = lambda split: load_split(args.data_dir, split, refs, kprompt, args.split_csv, args.reference_dir)
    train_rows = load("train")
    val_rows = load("validation")
    if {r["sample_id"] for r in train_rows} & {r["sample_id"] for r in val_rows}:
        raise ValueError("TRAIN/VALIDATION sample_id overlap")
    if args.pilot:
        train_rows = ([r for r in train_rows if r["gold_action"] == "ANSWER"][:8]
                      + [r for r in train_rows if r["gold_action"] == "ABSTAIN"][:8])
        val_rows = ([r for r in val_rows if r["gold_action"] == "ANSWER"][:4]
                    + [r for r in val_rows if r["gold_action"] == "ABSTAIN"][:4])
    elif args.max_train_samples > 0:
        train_rows = train_rows[:args.max_train_samples]

    sft_cache = args.sft_baseline_cache

    if args.mode == "train":
        lora_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW([
            {"params": lora_params, "lr": args.lr, "weight_decay": 0.0},
            {"params": list(ctrl.parameters()), "lr": args.controller_lr, "weight_decay": 0.01},
        ])
        n_jobs = int(len(train_rows) * (1 + args.blank_prob))
        total = max(1, args.epochs * math.ceil(n_jobs / args.grad_accum))
        warm = max(1, int(0.03 * total))
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
        print(f"[SCHEDULE] {args.epochs} epochs, ~{n_jobs} items/epoch, {total} optimizer steps, "
              f"lr LoRA={args.lr} controller={args.controller_lr}")
        best, history = None, []
        for epoch in range(1, args.epochs + 1):
            print(f"\n========== [EPOCH {epoch}/{args.epochs}] EPICA ({MODEL_KEY}) ==========")
            train_epoch(model, processor, ctrl, train_rows, args, optimizer, scheduler, epoch, a_id, b_id)
            pre = prefill_all(model, processor, ctrl, val_rows, args, a_id, b_id)
            policy = select_decision_policy(pre, ctrl.classify)
            mech = mechanistic_report(pre, ctrl.classify, plan, policy)
            _, summ = evaluate_generation(model, processor, ctrl, val_rows, args, a_id, b_id,
                                          policy, limit=args.val_gen_limit, tag=f"validation_epoch{epoch}")
            entry = {"epoch": epoch, "policy": policy, "mechanistic": mech, "generation": summ}
            history.append(entry)
            save_json(args.output / f"epoch{epoch}_validation.json", entry)
            print(json.dumps(entry, indent=2, default=str))
            score = (summ["lk_sft_joint_correctness"], summ["lk_sft_action_accuracy"])
            edir = args.output / f"epoch{epoch}"
            model.save_pretrained(str(edir / "adapter"))
            torch.save({"ctrl": ctrl.state_dict(), "policy": policy, "plan": plan}, edir / "controller.pt")
            if best is None or score > best[0]:
                best = (score, epoch, trainable_state(model, ctrl), policy)
                print(f"[SELECT] epoch {epoch} is best so far (validation joint, action) = {score}")
        _, best_epoch, state, policy = best
        restore_state(model, ctrl, state)
        model.save_pretrained(str(best_dir / "adapter"))
        torch.save({"ctrl": ctrl.state_dict(), "policy": policy, "plan": plan,
                    "best_epoch": best_epoch}, best_dir / "controller.pt")
        save_json(args.output / "training_history.json", history)
        print(f"[DONE TRAIN] best epoch = {best_epoch}; saved {best_dir}")
    else:
        ctrl.load_state_dict(saved["ctrl"])
        ctrl.to(model_device(model))
        policy = saved["policy"]





if __name__ == "__main__":
    main()
