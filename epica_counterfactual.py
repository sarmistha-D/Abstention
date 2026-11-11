#!/usr/bin/env python3
"""Anonymous Fin-Abs held-out SFT counterfactual evaluator.

Runs BASE + Stage I SFT adapters (not the LK-SFT controller) for four VLMs.
Evaluates ORIGINAL, BLACK, BLUR, MISMATCH, RANDOM_MASK and QUESTION_ONLY.
Perturbed inputs have no verified gold labels; only decision-flip rates and
A/B-normalized decision changes are reported for the interventions.
"""
from __future__ import annotations
import os, sys

if '--gpu' in sys.argv[1:]:
    GPU = sys.argv[sys.argv.index('--gpu') + 1]
if GPU != '':
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
import argparse, base64, csv, gc, hashlib, io, json, math, random, re, time, traceback
from collections import Counter, defaultdict
from html import escape
from pathlib import Path

# ================================ CONFIG ===============================
ROOT = Path(__file__).resolve().parent / 'data'
TEST_FILE = ROOT / 'finabs_step2_outputs' / 'test_v2_candidate.csv'
PAIR_FILE = ROOT / 'same_image_opposite_action_PAIR_CANDIDATES.csv'  # optional
STAGE2_ROOT = ROOT / 'finabs_stage2_sft_mechanistic_v2'
OUTPUT_ROOT = Path(__file__).resolve().parent / 'runs' / 'counterfactual_sft'
MODELS = {
    'qwen25_vl_3b': 'Qwen/Qwen2.5-VL-3B-Instruct',
    'qwen25_vl_7b': 'Qwen/Qwen2.5-VL-7B-Instruct',
    'gemma3_4b': 'google/gemma-3-4b-it',
    'gemma3_12b': 'google/gemma-3-12b-it',
}
ADAPTER_RUN_TAG = 'full'            # adapters from the full Stage II run
SFT_VARIANT = 'structured'
MAX_QUESTIONS = 0                  
GENERATE_TEXT = True                
MAX_HTML_EXAMPLES = 16
SEED = 42
MAX_NEW_TOKENS = 384
EVAL_BATCH_SIZE = 2               
ALLOW_UNVERIFIED_ADAPTER = False    
BOOTSTRAP = 2000

MAX_IMAGE_EDGE = 896
QWEN_MIN_PIXELS = 64 * 28 * 28
QWEN_MAX_PIXELS = 512 * 28 * 28
USE_4BIT = True
# ---- interventions ----
BLUR_RADIUS_FRACTION = 0.025
RANDOM_MASK_FRACTION = 0.25
MASK_GRID = 8
PROMPT_VERSION = 'FINABS_SFT_ACTIONFIRST_CF_V3'
CONDITIONS = ['ORIGINAL', 'BLACK', 'BLUR', 'MISMATCH', 'RANDOM_MASK', 'QUESTION_ONLY']
# ======================================================================

import numpy as np
import torch
from PIL import Image, ImageOps, ImageFilter, ImageDraw

# EXACT Stage II prompt (the adapter learned this serialization).
INSTRUCTION = (
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
FIELD_KEYS = ('EVIDENCE_STATUS', 'EVIDENCE_MODALITY', 'RESPONSE', 'DECISION_REASONING', 'JUSTIFICATION')


# ------------------------------------------------------------------ utils
def read_csv(path):
    if not path.is_file(): raise FileNotFoundError(path)
    with path.open('r', encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))


def write_csv(path, rows, columns=None):
    rows = list(rows); path.parent.mkdir(parents=True, exist_ok=True)
    if columns is None: columns = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore'); w.writeheader(); w.writerows(rows)


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding='utf-8')


def stable_seed(text):
    return SEED + int(hashlib.sha256(str(text).encode()).hexdigest()[:8], 16)


def sigmoid(x):
    x = max(-60.0, min(60.0, float(x))); return 1.0 / (1.0 + math.exp(-x))


def ids_sha256(ids):
    return hashlib.sha256('\n'.join(sorted(ids)).encode()).hexdigest()


# ------------------------------------------------------------------ data
def load_test():
    rows = read_csv(TEST_FILE)
    by_id = {r['sample_id']: r for r in rows}
    if len(by_id) != len(rows): raise ValueError('Duplicate test sample IDs.')
    for r in rows:
        r['expected_action'] = r['expected_action'].strip().lower()
        if r['expected_action'] not in ('answer', 'abstain'): raise ValueError('Invalid action: ' + r['sample_id'])
        if not r.get('v2_image_sha256'): raise ValueError('Missing image hash: ' + r['sample_id'])
        img = Path(r['v2_resolved_image'])
        img = img if img.is_absolute() else ROOT / img
        if not img.is_file(): raise FileNotFoundError(f"Image missing for {r['sample_id']}: {img}")
        r['v2_resolved_image'] = str(img)
    return rows, by_id


def find_pairs(test_rows, by_id):
    found = {}
    by_sha = defaultdict(list)
    for r in test_rows: by_sha[r['v2_image_sha256']].append(r)
    for sha, group in by_sha.items():
        ans = sorted(r['sample_id'] for r in group if r['expected_action'] == 'answer')
        abs_ = sorted(r['sample_id'] for r in group if r['expected_action'] == 'abstain')
        for a, b in zip(ans, abs_): found[(a, b)] = 'derived_from_test_sha256'
    if PAIR_FILE.is_file():
        for p in read_csv(PAIR_FILE):
            a, b = p.get('answer_sample_id'), p.get('abstain_sample_id')
            if a in by_id and b in by_id and by_id[a]['v2_image_sha256'] == by_id[b]['v2_image_sha256'] \
                    and by_id[a]['expected_action'] == 'answer' and by_id[b]['expected_action'] == 'abstain':
                found.setdefault((a, b), 'pair_file')
    return [{'answer_sample_id': a, 'abstain_sample_id': b, 'pair_source': s} for (a, b), s in sorted(found.items())]


def select_cases(test_rows, pairs):
    if not MAX_QUESTIONS: return sorted(r['sample_id'] for r in test_rows)
    rng = random.Random(SEED); selected = set()
    ordered = list(pairs); rng.shuffle(ordered)
    for p in ordered:                                  # complete pairs first
        prop = selected | {p['answer_sample_id'], p['abstain_sample_id']}
        if len(prop) <= MAX_QUESTIONS: selected = prop
    rest = [r for r in test_rows if r['sample_id'] not in selected]
    yes = [r['sample_id'] for r in rest if r['expected_action'] == 'answer']
    no = [r['sample_id'] for r in rest if r['expected_action'] == 'abstain']
    rng.shuffle(yes); rng.shuffle(no)
    label = {r['sample_id']: r['expected_action'] for r in test_rows}
    while len(selected) < MAX_QUESTIONS and (yes or no):      # FIX: label-balanced fill
        n_abs = sum(label[s] == 'abstain' for s in selected)
        src = no if no and (n_abs < len(selected) - n_abs or not yes) else yes
        selected.add(src.pop())
    return sorted(selected)


def build_donors(test_rows, selected_ids, by_id):
    pool = sorted(test_rows, key=lambda r: r['sample_id'])
    donors = {}
    for sid in selected_ids:
        src = by_id[sid]; fam = (src.get('v2_family_id') or '').strip()
        cands = [r for r in pool if r['sample_id'] != sid and r['v2_image_sha256'] != src['v2_image_sha256']
                 and not (fam and (r.get('v2_family_id') or '').strip() == fam)]
        if not cands: raise RuntimeError('No mismatched donor for ' + sid)
        donors[sid] = cands[stable_seed(sid) % len(cands)]
    return donors


# ------------------------------------------------------------------ interventions
def base_image(row):
    with Image.open(row['v2_resolved_image']) as im:
        img = ImageOps.exif_transpose(im).convert('RGB')
        if max(img.size) > MAX_IMAGE_EDGE: img.thumbnail((MAX_IMAGE_EDGE, MAX_IMAGE_EDGE), Image.Resampling.LANCZOS)
        return img.copy()


def random_mask_image(image, sample_id):
    out = image.copy(); w, h = out.size
    rng = random.Random(stable_seed(sample_id))
    cells = rng.sample(range(MASK_GRID * MASK_GRID), max(1, round(MASK_GRID * MASK_GRID * RANDOM_MASK_FRACTION)))
    d = ImageDraw.Draw(out)
    for c in cells:
        r_, c_ = divmod(c, MASK_GRID)
        d.rectangle((int(c_ * w / MASK_GRID), int(r_ * h / MASK_GRID),
                     int((c_ + 1) * w / MASK_GRID), int((r_ + 1) * h / MASK_GRID)), fill=(0, 0, 0))
    return out


def intervention_image(row, condition, donors):
    if condition == 'QUESTION_ONLY': return None
    if condition == 'MISMATCH': return base_image(donors[row['sample_id']])
    img = base_image(row)
    if condition == 'ORIGINAL': return img
    if condition == 'BLACK': return Image.new('RGB', img.size, (0, 0, 0))
    if condition == 'BLUR': return img.filter(ImageFilter.GaussianBlur(radius=max(5.0, max(img.size) * BLUR_RADIUS_FRACTION)))
    if condition == 'RANDOM_MASK': return random_mask_image(img, row['sample_id'])
    raise ValueError(condition)


def messages_for(row, condition, donors):
    img = intervention_image(row, condition, donors)
    content = ([{'type': 'image', 'image': img}] if img is not None else []) + \
              [{'type': 'text', 'text': INSTRUCTION + row['question'].strip()}]
    return [{'role': 'user', 'content': content}]


# ------------------------------------------------------------------ model (IDENTICAL to Stage II training)
def compute_dtype():
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def load_processor(tag, model_id):
    from transformers import AutoProcessor
    if tag.startswith('qwen'):
        # FIX: setting image_processor.max_pixels afterwards is ignored by preprocess(); without this the
        # model sees up to ~2x more visual tokens than during SFT.
        proc = AutoProcessor.from_pretrained(model_id, min_pixels=QWEN_MIN_PIXELS, max_pixels=QWEN_MAX_PIXELS)
        im = proc.image_processor; s = getattr(im, 'size', None)
        if isinstance(s, dict): im.size = {**s, 'shortest_edge': QWEN_MIN_PIXELS, 'longest_edge': QWEN_MAX_PIXELS}
        elif s is not None:
            try: s.shortest_edge = QWEN_MIN_PIXELS; s.longest_edge = QWEN_MAX_PIXELS
            except Exception: im.size = {'shortest_edge': QWEN_MIN_PIXELS, 'longest_edge': QWEN_MAX_PIXELS}
        probe = im(images=[Image.new('RGB', (MAX_IMAGE_EDGE, MAX_IMAGE_EDGE))], return_tensors='pt')
        n_tok = int(probe['image_grid_thw'][0].prod()) // (getattr(im, 'merge_size', 2) ** 2)
        if n_tok > QWEN_MAX_PIXELS // (28 * 28): raise RuntimeError(f'Qwen max_pixels not applied ({n_tok} tokens)')
    else:
        proc = AutoProcessor.from_pretrained(model_id)
    if proc.tokenizer.pad_token_id is None: proc.tokenizer.pad_token = proc.tokenizer.eos_token
    return proc


def check_adapter(tag, adapter, test_ids):
    """FIX: the old check read `train_rows`, a key the current Stage II scripts do not write, so every full
    run crashed. Now: same settings as training, trained on the TRAIN split, and no test id among its rows."""
    cfg_path = adapter.parent / 'train_config.json'
    if not (adapter / 'adapter_config.json').is_file():
        raise FileNotFoundError(f'SFT adapter missing for {tag}: {adapter}. Run the Stage II script first.')
    if not cfg_path.is_file(): raise FileNotFoundError(f'train_config.json missing next to {adapter}')
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    problems = []
    if cfg.get('sft_variant') != SFT_VARIANT: problems.append(f"variant {cfg.get('sft_variant')} != {SFT_VARIANT}")
    if cfg.get('quantization_4bit') not in (None, USE_4BIT): problems.append('4-bit setting differs from training')
    if tag.startswith('qwen') and cfg.get('qwen_max_pixels') not in (None, QWEN_MAX_PIXELS):
        problems.append(f"qwen_max_pixels {cfg.get('qwen_max_pixels')} != {QWEN_MAX_PIXELS}")
    ids_file = adapter.parent / 'train_sample_ids.txt'
    if ids_file.is_file():
        train_ids = set(ids_file.read_text().split())
        leak = train_ids & test_ids
        if leak: problems.append(f'{len(leak)} TEST ids were used in SFT, e.g. {sorted(leak)[:3]}')
    elif ALLOW_UNVERIFIED_ADAPTER:
        print(f'[WARNING] {tag}: train_sample_ids.txt missing -> train/test separation NOT verified '
              '(ALLOW_UNVERIFIED_ADAPTER=True). Report this in the paper.')
    else:
        problems.append('train_sample_ids.txt missing: cannot prove the adapter was trained on train only '
                        '(set ALLOW_UNVERIFIED_ADAPTER=True or pass --allow-unverified-adapter to continue)')
    if problems: raise RuntimeError(f'{tag} adapter check failed: ' + '; '.join(problems))
    print(f'{tag}: adapter verified (trained on {cfg.get("used_train_rows")} train rows, no test overlap).')
    return cfg


def load_sft_model(tag, model_id, test_ids):
    import transformers
    from transformers import BitsAndBytesConfig, AutoModelForImageTextToText
    from peft import PeftModel
    adapter = STAGE2_ROOT / ADAPTER_RUN_TAG / tag / SFT_VARIANT / 'adapter'
    cfg = check_adapter(tag, adapter, test_ids)
    major, minor = (int(x) for x in transformers.__version__.split('.')[:2])
    kw = {'device_map': 'auto', ('dtype' if (major, minor) >= (4, 56) else 'torch_dtype'): compute_dtype()}
    if tag.startswith('gemma'): kw['attn_implementation'] = 'sdpa'
    if USE_4BIT:
        # FIX: same quantization as training (vision tower / projector / lm_head NOT quantized).
        skip = ['vision_tower', 'multi_modal_projector', 'lm_head'] if tag.startswith('gemma') else ['visual', 'lm_head']
        kw['quantization_config'] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4',
                                                       bnb_4bit_use_double_quant=True,
                                                       bnb_4bit_compute_dtype=compute_dtype(), llm_int8_skip_modules=skip)
    C = AutoModelForImageTextToText
    proc = load_processor(tag, model_id)
    print('Loading', model_id, '+ adapter', adapter, flush=True)
    model = PeftModel.from_pretrained(C.from_pretrained(model_id, **kw), str(adapter), is_trainable=False).eval()
    fp = hashlib.sha256(b''.join(f'{f.name}:{f.stat().st_size}:{int(f.stat().st_mtime)}'.encode()
                                 for f in sorted(adapter.glob('adapter_*')))).hexdigest()[:16]
    return model, proc, f"stage2_sft/{tag}/{SFT_VARIANT}", fp


def choose_action_ids(tok):
    a = tok.encode('A', add_special_tokens=False); b = tok.encode('B', add_special_tokens=False)
    if len(a) != 1 or len(b) != 1 or a[0] == b[0]: raise RuntimeError('A/B are not distinct single tokens.')
    return a[0], b[0]


def parse_generation(text):
    lines = text.strip().splitlines()
    if not lines or lines[0].strip() not in ('A', 'B'): return {'generated_action': '', 'parse_status': 'unparsed'}
    fields = {}
    for ln in lines[1:]:
        if ':' in ln:
            k, v = ln.split(':', 1); k = k.strip()
            if k in FIELD_KEYS and k not in fields: fields[k] = '' if v.strip().upper() == 'N/A' else v.strip()
    return {'generated_action': 'answer' if lines[0].strip() == 'A' else 'abstain',
            'evidence_status': fields.get('EVIDENCE_STATUS', ''), 'evidence_modality': fields.get('EVIDENCE_MODALITY', ''),
            'response': fields.get('RESPONSE', ''), 'decision_reasoning': fields.get('DECISION_REASONING', ''),
            'justification': fields.get('JUSTIFICATION', ''),
            'parse_status': 'complete' if all(k in fields for k in FIELD_KEYS) else 'partial'}


def predict_batch(model, proc, rows, condition, donors):
    """One prefill per sample: p_abstain comes from the RAW first-step logits of generate()
    (FIX: the old code ran an extra full forward pass for every sample/condition)."""
    tok = proc.tokenizer; a_id, b_id = choose_action_ids(tok)
    msgs = [messages_for(r, condition, donors) for r in rows]
    kw = dict(tokenize=True, return_dict=True, return_tensors='pt', add_generation_prompt=True)
    if len(rows) > 1: tok.padding_side = 'left'; kw['padding'] = True
    enc = proc.apply_chat_template(msgs if len(rows) > 1 else msgs[0], **kw)
    enc = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in enc.items()}
    with torch.inference_mode():
        out = model.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW_TOKENS if GENERATE_TEXT else 1,
                             pad_token_id=tok.pad_token_id, use_cache=True,
                             return_dict_in_generate=True, output_logits=True)
    first = out.logits[0].float()
    texts = proc.batch_decode(out.sequences[:, enc['input_ids'].shape[1]:], skip_special_tokens=True,
                              clean_up_tokenization_spaces=False) if GENERATE_TEXT else [''] * len(rows)
    res = []
    for i, r in enumerate(rows):
        margin = float((first[i, b_id] - first[i, a_id]).cpu())
        p = sigmoid(margin); parsed = parse_generation(texts[i]) if GENERATE_TEXT else {}
        res.append({'sample_id': r['sample_id'], 'condition': condition, 'question': r['question'],
                    'legacy_gold_action': r['expected_action'], 'label_source': 'LEGACY_UNVERIFIED',
                    'image_sha256': r['v2_image_sha256'],
                    'donor_sample_id': donors[r['sample_id']]['sample_id'] if condition == 'MISMATCH' else '',
                    'p_abstain_AB_normalized': p, 'abstain_minus_answer_logit_margin': margin,
                    'forced_choice_action': 'abstain' if p >= 0.5 else 'answer',
                    'raw_generation': texts[i], 'generation_format': parsed.get('parse_status', 'not_generated'),
                    **{k: parsed.get(k, '') for k in ('generated_action', 'evidence_status', 'evidence_modality',
                                                      'response', 'decision_reasoning', 'justification')},
                    'status': 'ok', 'error': ''})
    del out, enc
    return res


# ------------------------------------------------------------------ metrics
def action_metrics(orig):
    labels = ['answer', 'abstain']; cm = Counter((r['legacy_gold_action'], r['forced_choice_action']) for r in orig)
    P, R, F = {}, {}, {}
    for l in labels:
        tp = cm[(l, l)]; fp = sum(cm[(o, l)] for o in labels if o != l); fn = sum(cm[(l, o)] for o in labels if o != l)
        P[l] = tp / (tp + fp) if tp + fp else 0.0; R[l] = tp / (tp + fn) if tp + fn else 0.0
        F[l] = 2 * P[l] * R[l] / (P[l] + R[l]) if P[l] + R[l] else 0.0
    an = sum(cm[('answer', p)] for p in labels); ab = sum(cm[('abstain', p)] for p in labels); n = len(orig)
    return {'n': n, 'legacy_action_accuracy': sum(cm[(l, l)] for l in labels) / n if n else None,
            'macro_precision': sum(P.values()) / 2, 'macro_recall': sum(R.values()) / 2, 'macro_f1': sum(F.values()) / 2,
            'answer_f1': F['answer'], 'abstain_f1': F['abstain'],
            'UAR': cm[('abstain', 'answer')] / ab if ab else None, 'OAR': cm[('answer', 'abstain')] / an if an else None,
            'confusion_matrix': {g: {p: cm[(g, p)] for p in labels} for g in labels},
            'note': 'Agreement with unverified legacy actions, not independently validated accuracy.'}


def boot_ci(values, seed):
    if len(values) < 2: return (None, None)
    rng = np.random.default_rng(seed); v = np.asarray(values, dtype=float)
    means = v[rng.integers(0, len(v), size=(BOOTSTRAP, len(v)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def visual_sensitivity(results):
    g = defaultdict(dict)
    for r in results:
        if r['status'] == 'ok': g[r['sample_id']][r['condition']] = r
    out = []
    for k, cond in enumerate(CONDITIONS[1:]):
        pairs = [(e['ORIGINAL'], e[cond]) for e in g.values() if 'ORIGINAL' in e and cond in e]
        if not pairs: continue
        absd = [abs(c['p_abstain_AB_normalized'] - o['p_abstain_AB_normalized']) for o, c in pairs]
        sgn = [c['p_abstain_AB_normalized'] - o['p_abstain_AB_normalized'] for o, c in pairs]
        lo, hi = boot_ci(absd, SEED + k)
        oa = [(o, c) for o, c in pairs if o['forced_choice_action'] == 'answer']
        ob = [(o, c) for o, c in pairs if o['forced_choice_action'] == 'abstain']
        out.append({'condition': cond, 'paired_n': len(pairs),
                    'V_mean_abs_delta_p_abstain': float(np.mean(absd)), 'V_ci95_low': lo, 'V_ci95_high': hi,
                    'mean_signed_delta_p_abstain': float(np.mean(sgn)),
                    'action_flip_rate': float(np.mean([o['forced_choice_action'] != c['forced_choice_action'] for o, c in pairs])),
                    'abstain_rate_under_condition': float(np.mean([c['forced_choice_action'] == 'abstain' for _, c in pairs])),
                    'original_predicted_answer_n': len(oa),
                    'answer_to_abstain_rate': float(np.mean([c['forced_choice_action'] == 'abstain' for _, c in oa])) if oa else None,
                    'original_predicted_abstain_n': len(ob),
                    'abstain_to_answer_rate': float(np.mean([c['forced_choice_action'] == 'answer' for _, c in ob])) if ob else None,
                    'interpretation': 'No gold label is assigned to altered-image controls.'})
    return out


def question_pairs(results, pairs):
    orig = {r['sample_id']: r for r in results if r['condition'] == 'ORIGINAL' and r['status'] == 'ok'}
    out = []
    for p in pairs:
        a, b = orig.get(p['answer_sample_id']), orig.get(p['abstain_sample_id'])
        if not a or not b: continue
        out.append({'image_sha256': a['image_sha256'], 'answer_sample_id': a['sample_id'], 'abstain_sample_id': b['sample_id'],
                    'pair_source': p['pair_source'], 'answer_question': a['question'], 'abstain_question': b['question'],
                    'answer_question_prediction': a['forced_choice_action'], 'abstain_question_prediction': b['forced_choice_action'],
                    'answer_question_p_abstain': a['p_abstain_AB_normalized'], 'abstain_question_p_abstain': b['p_abstain_AB_normalized'],
                    'delta_p_abstain_abstain_minus_answer_question': b['p_abstain_AB_normalized'] - a['p_abstain_AB_normalized'],
                    'action_changes_with_question': a['forced_choice_action'] != b['forced_choice_action'],
                    'matches_both_legacy_actions': a['forced_choice_action'] == 'answer' and b['forced_choice_action'] == 'abstain',
                    'answer_question_response': a.get('response', ''), 'answer_question_justification': a.get('justification', ''),
                    'abstain_question_response': b.get('response', ''), 'abstain_question_justification': b.get('justification', ''),
                    'pair_status': 'CANDIDATE_NOT_HUMAN_ADJUDICATED'})
    return out


# ------------------------------------------------------------------ HTML report
def thumb(image):
    image = image.copy(); image.thumbnail((440, 350)); buf = io.BytesIO()
    image.save(buf, format='JPEG', quality=75)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def html_report(results, by_id, donors, path, tag):
    g = defaultdict(dict)
    for r in results:
        if r['status'] == 'ok': g[r['sample_id']][r['condition']] = r
    ranked = []
    for sid, e in g.items():
        o = e.get('ORIGINAL')
        if not o: continue
        flips = sum(e[c]['forced_choice_action'] != o['forced_choice_action'] for c in CONDITIONS[1:] if c in e)
        shift = sum(abs(e[c]['p_abstain_AB_normalized'] - o['p_abstain_AB_normalized']) for c in CONDITIONS[1:] if c in e)
        ranked.append((flips, shift, sid))
    ranked.sort(reverse=True)
    # FIX: mix the most- and the least-sensitive cases (successes AND failures), not only flips.
    k = MAX_HTML_EXAMPLES
    chosen = [s for *_, s in ranked[:k // 2]] + [s for *_, s in ranked[::-1][:k - k // 2] if s not in {x for *_, x in ranked[:k // 2]}]
    parts = ['<html><head><meta charset="utf-8"><style>body{font-family:Arial,sans-serif;margin:25px;line-height:1.45}'
             '.case{border:1px solid #bbb;padding:18px;margin:22px 0}.grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px}'
             '.item{background:#f4f4f4;padding:10px;overflow-wrap:anywhere}img{max-width:100%;height:auto}'
             'pre{white-space:pre-wrap;overflow-wrap:anywhere}</style></head><body>',
             f'<h1>Fin-Abs SFT counterfactuals: {escape(tag)}</h1>',
             '<p>Model-generated outputs; not human-verified explanations. First half: most sensitive cases; '
             'second half: least sensitive cases.</p>']
    for sid in chosen:
        e = g[sid]; o = e['ORIGINAL']
        parts.append(f"<div class='case'><h2>{escape(sid)}</h2><p><b>Question:</b> {escape(o['question'])}</p>"
                     f"<p><b>Legacy action:</b> {escape(o['legacy_gold_action'])}</p><div class='grid'>")
        for c in CONDITIONS:
            rec = e.get(c)
            if not rec: continue
            parts.append(f"<div class='item'><h3>{escape(c)}</h3>")
            if c != 'QUESTION_ONLY':
                try: parts.append(f"<img src='{thumb(intervention_image(by_id[sid], c, donors))}'>")
                except Exception as ex: parts.append(f'<p>Image unavailable: {escape(str(ex))}</p>')
            p = rec.get('p_abstain_AB_normalized')
            parts.append(f"<p><b>Forced choice:</b> {escape(rec['forced_choice_action'])} &nbsp; "
                         f"<b>P(abstain|A/B):</b> {'%.4f' % p if isinstance(p, (int, float)) else 'n/a'}</p>"
                         f"<p><b>Generated action:</b> {escape(rec.get('generated_action', ''))} &nbsp; "
                         f"<b>Evidence status:</b> {escape(rec.get('evidence_status', ''))}</p>"
                         f"<p><b>Response:</b></p><pre>{escape(rec.get('response', ''))}</pre>"
                         f"<p><b>Reasoning:</b></p><pre>{escape(rec.get('decision_reasoning', ''))}</pre>"
                         f"<p><b>Justification:</b></p><pre>{escape(rec.get('justification', ''))}</pre></div>")
        parts.append('</div></div>')
    parts.append('</body></html>')
    path.write_text('\n'.join(parts), encoding='utf-8')


# ------------------------------------------------------------------ run one model
def cache_key(tag, row, condition, fp, donors):
    sig = {'tag': tag, 'adapter_fp': fp, 'prompt': PROMPT_VERSION, 'sid': row['sample_id'], 'sha': row['v2_image_sha256'],
           'cond': condition, 'q': row['question'], 'gen': GENERATE_TEXT,
           'donor': donors[row['sample_id']]['v2_image_sha256'] if condition == 'MISMATCH' else '',
           'tok': MAX_NEW_TOKENS, 'edge': MAX_IMAGE_EDGE, 'px': QWEN_MAX_PIXELS, 'blur': BLUR_RADIUS_FRACTION,
           'mask': RANDOM_MASK_FRACTION, 'grid': MASK_GRID, 'seed': SEED}
    return hashlib.sha256(json.dumps(sig, sort_keys=True).encode()).hexdigest()


def run_model(tag, model_id, ctx):
    by_id, selected, donors, pairs, test_ids = ctx['by_id'], ctx['selected'], ctx['donors'], ctx['pairs'], ctx['test_ids']
    out = OUTPUT_ROOT / tag; out.mkdir(parents=True, exist_ok=True)
    cache_file = out / 'predictions.jsonl'
    model, proc, adapter_path, fp = load_sft_model(tag, model_id, test_ids)
    cache = {}
    if cache_file.is_file():
        for ln in cache_file.read_text(encoding='utf-8').splitlines():
            try:
                x = json.loads(ln)
                if x.get('status') == 'ok': cache[x['cache_key']] = x
            except (ValueError, KeyError): pass
    print(f'{tag}: cached {len(cache)} predictions', flush=True)
    results = []
    try:
        t0 = time.time(); bs = max(1, EVAL_BATCH_SIZE)
        for cond in CONDITIONS:                               # batch within a condition (similar inputs)
            todo = []
            for sid in selected:
                key = cache_key(tag, by_id[sid], cond, fp, donors)
                if key in cache: results.append(cache[key])
                else: todo.append((sid, key))
            i = 0
            while i < len(todo):
                chunk = todo[i:i + bs]
                try:
                    recs = predict_batch(model, proc, [by_id[s] for s, _ in chunk], cond, donors)
                except Exception as exc:
                    if len(chunk) > 1:
                        print(f'  batch failed ({type(exc).__name__}); falling back to batch size 1', flush=True)
                        gc.collect(); torch.cuda.empty_cache(); bs = 1; continue
                    sid = chunk[0][0]
                    recs = [{'sample_id': sid, 'condition': cond, 'question': by_id[sid]['question'],
                             'legacy_gold_action': by_id[sid]['expected_action'], 'status': 'error', 'error': repr(exc)[:500],
                             'forced_choice_action': '', 'p_abstain_AB_normalized': None}]
                    gc.collect(); torch.cuda.empty_cache()
                with cache_file.open('a', encoding='utf-8') as f:
                    for (sid, key), rec in zip(chunk, recs):
                        rec.update(model=tag, adapter_path=adapter_path, adapter_fingerprint=fp, cache_key=key)
                        f.write(json.dumps(rec, ensure_ascii=False) + '\n'); results.append(rec)
                        if rec['status'] == 'ok': cache[key] = rec
                i += len(chunk)
                print(f'{tag} | {cond} | {i}/{len(todo)} | {time.time() - t0:.0f}s', flush=True)
        write_csv(out / 'counterfactual_per_sample.csv', results)
        orig = [r for r in results if r['condition'] == 'ORIGINAL' and r['status'] == 'ok']
        act = action_metrics(orig); act.update(model=tag, selected_test_n=len(selected),
                                               prediction_coverage=len(orig) / len(selected) if selected else 0)
        write_json(out / 'original_action_metrics.json', act)
        vis = visual_sensitivity(results); write_csv(out / 'visual_counterfactual_summary.csv', vis)
        qp = question_pairs(results, pairs); write_csv(out / 'question_counterfactual_pairs.csv', qp)
        html_report(results, by_id, donors, out / 'qualitative_examples.html', tag)
        n_err = sum(r['status'] != 'ok' for r in results)
        write_json(out / 'summary.json', {
            'model': tag, 'base_checkpoint': model_id, 'sft_adapter': adapter_path, 'sft_variant': SFT_VARIANT,
            'selected_test_records': len(selected), 'conditions': CONDITIONS,
            'expected_rows': len(selected) * len(CONDITIONS), 'successful_rows': len(results) - n_err, 'error_rows': n_err,
            'generation_complete_rows': sum(r.get('generation_format') == 'complete' for r in results),
            'action_metrics': act, 'visual_sensitivity': vis, 'evaluated_question_pairs': len(qp),
            'pair_matches_legacy_rate': (sum(r['matches_both_legacy_actions'] for r in qp) / len(qp)) if qp else None,
            'limitations': ['Legacy action annotations are not fully human-adjudicated.',
                            'Question counterfactual pairs are candidates (same image bytes), not adjudicated.',
                            'Random masking is not targeted evidence removal.',
                            'A/B-normalized probabilities are not calibrated.',
                            'Generated reasoning is not a faithful internal thought trace.']})
        print(f'\n{tag} DONE | original n={len(orig)} | Act-D agreement={act["legacy_action_accuracy"]} | '
              f'pairs={len(qp)} | errors={n_err}')
        for v in vis:
            print(f"  {v['condition']:13s} V={v['V_mean_abs_delta_p_abstain']:.4f} "
                  f"[{v['V_ci95_low']:.4f}, {v['V_ci95_high']:.4f}]  flip={v['action_flip_rate']:.3f}"
                  if v['V_ci95_low'] is not None else f"  {v['condition']} V={v['V_mean_abs_delta_p_abstain']:.4f}")
        return {'model': tag, 'original_n': len(orig), 'legacy_action_agreement': act['legacy_action_accuracy'],
                'macro_f1': act['macro_f1'], 'UAR': act['UAR'], 'OAR': act['OAR'], 'question_pairs': len(qp),
                **{v['condition'] + '_V': v['V_mean_abs_delta_p_abstain'] for v in vis},
                **{v['condition'] + '_V_ci95': f"[{v['V_ci95_low']}, {v['V_ci95_high']}]" for v in vis},
                **{v['condition'] + '_flip_rate': v['action_flip_rate'] for v in vis}}
    finally:
        del model, proc; gc.collect(); torch.cuda.empty_cache()


def merge_summary_csv(path, new_rows):
    """Keep the rows of models that are NOT re-run here (gemma3_4b, qwen25_vl_7b, qwen25_vl_3b)."""
    kept = []
    if path.is_file():
        new_models = {r['model'] for r in new_rows}
        kept = [r for r in read_csv(path) if r.get('model') not in new_models]
    order = list(MODELS)
    rows = sorted(kept + list(new_rows), key=lambda r: order.index(r['model']) if r['model'] in order else 99)
    write_csv(path, rows)
    print(f'Summary merged: {path} -> models {[r["model"] for r in rows]}')


def check_same_selection(selected, donors=None):
    """Require identical case selection and mismatch donors on repeated runs."""
    setup = OUTPUT_ROOT / 'experiment_setup.json'
    if not setup.is_file():
        return
    old = json.loads(setup.read_text(encoding='utf-8'))
    if old.get('selected_ids_sha256') != ids_sha256(selected):
        raise RuntimeError(
            f"Selected cases differ from existing experiment setup ({old.get('n')} vs {len(selected)} items). "
            'Use the same --max-questions and --seed, or choose a new --output-root.')
    if donors is not None and old.get('donors'):
        now = {k: d['sample_id'] for k, d in donors.items()}
        diff = [k for k in now if old['donors'].get(k) != now[k]]
        if diff:
            raise RuntimeError(f'MISMATCH donors differ from the earlier run for {len(diff)} items, e.g. {diff[:3]}')
    print(f"Selection matches the earlier run: {old.get('n')} test items (sha256 {old['selected_ids_sha256'][:12]}...), "
          "same MISMATCH donors.")


def main():
    global MAX_QUESTIONS, GENERATE_TEXT, EVAL_BATCH_SIZE, MODELS, ALLOW_UNVERIFIED_ADAPTER
    global ROOT, TEST_FILE, PAIR_FILE, STAGE2_ROOT, OUTPUT_ROOT, SEED, USE_4BIT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--models', nargs='+', default=['qwen25_vl_7b'], help="Model keys or 'all'")
    ap.add_argument('--root', type=Path, default=ROOT, help='Public dataset root')
    ap.add_argument('--input-csv', type=Path, default=None, help='Custom test CSV')
    ap.add_argument('--pair-file', type=Path, default=None, help='Optional candidate pairs')
    ap.add_argument('--stage2-root', type=Path, default=None, help='Stage I SFT adapter directory')
    ap.add_argument('--output-root', type=Path, default=OUTPUT_ROOT)
    ap.add_argument('--seed', type=int, default=SEED)
    ap.add_argument('--load-4bit', action=argparse.BooleanOptionalAction, default=USE_4BIT)
    ap.add_argument('--max-questions', type=int, default=None)
    ap.add_argument('--batch', type=int, default=None)
    ap.add_argument('--no-text', action='store_true', help='probabilities only (fast)')
    ap.add_argument('--gpu', default=None, help='GPU id (handled at the top of the file)')
    ap.add_argument('--allow-unverified-adapter', action='store_true')
    ap.add_argument('--describe', action='store_true', help='Print intervention protocol; no model loading')
    a = ap.parse_args()
    ROOT = a.root
    TEST_FILE = a.input_csv or (ROOT / 'finabs_step2_outputs' / 'test_v2_candidate.csv')
    PAIR_FILE = a.pair_file or (ROOT / 'same_image_opposite_action_PAIR_CANDIDATES.csv')
    STAGE2_ROOT = a.stage2_root or (ROOT / 'finabs_stage2_sft_mechanistic_v2')
    OUTPUT_ROOT = a.output_root
    SEED = a.seed
    USE_4BIT = a.load_4bit
    all_models = dict(MODELS)
    wanted = list(MODELS) if a.models == ['all'] else a.models
    if any(m not in all_models for m in wanted):
        ap.error(f'Unknown models. Choices: {list(all_models)} or all')
    MODELS = {m: all_models[m] for m in wanted}
    if a.max_questions is not None: MAX_QUESTIONS = a.max_questions
    if a.batch: EVAL_BATCH_SIZE = a.batch
    if a.no_text: GENERATE_TEXT = False
    if a.allow_unverified_adapter: ALLOW_UNVERIFIED_ADAPTER = True
    if a.describe:
        print(json.dumps({'models': list(MODELS), 'conditions': CONDITIONS,
                          'seed': SEED, 'no_gold_for_perturbations': True}, indent=2))
        return
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU not available.')
    random.seed(SEED)
    test_rows, by_id = load_test()
    pairs = find_pairs(test_rows, by_id)
    selected = select_cases(test_rows, pairs)
    donors = build_donors(test_rows, selected, by_id)
    sel_set = set(selected)
    pairs_in = [p for p in pairs if p['answer_sample_id'] in sel_set and p['abstain_sample_id'] in sel_set]
    print('=' * 75 + '\nFIN-ABS SFT COUNTERFACTUAL EXPERIMENT (FOUR BACKBONES)\n' + '=' * 75)
    check_same_selection(selected, donors)
    print('Test samples:', len(test_rows), '| selected:', len(selected),
          '| labels:', dict(Counter(by_id[s]['expected_action'] for s in selected)))
    print('Same-image answer/abstain pairs in selection:', len(pairs_in),
          '(0 means the question-counterfactual table will be empty: every test image has one question)')
    print('Distinct mismatch donors:', len({d['v2_image_sha256'] for d in donors.values()}),
          '| records per model:', len(selected) * len(CONDITIONS))
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if not (OUTPUT_ROOT / 'experiment_setup.json').is_file():      # keep the 3-model run's setup file
        write_json(OUTPUT_ROOT / 'experiment_setup.json', {'selected_ids_sha256': ids_sha256(selected), 'n': len(selected),
                   'pairs': pairs_in, 'conditions': CONDITIONS, 'donors': {s: d['sample_id'] for s, d in donors.items()}})
    ctx = {'by_id': by_id, 'selected': selected, 'donors': donors, 'pairs': pairs_in, 'test_ids': set(by_id)}
    summaries, failed = [], {}
    for tag, mid in MODELS.items():
        print('\n' + '=' * 75 + '\nSTARTING SFT COUNTERFACTUAL: ' + tag + '\n' + '=' * 75, flush=True)
        try:
            summaries.append(run_model(tag, mid, ctx))
            merge_summary_csv(OUTPUT_ROOT / 'ALL_MODELS_COUNTERFACTUAL_SUMMARY.csv', summaries)
        except Exception as exc:
            failed[tag] = f'{type(exc).__name__}: {exc}'
            print(f'!!! {tag} FAILED, continuing:\n' + traceback.format_exc(), flush=True)
        gc.collect(); torch.cuda.empty_cache()
    if failed: write_json(OUTPUT_ROOT / 'FAILED_MODELS.json', failed); print('FAILED:', failed)
    print('\nResults:', OUTPUT_ROOT, '| open each model\'s qualitative_examples.html for case studies.')


if __name__ == '__main__':
    main()


