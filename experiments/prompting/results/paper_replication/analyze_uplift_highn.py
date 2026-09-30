"""High-n uplift analysis: first_letter vs restated for gpt_5_nano and qwen3_235b on the
5000-pool cells (results/paper_replication/eval_{model}_{cond}_len{L}). Per-length and
pooled one-sided paired McNemar (pairing on expected_reasoning within a length, as before),
plus post-hoc first-letter trace decode (logic from the deleted
decode_paper_schemes.py: first letter of each sentence, N->0/S->1, better of the two
alignments; chance 0.5)."""
import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest

from lib.trace_decode import decode_first_letter, trace_correct as trace_acc

BASE = Path(__file__).parent
MODELS = ["gpt_5_nano", "qwen3_235b_a22b_2507"]
LENS = ["len3", "len5", "len8", "len12"]


def load(model, cond, ln):
    f = BASE / f"eval_{model}_{cond}_{ln}" / "completions.jsonl"
    return [json.loads(l) for l in open(f)] if f.exists() else []


def paired(a_rows, b_rows):
    bykey = defaultdict(list)
    for r in b_rows:
        bykey[tuple(r["expected_reasoning"])].append(r["output_correct"])
    return [(a["output_correct"], bykey[tuple(a["expected_reasoning"])].pop())
            for a in a_rows if bykey.get(tuple(a["expected_reasoning"]))]


for model in MODELS:
    print(f"\n=== {model} ===")
    fl_w = rs_w = n_all = 0
    for ln in LENS:
        fl, rs = load(model, "first_letter", ln), load(model, "restated", ln)
        if not fl or not rs:
            print(f"  {ln}: MISSING ({len(fl)}/{len(rs)} rows)")
            continue
        pairs = paired(fl, rs)
        w = sum(a > b for a, b in pairs)
        l = sum(a < b for a, b in pairs)
        p = binomtest(w, w + l, alternative="greater").pvalue if w + l else 1.0
        fl_w += w; rs_w += l; n_all += len(pairs)
        tr = [trace_acc(decode_first_letter(r["cot"] if r["cot"] is not None else r["completion"]),
                        r["expected_reasoning"]) for r in fl]
        fl_acc = sum(r["output_correct"] for r in fl) / len(fl)
        rs_acc = sum(r["output_correct"] for r in rs) / len(rs)
        print(f"  {ln:6s} restated {rs_acc:.3f} (n={len(rs)})  first_letter {fl_acc:.3f} (n={len(fl)})  "
              f"delta {fl_acc - rs_acc:+.3f}  p={p:.4f}  trace {sum(tr)/len(tr):.3f}")
    if n_all:
        p = binomtest(fl_w, fl_w + rs_w, alternative="greater").pvalue if fl_w + rs_w else 1.0
        print(f"  POOLED {n_all} pairs: fl_wins {fl_w}, rs_wins {rs_w}, one-sided p={p:.5f}")
