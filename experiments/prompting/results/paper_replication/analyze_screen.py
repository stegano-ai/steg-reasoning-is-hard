"""Four-arm paired screen for the two survivor models on the unified 5000-pool:
no_encoding / restated / first_letter / word_count (all cells owned by uplift.yaml).
Per length and pooled: arm accuracies, then the key one-sided paired McNemar contrasts —
leak (restated>no_encoding), uplift vs baseline (first_letter>no_encoding), uplift vs
matched control (first_letter>restated), and word_count vs both."""
import json
from collections import defaultdict
from pathlib import Path

from scipy.stats import binomtest

BASE = Path(__file__).parent
MODELS = ["gpt_5_nano", "qwen3_235b_a22b_2507"]
ARMS = ["no_encoding", "restated", "first_letter", "word_count"]
LENS = ["len3", "len5", "len8", "len12"]
CONTRASTS = [("restated", "no_encoding"), ("first_letter", "no_encoding"),
             ("first_letter", "restated"), ("word_count", "no_encoding"),
             ("word_count", "restated")]


def load(model, arm, ln):
    f = BASE / f"eval_{model}_{arm}_{ln}" / "completions.jsonl"
    return [json.loads(l) for l in open(f)] if f.exists() else []


def mcnemar(a_rows, b_rows):
    """one-sided p for a > b, paired on expected_reasoning; returns (p, wins_a, wins_b, n)."""
    bykey = defaultdict(list)
    for r in b_rows:
        bykey[tuple(r["expected_reasoning"])].append(r["output_correct"])
    w = l = n = 0
    for r in a_rows:
        k = tuple(r["expected_reasoning"])
        if bykey.get(k):
            v = bykey[k].pop()
            w += r["output_correct"] > v
            l += r["output_correct"] < v
            n += 1
    p = binomtest(w, w + l, alternative="greater").pvalue if w + l else 1.0
    return p, w, l, n


for model in MODELS:
    rows = {(a, ln): load(model, a, ln) for a in ARMS for ln in LENS}
    print(f"\n=== {model} ===")
    hdr = f"{'len':6s}" + "".join(f" {a:>13s}" for a in ARMS)
    print(hdr)
    for ln in LENS + ["ALL"]:
        line = f"{ln:6s}"
        for a in ARMS:
            r = [x for l2 in (LENS if ln == 'ALL' else [ln]) for x in rows[(a, l2)]]
            acc = sum(x["output_correct"] for x in r) / len(r) if r else float("nan")
            line += f" {acc:8.3f}({len(r):4d})"
        print(line)
    print("contrasts (one-sided paired McNemar, pooled over lengths):")
    for a, b in CONTRASTS:
        ra = [x for ln in LENS for x in rows[(a, ln)]]
        rb = [x for ln in LENS for x in rows[(b, ln)]]
        p, w, l, n = mcnemar(ra, rb)
        aa = sum(x["output_correct"] for x in ra) / len(ra)
        ab = sum(x["output_correct"] for x in rb) / len(rb)
        print(f"  {a:>12s} > {b:12s} delta {aa - ab:+.3f}  pairs {n:5d} ({w}:{l})  p={p:.5f}")
