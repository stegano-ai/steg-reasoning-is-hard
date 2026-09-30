"""Three-way qwen len12 comparison: latent (answer-then-state, no steg framing) vs
restated (leak-content control) vs first_letter (the uplift arm). All three cells share
the same seed-0 5000-pool -> paired sample-by-sample on expected_reasoning. Also reports
answer length per arm and within-arm length->correctness (the pacing hypothesis predicts
longer answers help)."""
import json
from collections import defaultdict
from itertools import combinations
from pathlib import Path

from scipy.stats import binomtest, pointbiserialr

BASE = Path(__file__).parent
ARMS = ["restated", "latent", "first_letter"]


def load(cond):
    f = BASE / f"eval_qwen3_235b_a22b_2507_{cond}_len12" / "completions.jsonl"
    return [json.loads(l) for l in open(f)]


rows = {a: load(a) for a in ARMS}

print(f"{'arm':14s} {'n':>5s} {'acc':>6s} {'answer chars':>13s}")
for a in ARMS:
    r = rows[a]
    acc = sum(x["output_correct"] for x in r) / len(r)
    chars = [len(x["completion"].rsplit("####", 1)[0]) for x in r]
    print(f"{a:14s} {len(r):5d} {acc:6.3f} {sum(chars)/len(chars):13.0f}")

print("\npairwise one-sided paired McNemar (rows arm > cols arm):")
for a, b in combinations(ARMS, 2):
    bykey = defaultdict(list)
    for r in rows[b]:
        bykey[tuple(r["expected_reasoning"])].append(r["output_correct"])
    w = l = n = 0
    for r in rows[a]:
        k = tuple(r["expected_reasoning"])
        if bykey.get(k):
            v = bykey[k].pop()
            w += r["output_correct"] > v
            l += r["output_correct"] < v
            n += 1
    p_a = binomtest(w, w + l, alternative="greater").pvalue if w + l else 1.0
    p_b = binomtest(l, w + l, alternative="greater").pvalue if w + l else 1.0
    print(f"  {a} vs {b}: pairs={n} {a}_wins={w} {b}_wins={l}  p({a}>{b})={p_a:.4f}  p({b}>{a})={p_b:.4f}")

print("\nwithin-arm answer-length -> correctness (point-biserial r):")
for a in ARMS:
    r = rows[a]
    chars = [len(x["completion"].rsplit("####", 1)[0]) for x in r]
    corr = [x["output_correct"] for x in r]
    pb = pointbiserialr(corr, chars)
    print(f"  {a:14s} r={pb.statistic:+.3f} p={pb.pvalue:.4f}")
