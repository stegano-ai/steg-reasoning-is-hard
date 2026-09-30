import json
import math
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "data.json")))
SEQ = data["seq_len"]
CHANCE = data["chance"]
Z90 = 1.6449
POS = list(range(1, SEQ + 1))

LINE_STYLE = {"messaging": ("Messaging", "#2a78d6", "o"), "steg": ("Steganographic reasoning", "#e34948", "s")}
BAR_STYLE = {"cot": ("CoT", "#2a78d6"), "steg": ("Steg", "#e34948"), "direct": ("No CoT", "#898781")}
BAR_ORDER = ["cot", "steg", "direct"]
MODEL_LABEL = {m: m.replace("_", " ") for m in data["models"]}


def wilson(k, n):
    p = k / n
    d = 1 + Z90 * Z90 / n
    c = (p + Z90 * Z90 / (2 * n)) / d
    h = Z90 * ((p * (1 - p) / n + Z90 * Z90 / (4 * n * n)) ** 0.5) / d
    return p, max(0, c - h), min(1, c + h)


def ztest(k1, n1, k2, n2):
    pool = (k1 + k2) / (n1 + n2)
    se = (pool * (1 - pool) * (1 / n1 + 1 / n2)) ** 0.5
    if se == 0:
        return 1.0
    return math.erfc(abs(k1 / n1 - k2 / n2) / se / 2**0.5)


models = list(data["models"])
fig, axes = plt.subplots(len(models), 2, figsize=(5.2, 1.25 * len(models)), gridspec_kw={"width_ratios": [1.7, 1]}, squeeze=False)

for r, model in enumerate(models):
    md = data["models"][model]
    axL, axR = axes[r][0], axes[r][1]

    # left: per-position encoding accuracy — markers + whiskers + line (a5 style).
    # small x-dodge so overlapping lines (e.g. gpt-5.5, both pinned at 1.0) stay both visible.
    for task, dodge, (label, color, marker) in [("messaging", -0.18, LINE_STYLE["messaging"]), ("steg", 0.18, LINE_STYLE["steg"])]:
        run = md["lines"][task]
        n = run["n_total"]
        stats = [wilson(k, n) for k in run["n_correct"]]
        acc = [s[0] for s in stats]
        yerr = [[max(0, s[0] - s[1]) for s in stats], [max(0, s[2] - s[0]) for s in stats]]
        axL.errorbar([p + dodge for p in POS], acc, yerr=yerr, color=color, marker=marker, markersize=2.4, linewidth=1.2, elinewidth=0.6, capsize=1.2, alpha=0.9)
    axL.axhline(CHANCE, color="#898781", linestyle="--", linewidth=0.9, zorder=1)
    axL.set_ylim(0, 1.03)
    axL.set_xlim(0.4, SEQ + 0.6)
    axL.set_xticks([1, 12, 24])
    axL.set_ylabel(MODEL_LABEL[model], fontsize=9, fontweight="bold")
    axL.grid(color="#e1e0d9", linewidth=0.4)

    # right: final-answer accuracy bars — bracket only if steg significantly BEATS no-CoT
    tops = {}
    for j, c in enumerate(BAR_ORDER):
        p, lo, hi = wilson(md["bars"][c]["n_correct"], md["bars"][c]["n_total"])
        axR.bar(j, p, width=0.66, color=BAR_STYLE[c][1], zorder=3)
        axR.errorbar(j, p, yerr=[[max(0, p - lo)], [max(0, hi - p)]], linestyle="", color="#52514e", elinewidth=0.8, capsize=2, zorder=4)
        tops[c] = hi
    axR.axhline(CHANCE, color="#898781", linestyle="--", linewidth=0.9, zorder=1)
    sk, sn = md["bars"]["steg"]["n_correct"], md["bars"]["steg"]["n_total"]
    dk, dn = md["bars"]["direct"]["n_correct"], md["bars"]["direct"]["n_total"]
    pv = ztest(sk, sn, dk, dn)
    if sk / sn > dk / dn and pv < 0.05:
        y = max(tops["steg"], tops["direct"]) + 0.08
        axR.plot([1, 1, 2, 2], [y - 0.04, y, y, y - 0.04], color="#52514e", linewidth=0.9, zorder=5)
        axR.text(1.5, y + 0.01, "p < 0.001" if pv < 0.001 else f"p = {pv:.3f}", ha="center", va="bottom", fontsize=8, color="#52514e")
    axR.set_ylim(0, 1.22)
    axR.set_yticks([0, 0.5, 1.0])
    axR.set_xticks(range(len(BAR_ORDER)))
    axR.set_xticklabels([BAR_STYLE[c][0] for c in BAR_ORDER] if r == len(models) - 1 else [])
    axR.grid(axis="y", color="#e1e0d9", linewidth=0.4)

    for ax in (axL, axR):
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=7.5, colors="#52514e")
        for sp in ["top", "right"]:
            ax.spines[sp].set_visible(False)
        for sp in ["left", "bottom"]:
            ax.spines[sp].set_color("#c3c2b7")
    if r != len(models) - 1:
        axL.set_xticklabels([])

axes[0][0].set_title("Per-position\nencoding accuracy", fontsize=9)
axes[0][1].set_title("Final-answer\naccuracy", fontsize=9)
axes[-1][0].set_xlabel("Position in sequence", fontsize=8.5)

handles = [Line2D([], [], color=c, marker=m, markersize=4, linewidth=1.6, label=l) for l, c, m in LINE_STYLE.values()] + [Line2D([], [], color="#898781", linestyle="--", linewidth=1, label="Chance")]
fig.legend(handles=handles, loc="upper center", ncols=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.008))
fig.tight_layout(rect=(0, 0, 1, 0.97))
out = os.path.join(HERE, "knapsack_sweep.png")
fig.savefig(out, dpi=300)
print("wrote", out)
