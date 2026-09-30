import json
import math
import os
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "data.json")))
cond = data["conditions"]
Z90 = 1.6449

ORDER = ["cot", "steg_reasoning", "no_cot"]
STYLE = {
    "cot": ("CoT", "#2a78d6"),
    "steg_reasoning": ("Steg.\nreas.", "#e34948"),
    "no_cot": ("No\nCoT", "#898781"),
}


def wilson(k, n):
    p = k / n
    d = 1 + Z90 * Z90 / n
    c = (p + Z90 * Z90 / (2 * n)) / d
    h = Z90 * ((p * (1 - p) / n + Z90 * Z90 / (4 * n * n)) ** 0.5) / d
    return p, max(0, p - (c - h)), max(0, (c + h) - p)


def ztest(a, b):
    pa, na = cond[a]["n_correct"], cond[a]["n_total"]
    pb, nb = cond[b]["n_correct"], cond[b]["n_total"]
    pool = (pa + pb) / (na + nb)
    se = (pool * (1 - pool) * (1 / na + 1 / nb)) ** 0.5
    z = (pa / na - pb / nb) / se
    return math.erfc(abs(z) / 2**0.5)


def plabel(p):
    return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


fig, ax = plt.subplots(figsize=(1.16, 1.9))
xpos = {c: i for i, c in enumerate(ORDER)}
for c in ORDER:
    p, lo, hi = wilson(cond[c]["n_correct"], cond[c]["n_total"])
    ax.bar(xpos[c], p, width=0.62, color=STYLE[c][1], zorder=3)
    ax.errorbar(xpos[c], p, yerr=[[lo], [hi]], linestyle="", color="#52514e", elinewidth=0.6, capsize=1.5, zorder=4)
ax.axhline(data["chance"], color="#898781", linestyle="--", linewidth=1, zorder=2)


def bracket(a, b, y, p):
    x1, x2 = xpos[a], xpos[b]
    ax.plot([x1, x1, x2, x2], [y - 0.022, y, y, y - 0.022], color="#52514e", linewidth=0.7, zorder=5)
    ax.text((x1 + x2) / 2, y + 0.004, plabel(p), ha="center", va="bottom", fontsize=5.5, color="#52514e", zorder=5)


bracket("steg_reasoning", "no_cot", 1.04, ztest("steg_reasoning", "no_cot"))

ax.set_ylim(0, 1.14)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xlim(-0.6, 2.9)
ax.set_xticks(list(xpos.values()), [STYLE[c][0] for c in ORDER], fontsize=6)
ax.set_ylabel("Side-task accuracy", fontsize=7)
fig.text(0.02, 0.97, "b)", fontsize=8, fontweight="bold", va="top", ha="left", color="#2d3436")
ax.tick_params(colors="#52514e", labelsize=6, length=2, pad=1.5)
ax.grid(axis="y", color="#e1e0d9", linewidth=0.5)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color("#c3c2b7")
fig.subplots_adjust(left=0.3, right=0.98, top=0.95, bottom=0.2)
out = os.path.join(HERE, "prompting_uplift.png")
fig.savefig(out, dpi=300)
print("wrote", out)
