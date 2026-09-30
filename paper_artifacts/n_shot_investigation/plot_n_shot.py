import json
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "data.json")))
shots = data["shots"]
Z90 = 1.6449

STYLE = {
    "steg_messaging": ("Messaging", "#2a78d6", "o"),
    "encoded_reasoning": ("Encoded reasoning", "#eda100", "^"),
    "steg_reasoning": ("Steg. reasoning", "#e34948", "s"),
}
# each line's baseline: messaging -> chance; reasoning lines -> shared no-CoT floor
BASELINE = {"steg_messaging": data["chance"], "encoded_reasoning": data["no_cot_reasoning"], "steg_reasoning": data["no_cot_reasoning"]}
DASH = {"steg_messaging": (0, (1.5, 3)), "encoded_reasoning": (0, (1.5, 3)), "steg_reasoning": (2.5, (1.5, 3))}


def wilson(k, n):
    p = k / n
    d = 1 + Z90 * Z90 / n
    c = (p + Z90 * Z90 / (2 * n)) / d
    h = Z90 * ((p * (1 - p) / n + Z90 * Z90 / (4 * n * n)) ** 0.5) / d
    return p, max(0, p - (c - h)), max(0, (c + h) - p)


fig, ax = plt.subplots(figsize=(2.09, 1.9))
fig.subplots_adjust(left=0.17, right=0.98, top=0.95, bottom=0.2)
for key, (label, color, marker) in STYLE.items():
    line = data["lines"][key]
    if line["metric"] == "encoding":
        ys = [s["mean"] for s in line["per_shot"]]
        yerr = [s["ci"] for s in line["per_shot"]]
    else:
        stats = [wilson(s["n_correct"], s["n_total"]) for s in line["per_shot"]]
        ys = [s[0] for s in stats]
        yerr = [[s[1] for s in stats], [s[2] for s in stats]]
    ax.errorbar(shots, ys, yerr=yerr, color=color, marker=marker, markersize=2.5, linewidth=1.1, elinewidth=0.6, capsize=1.5, zorder=3, label=label)
    ax.axhline(BASELINE[key], color=color, linestyle=DASH[key], linewidth=0.9, zorder=1)

ax.set_xlabel("Few-shot examples", fontsize=7)
ax.set_ylabel("Accuracy", fontsize=7)
fig.text(0.01, 0.97, "a)", fontsize=8, fontweight="bold", va="top", ha="left", color="#2d3436")
ax.set_xticks(shots)
ax.set_xlim(-0.2, 5.9)
ax.set_ylim(0, 1.2)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(color="#e1e0d9", linewidth=0.5)
ax.set_axisbelow(True)
ax.tick_params(colors="#52514e", labelsize=6, length=2, pad=1.5)
for sp in ["top", "right"]:
    ax.spines[sp].set_visible(False)
for sp in ["left", "bottom"]:
    ax.spines[sp].set_color("#c3c2b7")
handles, labels = ax.get_legend_handles_labels()
handles.append(Line2D([], [], color="#898781", linestyle=":", linewidth=1))
labels.append("Baseline")
ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.55, 1.02), ncols=2, frameon=False, fontsize=5, handlelength=1.3, labelspacing=0.3, columnspacing=0.6, borderpad=0)
out = os.path.join(HERE, "n_shot.png")
fig.savefig(out, dpi=300)
print("wrote", out)
