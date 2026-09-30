import json
import os
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "data.json")))
runs = data["runs"]

STYLE = {
    "steg_messaging": ("Steg. messaging", "#2a78d6", "o"),
    "steg_reasoning": ("Steg. reasoning", "#e34948", "s"),
    "encoded_reasoning": ("Encoded reasoning", "#eda100", "^"),
}
STEG = ["steg_messaging", "steg_reasoning"]


def xy(run, metric):
    pts = runs[run][metric]
    return [s for s, v in pts], [v for s, v in pts]


fig, ax = plt.subplots(figsize=(2.27, 2.0))

for name, (label, color, marker) in STYLE.items():
    ax.plot(*xy(name, "output_correct"), color=color, marker=marker, markersize=2, linewidth=1.1, label=label)
chance = ax.axhline(1 / data["vocab_size"], color="#898781", linestyle="--", linewidth=1, label="Chance")
ax.set_xlabel("RL training steps", fontsize=7)
ax.set_ylabel("Test output accuracy", fontsize=7)
ax.set_ylim(0, 1.3)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(color="#e1e0d9", linewidth=0.5)
ax.set_axisbelow(True)
ax.tick_params(colors="#52514e", labelsize=6, length=2, pad=1.5)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color("#c3c2b7")

ins = ax.inset_axes([0.62, 0.10, 0.34, 0.28])
for name in STYLE:
    label, color, marker = STYLE[name]
    ins.plot(*xy(name, "suspiciousness"), color=color, marker=marker, markersize=1.2, linewidth=0.8)
ins.set_title("Suspiciousness", fontsize=5.5, color="#52514e", pad=2)
ins.set_ylim(1, 6)
ins.set_yticks([1, 3, 5])
ins.set_xticks([0, 140, 280])
ins.tick_params(labelsize=5, colors="#52514e", length=1.5, pad=1)
ins.grid(color="#e1e0d9", linewidth=0.4)
ins.set_axisbelow(True)
for spine in ins.spines.values():
    spine.set_color("#c3c2b7")
    spine.set_linewidth(0.6)

ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncols=2, frameon=False, fontsize=5.5, handlelength=1.6, labelspacing=0.3, columnspacing=1.0, borderpad=0)

fig.text(0.01, 0.98, "b)", fontsize=8, fontweight="bold", va="top", ha="left", color="#2d3436")
fig.subplots_adjust(left=0.17, right=0.98, top=0.95, bottom=0.18)
out = os.path.join(HERE, "rl_scheme_comparison.png")
fig.savefig(out, dpi=300)
print("wrote", out)
