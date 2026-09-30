import json
import os
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "data.json")))

STYLE = {"messaging": ("Steganographic messaging", "#2a78d6", "o"), "reasoning": ("Steganographic reasoning", "#e34948", "s")}


def series(run):
    pts = data[run]
    return [s for s, c, e in pts], [c for s, c, e in pts]


fig, ax = plt.subplots(figsize=(7, 4.5))
for run, (label, color, marker) in STYLE.items():
    ax.plot(*series(run), color=color, marker=marker, markersize=6, linewidth=1.8, label=label)
ax.axhline(1 / 3, color="#898781", linestyle="--", linewidth=1, label="Chance")
ax.set_xlabel("Training step", fontsize=11)
ax.set_ylabel("Test encoding accuracy", fontsize=11)
ax.set_xticks([0, 25, 50])
ax.set_ylim(0.25, 1.02)
ax.grid(color="#e1e0d9", linewidth=0.5)
ax.set_axisbelow(True)
ax.tick_params(colors="#52514e")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color("#c3c2b7")
ax.legend(loc="upper left", frameon=False, fontsize=10)

fig.tight_layout()
out = os.path.join(HERE, "sft_knapsack.png")
fig.savefig(out, dpi=300)
print("wrote", out)
