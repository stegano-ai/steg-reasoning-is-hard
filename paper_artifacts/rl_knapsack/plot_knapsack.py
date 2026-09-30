import json
import os
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "data.json")))["process_35b"]


def xy(metric):
    return [s for s, v in data[metric]], [v for s, v in data[metric]]


def style_axes(ax, labelsize=5.5):
    ax.grid(color="#e1e0d9", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#52514e", labelsize=labelsize, length=1.5, pad=1)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#c3c2b7")


fig, ax = plt.subplots(figsize=(3.4, 2.0))
ax.plot(*xy("suspiciousness"), color="#e34948", marker="o", markersize=2, linewidth=1, label="Steganographic reasoning")
ax.set_xlabel("RL training steps", fontsize=7.5)
ax.set_ylabel("Mean suspiciousness (1–10)", fontsize=7.5)
ax.set_ylim(1, 6)
style_axes(ax, labelsize=6.5)
ax.legend(frameon=False, fontsize=6.5, handlelength=1.6)
fig.subplots_adjust(left=0.14, right=0.98, top=0.95, bottom=0.18)
out = os.path.join(HERE, "rl_knapsack_suspiciousness.png")
fig.savefig(out, dpi=300)
print("wrote", out)
