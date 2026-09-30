import json
import os
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "data.json")))

# Colours match the per-model colours in plots/prompting_steg_comparison/pareto.png.
STYLE = {
    "outcome_35b": ("Qwen3.6-35B-A3B", "#e87ba4"),
    "outcome_397b": ("Qwen3.5-397B-A17B", "#eb6834"),
}
WINDOW = 5


def xy(run, metric):
    return [s for s, v in data[run][metric]], [v for s, v in data[run][metric]]


def smooth(ys, w=WINDOW):
    return [sum(win := ys[max(0, i - w // 2):i + w // 2 + 1]) / len(win) for i in range(len(ys))]


def draw(axis, metric, lw, raw_lw):
    for run, (label, color) in STYLE.items():
        x, y = xy(run, metric)
        axis.plot(x, y, color=color, linewidth=raw_lw, alpha=0.28, zorder=2)
        axis.plot(x, smooth(y), color=color, linewidth=lw, label=label, zorder=3)


fig, ax = plt.subplots(figsize=(7, 4.5))
draw(ax, "output_correct", lw=2.6, raw_lw=1.0)
ax.axhline(0.5, color="#898781", linestyle="--", linewidth=1, label="Random baseline")
ax.set_xlabel("RL training steps", fontsize=11)
ax.set_ylabel("Answer accuracy", fontsize=11)
ax.set_ylim(0, 1.45)  # headroom so the inset sits above the data rather than over it
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.02), frameon=False, fontsize=9.5)
ax.grid(color="#e1e0d9", linewidth=0.5)
ax.set_axisbelow(True)
ax.tick_params(colors="#52514e")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color("#c3c2b7")
ax.spines["left"].set_bounds(0, 1.0)

ins = ax.inset_axes([0.56, 0.735, 0.41, 0.20])
draw(ins, "suspiciousness", lw=1.6, raw_lw=0.8)
ins.set_title("Monitor suspiciousness", fontsize=8.5, color="#52514e", pad=3)
ins.set_ylim(1, 7.5)
ins.set_yticks([1, 4, 7])
ins.set_xticks([0, 20, 39])
ins.tick_params(labelsize=7, colors="#52514e", length=2)
ins.grid(color="#e1e0d9", linewidth=0.4)
ins.set_axisbelow(True)
for spine in ins.spines.values():
    spine.set_color("#c3c2b7")
    spine.set_linewidth(0.6)

fig.tight_layout()
out = os.path.join(HERE, "rl_knapsack_outcome.png")
fig.savefig(out, dpi=300)
print("wrote", out)
