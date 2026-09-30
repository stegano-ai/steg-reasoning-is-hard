import json
import os
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "data.json")))

SUS_THRESHOLD = 3
COLORS = {"messaging": "#2a78d6", "reasoning": "#e34948"}
TITLES = {("messaging", "a5"): "Messaging, 5 symbols", ("reasoning", "a5"): "Reasoning, $A_5$",
          ("messaging", "binary"): "Messaging, 2 symbols", ("reasoning", "binary"): "Reasoning, binary"}
PANEL_LABELS = ["a", "b"]

fig, axes = plt.subplots(1, 2, figsize=(2.45, 1.55), sharex=True, sharey=True)
for r, sweep in enumerate(["a5"]):
    chance = 1 / data["meta"][sweep]["vocab_size"]
    for c, task in enumerate(["messaging", "reasoning"]):
        ax = axes[c]
        pts = []
        for schemes in data["sweeps"][sweep].values():
            for tasks in schemes.values():
                run = tasks[task]
                pts.append((run["sus_mean"], sum(run["n_correct"]) / (10 * run["n_total"])))
        front = sorted(p for p in pts if not any(q[0] <= p[0] and q[1] >= p[1] and q != p for q in pts))
        ax.axvspan(0.5, SUS_THRESHOLD, color="#f0efe9", zorder=0, lw=0)
        ax.scatter([p[0] for p in pts], [p[1] for p in pts], s=8, color="#898781", alpha=0.55, linewidths=0, zorder=2)
        ax.step([p[0] for p in front] + [10.5], [p[1] for p in front] + [front[-1][1]], where="post", color=COLORS[task], linewidth=1.4, zorder=3)
        ax.scatter([p[0] for p in front], [p[1] for p in front], s=12, color=COLORS[task], zorder=4, linewidths=0)
        ax.axhline(chance, color="#898781", linestyle="--", linewidth=1, zorder=1)
        ax.set_xlim(0.5, 10.5)
        ax.set_ylim(0, 1.02)
        ax.set_title(TITLES[(task, sweep)], fontsize=8, color="#2d3436", pad=3)
        ax.text(0.04, 0.97, f"{PANEL_LABELS[c]})", transform=ax.transAxes, fontsize=8, fontweight="bold", va="top", ha="left", color="#2d3436")
        ax.grid(color="#e1e0d9", linewidth=0.5, zorder=0)
        ax.set_xticks([2, 4, 6, 8, 10])
        ax.set_yticks([0, 0.5, 1])
        ax.tick_params(colors="#52514e", labelsize=7, length=2, pad=1.5)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#c3c2b7")
        if r == 0 and c == 0:
            ax.text(SUS_THRESHOLD - 0.15, 0.04, "covert", ha="right", va="bottom", fontsize=6.5, color="#52514e", style="italic")
            ax.text(10.3, chance + 0.02, "chance", ha="right", va="bottom", fontsize=6.5, color="#52514e", style="italic")
fig.supxlabel("Mean suspiciousness (1–10, lower is better)", fontsize=8, y=0.02)
fig.supylabel("Encoding accuracy", fontsize=8, x=0.02)
fig.subplots_adjust(left=0.15, right=0.98, top=0.85, bottom=0.27, wspace=0.24)
out = os.path.join(HERE, "pareto_simple.png")
fig.savefig(out, dpi=300)
print("wrote", out)
