import json
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "data.json")))

CONDITIONS = [("encoded", "Encoded reasoning", "#eda100"), ("messaging", "Steganographic messaging", "#2a78d6"), ("reasoning", "Steganographic reasoning", "#e34948")]
ROWS = [
    ("mod3", 1 / 3, [("qwen", "Qwen3.6-35B-A3B"), ("nemotron_super", "Nemotron-3-Super"), ("gptoss", "gpt-oss-120B"), ("inkling_small", "Inkling-Small"), ("lightning", "Lightning-30B"), ("inkling", "Inkling")]),
    ("a5ws", 1 / 5, [("qwen", "Qwen3.6-35B-A3B"), ("nemotron_super", "Nemotron-3-Super"), ("gptoss", "gpt-oss-120B"), ("inkling_small", "Inkling-Small"), ("lightning", "Lightning-30B")]),
]

fig, axes = plt.subplots(1, 5, figsize=(10, 2.7), sharey=True)
for r, (variant, chance, models) in enumerate(ROWS[1:]):
    for c, entry in enumerate(models):
        ax = axes[c]
        model, label = entry
        for cond, _, color in CONDITIONS:
            pts = data[variant][model][cond]
            ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, linewidth=1.5, marker="o", markersize=2)
        ax.axhline(chance, color="#898781", linestyle="--", linewidth=0.8)
        if r == 0:
            ax.set_title(label, fontsize=12, color="#2d3436", pad=4)
        ax.set_ylim(0, 1.04)
        ax.xaxis.set_major_locator(MaxNLocator(3, integer=True))
        ax.grid(color="#e1e0d9", linewidth=0.5)
        ax.set_axisbelow(True)
        ax.tick_params(labelsize=9, colors="#52514e")
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#c3c2b7")

axes[0].set_ylabel("Test encoding accuracy", fontsize=11)
for c in range(5):
    axes[c].set_xlabel("Training step", fontsize=12)

handles = [Line2D([], [], color=color, marker="o", markersize=3, linewidth=1.5, label=label) for _, label, color in CONDITIONS]
handles.append(Line2D([], [], color="#898781", linestyle="--", linewidth=0.8, label="Chance"))
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=12)

fig.tight_layout(rect=(0, 0.14, 1, 1))
out = os.path.join(HERE, "sft_model_trajectories.png")
fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.06)
print("wrote", out)

# 2x3 grid version (running-sum-mod-3 only, five sweep models plus Inkling).
variant, chance, models = ROWS[0]
fig, axes = plt.subplots(2, 3, figsize=(5.2, 3.2), sharey=True)
for c, (model, label) in enumerate(models):
    ax = axes[c // 3][c % 3]
    for cond, _, color in CONDITIONS:
        pts = data[variant][model][cond]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, linewidth=1.2, marker="o", markersize=1.8)
    ax.axhline(chance, color="#898781", linestyle="--", linewidth=0.8)
    ax.set_title(label, fontsize=8, color="#2d3436", pad=3)
    ax.set_ylim(0, 1.04)
    ax.xaxis.set_major_locator(MaxNLocator(3, integer=True))
    ax.grid(color="#e1e0d9", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=7, colors="#52514e")
    ax.set_xlabel("Training step", fontsize=7.5, labelpad=1)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#c3c2b7")
axes[0][0].set_ylabel("Encoding accuracy", fontsize=7.5)
axes[1][0].set_ylabel("Encoding accuracy", fontsize=7.5)
handles = [Line2D([], [], color=color, marker="o", markersize=3, linewidth=1.2, label=label) for _, label, color in CONDITIONS]
handles.append(Line2D([], [], color="#898781", linestyle="--", linewidth=0.8, label="Chance"))
fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=7, handlelength=1.6, columnspacing=1.2)
fig.tight_layout(rect=(0, 0.06, 1, 1))
out = os.path.join(HERE, "sft_model_trajectories_mod3.png")
fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.06)
print("wrote", out)
