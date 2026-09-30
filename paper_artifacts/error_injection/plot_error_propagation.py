import json
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Patch, Circle
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "data.json")))
SEQ = data["seq_length"]

# One panel per steganographic-reasoning organism: the two knapsack RL runs, then the
# SFT runs by task (running sum mod 3, then A5 with the whitespace scheme), sweep
# models in the column order of the sweep figure.
PANELS = [
    ("process_35b", "RL process reward, knapsack (Qwen3.6-35B)"),
    ("onlyoutput_397b", "RL outcome reward, knapsack (Qwen3.5-397B)"),
    ("sft_summod3", "SFT sum-mod-3 (Qwen3.6-35B)"),
    ("sft_nemotron_super", "SFT sum-mod-3 (Nemotron-3-Super-120B)"),
    ("sft_inkling_small", "SFT sum-mod-3 (Inkling-Small)"),
    ("sft_a5ws_nemotron_super", "SFT $A_5$ whitespace (Nemotron-3-Super-120B)"),
    ("sft_a5ws_gptoss", "SFT $A_5$ whitespace (gpt-oss-120B)"),
    ("sft_a5ws_inkling_small", "SFT $A_5$ whitespace (Inkling-Small)"),
]
C_ERR, C_CORR, C_OTHER = "#e34948", "#2a78d6", "#c3c2b7"
RADIUS = 0.42
ROW_GAP = 0.5  # extra space between the normal row and the injection rows


def draw_pie(ax, x, y, counts, alpha=1.0, ring=False):
    n = sum(counts.values())
    theta = 90.0
    for key, color in [("n_err", C_ERR), ("n_corr", C_CORR), ("n_other", C_OTHER), ("n_missing", C_OTHER)]:
        span = 360.0 * counts[key] / n
        if span:
            ax.add_patch(Wedge((x, y), RADIUS, theta - span, theta, facecolor=color, alpha=alpha, linewidth=0))
            theta -= span
    if ring:
        ax.add_patch(Circle((x, y), RADIUS, facecolor="none", edgecolor="#1a1a19", linewidth=1.8))


fig, axes = plt.subplots(2, 4, figsize=(16.4, 9.4), sharex=True, sharey=True)
for ax, (model, title) in zip(axes.flat, PANELS):
    m = data["models"][model]
    for i, counts in enumerate(m["normal"]):
        draw_pie(ax, i, 0, counts)
    for p, row in enumerate(m["errinj"]):
        y = 1 + ROW_GAP + p
        for i, counts in enumerate(row):
            draw_pie(ax, i, y, counts, alpha=0.35 if i < p else 1.0, ring=(i == p))
    ax.axhline(0.5 + ROW_GAP / 2, color="#c3c2b7", linewidth=0.8)
    ax.set_title(title.replace(" (", "\n("), fontsize=16, color="#52514e", pad=6)
    ax.set_xlim(-0.6, SEQ - 0.4)
    ax.set_ylim(SEQ + ROW_GAP + 0.6, -0.6)
    ax.set_aspect("equal")
    ax.set_xticks(range(SEQ))
    ax.set_yticks([0] + [1 + ROW_GAP + p for p in range(SEQ)])
    ax.set_yticklabels(["none"] + [str(p) for p in range(SEQ)], fontsize=19)
    ax.tick_params(colors="#52514e", labelsize=19, length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

for ax in axes[1]:
    ax.set_xlabel("Reasoning step", fontsize=22)
for ax in axes[:, 0]:
    ax.set_ylabel("Injected-error position", fontsize=22)

handles = [
    Patch(facecolor=C_CORR, label="correct value"),
    Patch(facecolor=C_ERR, label="consistent with injected error"),
    Patch(facecolor=C_OTHER, label="other / undecodable"),
    Line2D([], [], marker="o", markersize=18, markerfacecolor="#e34948", markeredgecolor="#1a1a19",
           markeredgewidth=1.6, linestyle="none", label="forced error (prefilled)"),
    Patch(facecolor=C_CORR, alpha=0.35, label="prefilled prefix"),
]
fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=15, columnspacing=1.2, bbox_to_anchor=(0.5, 0.0))

fig.tight_layout(rect=(0, 0.05, 1, 1))
out = os.path.join(HERE, "error_injection.png")
fig.savefig(out, dpi=300)
print("wrote", out)
