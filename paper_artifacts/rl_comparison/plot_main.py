import json
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "data.json")))

STYLE = {
    "messaging": ("Steganographic messaging", "#2a78d6", "o"),
    "reasoning": ("Steganographic reasoning", "#e34948", "s"),
    "wordprefix": ("Encoded reasoning", "#eda100", "^"),
}


def series(run, key):
    pts = data[run][key]
    return [s for s, v in pts], [v for s, v in pts]


def style_axes(ax, labelsize=5.5):
    ax.grid(color="#e1e0d9", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#52514e", labelsize=labelsize, length=1.5, pad=1)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#c3c2b7")


# suspiciousness panel (appendix)
fig, ax = plt.subplots(figsize=(3.4, 2.0))
for run, (label, color, marker) in STYLE.items():
    if "test/env/all/suspiciousness" not in data[run]:
        continue  # wordprefix control ran without a monitor
    ax.plot(*series(run, "test/env/all/suspiciousness"), color=color, marker=marker, markersize=2, linewidth=1, label=label)
ax.set_xlabel("RL training steps", fontsize=7.5)
ax.set_ylabel("Mean suspiciousness (1–10)", fontsize=7.5)
ax.set_ylim(0.7, 7.6)
style_axes(ax, labelsize=6.5)
ax.legend(frameon=False, fontsize=6.5, handlelength=1.6)
fig.subplots_adjust(left=0.14, right=0.98, top=0.95, bottom=0.18)
out = os.path.join(HERE, "rl_suspiciousness.png")
fig.savefig(out, dpi=300)
print("wrote", out)

# Side-by-side two-panel version for the single-column layout: (a) given scheme on AQuA-RAT,
# (b) knapsack; legend at the bottom. Sized to render at 0.48\textwidth (2.64in).
knap = json.load(open(os.path.join(HERE, "..", "rl_knapsack", "data.json")))["process_35b"]
fig, axes = plt.subplots(1, 2, figsize=(2.45, 1.75), sharey=True)
ax = axes[0]
for run, (label, color, marker) in STYLE.items():
    ax.plot(*series(run, "test/env/all/reasoning_correct"), color=color, marker=marker, markersize=1.8, linewidth=1)
ax.axhline(0.5, color="#898781", linestyle="--", linewidth=0.9)
ax.set_title("a) AQuA-RAT", fontsize=8, color="#2d3436", pad=3, loc="left")
style_axes(ax, labelsize=7)
ax = axes[1]
pts = knap["reasoning_correct"]
ax.plot([p[0] for p in pts], [p[1] for p in pts], color="#e34948", marker="s", markersize=1.8, linewidth=1)
ax.axhline(0.5, color="#898781", linestyle="--", linewidth=0.9)
ax.set_title("b) Knapsack", fontsize=8, color="#2d3436", pad=3, loc="left")
style_axes(ax, labelsize=7)
for ax in axes:
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0, 0.5, 1])
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
handles = [Line2D([], [], color=c, marker=m, markersize=2.5, linewidth=1, label=l) for l, c, m in STYLE.values()]
handles.append(Line2D([], [], color="#898781", linestyle="--", linewidth=0.9, label="Chance"))
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=6.5, handlelength=1.0, columnspacing=0.4, handletextpad=0.25)
fig.supxlabel("RL training steps", fontsize=8, y=0.21)
fig.supylabel("Test encoding accuracy", fontsize=8, x=0.02, y=0.6)
fig.subplots_adjust(left=0.15, right=0.98, top=0.88, bottom=0.42, wspace=0.16)
out = os.path.join(HERE, "rl_main_stacked.png")
fig.savefig(out, dpi=300)
print("wrote", out)
