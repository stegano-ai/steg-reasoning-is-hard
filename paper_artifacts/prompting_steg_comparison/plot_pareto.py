import json
import os
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "data.json")))

MODEL_COLORS = {
    "gpt_5_5": "#2a78d6",
    "gemini_3_5_flash": "#1baf7a",
    "deepseek_v4_pro": "#eda100",
    "kimi_2_6": "#008300",
    "glm_5_1": "#4a3aa7",
    "qwen3_5_397b": "#eb6834",
    "qwen3_6_35b_a3b": "#e87ba4",
    "qwen3_6_27b": "#e34948",
}
SCHEME_MARKERS = {
    "markdown": "o",
    "sentence_interjection": "^",
    "sentence_markdown": "*",
    "sentence_newline": "s",
    "whitespace": "D",
    "capitalization": "v",
    "model_designed": "P",
}

PANEL_LABELS = [["a", "b"], ["c", "d"]]
SWEEP_TITLES = {("messaging", "a5"): "messaging, 5 symbols", ("reasoning", "a5"): "reasoning, $A_5$", ("messaging", "binary"): "messaging, 2 symbols", ("reasoning", "binary"): "reasoning, binary"}
fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.75), sharex=True, sharey=True)
for r, sweep in enumerate(["a5", "binary"]):
    chance = 1 / data["meta"][sweep]["vocab_size"]
    for c, task in enumerate(["messaging", "reasoning"]):
        ax = axes[r][c]
        pts = []
        for model, schemes in data["sweeps"][sweep].items():
            for scheme, tasks in schemes.items():
                run = tasks[task]
                n = run["n_total"]
                acc = sum(run["n_correct"]) / (10 * n)
                sus = run["sus_mean"]
                pts.append((sus, acc))
                xerr = 1.6449 * run["sus_std"] / n**0.5
                yerr = 1.6449 * run["acc_std"] / n**0.5
                ax.errorbar(sus, acc, xerr=xerr, yerr=yerr, color=MODEL_COLORS[model], marker=SCHEME_MARKERS[scheme], markersize=9, linestyle="", elinewidth=0.8, capsize=1.5, alpha=0.9, zorder=3)
        front = sorted(p for p in pts if not any(q[0] <= p[0] and q[1] >= p[1] and q != p for q in pts))
        ax.step([p[0] for p in front], [p[1] for p in front], where="post", color="#898781", linewidth=1.5, zorder=2)
        ax.axhline(chance, color="#898781", linestyle="--", linewidth=1)
        ax.set_ylim(0, 1.02)
        ax.set_title(SWEEP_TITLES[(task, sweep)], fontsize=14)
        ax.text(-0.02, 1.04, f"{PANEL_LABELS[r][c]})", transform=ax.transAxes, fontsize=16, fontweight="bold", va="bottom", ha="right", color="#2d3436")
        ax.grid(color="#e1e0d9", linewidth=0.5)
        ax.tick_params(colors="#52514e", labelsize=12)
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#c3c2b7")

model_handles = [Line2D([], [], linestyle="", marker="o", markersize=8, color=col, label=m) for m, col in MODEL_COLORS.items()]
scheme_handles = [Line2D([], [], linestyle="", marker=mk, markersize=8, color="#52514e", label=s.replace("_", " ").replace("capitalization", "capitalisation")) for s, mk in SCHEME_MARKERS.items()]
fig.legend(handles=model_handles, loc="center left", bbox_to_anchor=(0.79, 0.68), frameon=False, fontsize=10, title="Model", alignment="left")
fig.legend(handles=scheme_handles, loc="center left", bbox_to_anchor=(0.79, 0.30), frameon=False, fontsize=10, title="Scheme", alignment="left")
fig.supxlabel("Mean suspiciousness (1–10, lower is better)", fontsize=13, x=0.44)
fig.supylabel("Mean encoding accuracy", fontsize=13)
fig.tight_layout(rect=(0.01, 0.01, 0.785, 1))
out = os.path.join(HERE, "pareto.png")
fig.savefig(out, dpi=300)
print("wrote", out)
