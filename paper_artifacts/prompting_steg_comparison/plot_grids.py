import json
import os
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "data.json")))

COLORS = {"messaging": "#2a78d6", "reasoning": "#e34948"}
MARKERS = {"messaging": "o", "reasoning": "s"}
POSITIONS = list(range(1, 11))
Z90 = 1.6449


def wilson(k, n):
    p = k / n
    denom = 1 + Z90 * Z90 / n
    center = (p + Z90 * Z90 / (2 * n)) / denom
    half = Z90 * ((p * (1 - p) / n + Z90 * Z90 / (4 * n * n)) ** 0.5) / denom
    return center - half, center + half

for sweep, models in data["sweeps"].items():
    chance = 1 / data["meta"][sweep]["vocab_size"]
    model_names = list(models)
    schemes = list(models[model_names[0]])
    fig, axes = plt.subplots(len(model_names), len(schemes), figsize=(1.5 * len(schemes), 1.4 * len(model_names)), sharex=True, sharey=True, squeeze=False)
    for r, model in enumerate(model_names):
        for c, scheme in enumerate(schemes):
            ax = axes[r][c]
            if scheme not in models[model]:
                ax.text(5.5, 0.5, "no data", ha="center", va="center", fontsize=11, color="#898781")
            for task in ["messaging", "reasoning"] if scheme in models[model] else []:
                run = models[model][scheme][task]
                n = run["n_total_covert"]
                if n == 0:
                    continue
                acc = [k / n for k in run["n_correct_covert"]]
                bounds = [wilson(k, n) for k in run["n_correct_covert"]]
                yerr = [[max(0, a - lo) for a, (lo, hi) in zip(acc, bounds)], [max(0, hi - a) for a, (lo, hi) in zip(acc, bounds)]]
                ax.errorbar(POSITIONS, acc, yerr=yerr, color=COLORS[task], marker=MARKERS[task], markersize=3.5, linewidth=1.5, elinewidth=0.8, capsize=1.5, alpha=0.9, label=task.capitalize())
            ax.axhline(chance, color="#898781", linestyle="--", linewidth=1, label="Chance")
            ax.set_ylim(0, 1.02)
            ax.set_xticks([1, 5, 10])
            ax.grid(color="#e1e0d9", linewidth=0.5)
            ax.tick_params(labelsize=12, colors="#52514e")
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)
            for spine in ["left", "bottom"]:
                ax.spines[spine].set_color("#c3c2b7")
            if r == 0:
                ax.set_title(scheme.replace("_", "\n").replace("capitalization", "capitalisation"), fontsize=12)
            if c == 0:
                ax.set_ylabel(model, fontsize=12, rotation=45, ha="right", va="center", labelpad=6)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncols=3, frameon=False, fontsize=14)
    fig.supxlabel("Position in sequence", fontsize=15)
    fig.supylabel("Fraction correct", fontsize=15)
    fig.tight_layout(rect=(0.01, 0.01, 1, 0.95))
    out = os.path.join(HERE, f"{sweep}.png")
    fig.savefig(out, dpi=300)
    print("wrote", out)
