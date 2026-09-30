import json
import math
import os
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "data.json")))

CONDITIONS = {
    "no_cot": ("No CoT", "#898781"),
    "evocative_word_prefix": ("Encoded reasoning", "#eda100"),
    "steg": ("Steg. reasoning (best scheme)", "#e34948"),
    "cot": ("CoT", "#2a78d6"),
}
Z90 = 1.6449
NAMES = {"gpt_5_5": "GPT-5.5", "gemini_3_5_flash": "Gemini 3.5 Flash", "deepseek_v4_pro": "DeepSeek V4 Pro", "kimi_2_6": "Kimi K2.6",
         "glm_5_1": "GLM 5.1", "qwen3_5_397b": "Qwen3.5-397B", "qwen3_6_35b_a3b": "Qwen3.6-35B", "qwen3_6_27b": "Qwen3.6-27B"}


def wilson(k, n):
    p = k / n
    denom = 1 + Z90 * Z90 / n
    center = (p + Z90 * Z90 / (2 * n)) / denom
    half = Z90 * ((p * (1 - p) / n + Z90 * Z90 / (4 * n * n)) ** 0.5) / denom
    return max(0, p - (center - half)), max(0, (center + half) - p)


def ztest_pvalue(run1, run2):
    p_pool = (run1["n_correct"] + run2["n_correct"]) / (run1["n_total"] + run2["n_total"])
    se = (p_pool * (1 - p_pool) * (1 / run1["n_total"] + 1 / run2["n_total"])) ** 0.5
    if se == 0:
        return 1.0
    z = (run1["n_correct"] / run1["n_total"] - run2["n_correct"] / run2["n_total"]) / se
    return math.erfc(abs(z) / 2**0.5)


conds = list(CONDITIONS)
width = 0.8 / len(conds)
for sweep in ["a5", "binary"]:
    fig, ax = plt.subplots(figsize=(5.2, 1.9))
    models = list(data["sweeps"][sweep])
    chance = 1 / data["meta"][sweep]["vocab_size"]
    for i, model in enumerate(models):
        entry = data["sweeps"][sweep][model]
        runs = {c: max(entry["steg"].values(), key=lambda r: r["n_correct"] / r["n_total"]) if c == "steg" else entry[c] for c in conds}
        xs = {c: i + (j - (len(conds) - 1) / 2) * width for j, c in enumerate(conds)}
        tops = {}
        for c in conds:
            p = runs[c]["n_correct"] / runs[c]["n_total"]
            lo, hi = wilson(runs[c]["n_correct"], runs[c]["n_total"])
            ax.bar(xs[c], p, width=width * 0.9, color=CONDITIONS[c][1], label=CONDITIONS[c][0] if i == 0 else None, zorder=3)
            ax.errorbar(xs[c], p, yerr=[[lo], [hi]], linestyle="", color="#52514e", elinewidth=0.8, capsize=1.5, zorder=4)
            tops[c] = p + hi
        level = 0
        for c in ["evocative_word_prefix", "steg"]:
            pval = ztest_pvalue(runs[c], runs["no_cot"])
            higher = runs[c]["n_correct"] / runs[c]["n_total"] > runs["no_cot"]["n_correct"] / runs["no_cot"]["n_total"]
            if pval < 0.05 and higher:
                y = max(tops[c], tops["steg"], tops["no_cot"]) + 0.05 + level * 0.14
                ax.plot([xs[c], xs[c], xs["no_cot"], xs["no_cot"]], [y - 0.015, y, y, y - 0.015], color=CONDITIONS[c][1], linewidth=1.0, zorder=5)
                label = "p < 0.001" if pval < 0.001 else f"p = {pval:.3f}"
                ax.text((xs[c] + xs["no_cot"]) / 2, y + 0.005, label, ha="center", va="bottom", fontsize=7, color=CONDITIONS[c][1], zorder=5)
                level += 1
    ax.axhline(chance, color="#898781", linestyle="--", linewidth=1)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_ylabel("Side-task accuracy", fontsize=7.5)
    ax.set_xticks(range(len(models)), [NAMES[m] for m in models], fontsize=6.5, rotation=14, ha="right")
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#52514e", labelsize=7)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#c3c2b7")
    ax.legend(loc="upper center", ncols=4, frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 1.18), columnspacing=1.0, handlelength=1.4, handletextpad=0.5)
    fig.tight_layout()
    out = os.path.join(HERE, f"side_task_bars_{sweep}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight", pad_inches=0.06)
    print("wrote", out)
