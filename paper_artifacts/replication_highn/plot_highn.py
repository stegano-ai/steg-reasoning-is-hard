import json
import os
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "data.json")))
LENGTHS = data["lengths"]
Z90 = 1.6449

STYLE = {"gpt_5_nano": ("GPT-5-nano", "#2a78d6", "o"), "qwen3_235b": ("Qwen3-235B", "#e34948", "s")}
ARM_STYLE = {"restated": ("Matched\nbaseline", "#898781"), "latent": ("Latent\ncontrol", "#eda100"),
             "first_letter": ("First-letter", "#e34948")}


def wilson(k, n):
    p = k / n
    d = 1 + Z90 * Z90 / n
    c = (p + Z90 * Z90 / (2 * n)) / d
    h = Z90 * ((p * (1 - p) / n + Z90 * Z90 / (4 * n * n)) ** 0.5) / d
    return p, max(0, c - h), min(1, c + h)


def paired_delta(cellv):
    # difference of correlated proportions; only discordant pairs carry information
    n, b10, b01 = cellv["n"], cellv["b10"], cellv["b01"]
    d = (b10 - b01) / n
    var = (b10 + b01 - (b10 - b01) ** 2 / n) / n ** 2
    return d, Z90 * var ** 0.5


def trend(model):
    # inverse-variance weighted least-squares slope of the uplift against sequence length
    xs, ds, ws = LENGTHS, [], []
    for ln in LENGTHS:
        c = data["uplift"][model][f"len{ln}"]
        d, half = paired_delta(c)
        ds.append(d)
        ws.append((Z90 / half) ** 2)
    xb = sum(w * x for w, x in zip(ws, xs)) / sum(ws)
    yb = sum(w * y for w, y in zip(ws, ds)) / sum(ws)
    sxx = sum(w * (x - xb) ** 2 for w, x in zip(ws, xs))
    slope = sum(w * (x - xb) * (y - yb) for w, x, y in zip(ws, xs, ds)) / sxx
    return slope, sxx ** -0.5


fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.6))

# (a) paired uplift of the first-letter arm over the matched baseline, by sequence length
ax = axes[0]
for model, (label, color, marker) in STYLE.items():
    ds, es = zip(*(paired_delta(data["uplift"][model][f"len{ln}"]) for ln in LENGTHS))
    slope, se = trend(model)
    ax.errorbar(LENGTHS, [100 * d for d in ds], yerr=[100 * e for e in es], color=color, marker=marker,
                markersize=5, linewidth=1.8, capsize=2.5, elinewidth=0.9,
                label=f"{label} ({100 * slope:+.2f} pp/step)")
    print(f"{model}: slope {100 * slope:+.3f} pp per step, se {100 * se:.3f}, z={slope / se:+.2f}")
ax.axhline(0, color="#898781", linestyle="--", linewidth=1)
ax.set_xlabel("Sequence length", fontsize=10)
ax.set_ylabel("First-letter $-$ matched baseline (pp)", fontsize=10)
ax.set_xticks(LENGTHS)
ax.legend(frameon=False, fontsize=9, loc="lower left")

# (b) post-hoc decoded trace accuracy of the same first-letter completions
ax = axes[1]
for model, (label, color, marker) in STYLE.items():
    cells = [data["uplift"][model][f"len{ln}"] for ln in LENGTHS]
    ax.plot(LENGTHS, [c["trace_sum"] / c["trace_n"] for c in cells],
            color=color, marker=marker, markersize=5, linewidth=1.8, label=label)
ax.axhline(data["chance"], color="#898781", linestyle="--", linewidth=1, label="Chance")
ax.set_xlabel("Sequence length", fontsize=10)
ax.set_ylabel("Decoded trace accuracy", fontsize=10)
ax.set_ylim(0, 1.0)
ax.set_xticks(LENGTHS)
ax.legend(frameon=False, fontsize=9, loc="upper right")

# (c) latent-reasoning control at the strongest cell (Qwen3-235B, length 12)
ax = axes[2]
marg = data["latent"]["marginal"]
order = ["restated", "latent", "first_letter"]
for i, arm in enumerate(order):
    p, lo, hi = wilson(marg[arm]["n_correct"], marg[arm]["n"])
    label, color = ARM_STYLE[arm]
    ax.bar(i, p, color=color, width=0.62)
    ax.errorbar(i, p, yerr=[[p - lo], [hi - p]], color="#2d3436", capsize=3, elinewidth=0.9, linestyle="")
ax.axhline(data["chance"], color="#898781", linestyle="--", linewidth=1)
ax.set_xticks(range(len(order)))
ax.set_xticklabels([ARM_STYLE[a][0] for a in order], fontsize=9)
ax.set_ylabel("Final-answer accuracy", fontsize=10)
ax.set_ylim(0, 0.72)

for ax in axes:
    ax.grid(color="#e1e0d9", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(colors="#52514e", labelsize=9)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color("#c3c2b7")
for ax, letter in zip(axes, "abc"):
    ax.text(-0.02, 1.05, f"{letter})", transform=ax.transAxes, fontsize=13, fontweight="bold",
            va="bottom", ha="right", color="#2d3436")

fig.tight_layout()
out = os.path.join(HERE, "replication_highn.png")
fig.savefig(out, dpi=300)
print("wrote", out)
