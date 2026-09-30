import json
import math
import os
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "data.json")))

STYLE = {
    "messaging": ("Messaging (steg)", "#2a78d6", "o"),
    "reasoning": ("Reasoning (steg)", "#e34948", "s"),
    "wordprefix": ("Encoded reasoning", "#eda100", "^"),
}
N_TEST_EPISODES = 64  # test/env/all/total_episodes in each run's metrics.jsonl
POSITIONS = [1, 2, 3, 4, 5]
Z = 1.645  # 90% CI


def wilson(p, n):
    center = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
    half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / (1 + Z * Z / n)
    return p - (center - half), (center + half) - p


fig, ax = plt.subplots(figsize=(7, 4.5))
for i, (run, (label, color, marker)) in enumerate(STYLE.items()):
    final_step = data[run]["test/env/all/reasoning_step_1_correct"][-1][0]
    ps = [data[run][f"test/env/all/reasoning_step_{k}_correct"][-1][1] for k in POSITIONS]
    yerr = list(zip(*[wilson(p, N_TEST_EPISODES) for p in ps]))
    x = [k + (i - 1) * 0.07 for k in POSITIONS]
    ax.errorbar(x, ps, yerr=yerr, color=color, marker=marker, markersize=5,
                linewidth=1.8, capsize=3, capthick=1.2, label=f"{label}, step {final_step}")
ax.axhline(0.5, color="#898781", linestyle="--", linewidth=1, label="Chance")
ax.set_xlabel("Position in side-task sequence", fontsize=11)
ax.set_ylabel("Test per-position decode accuracy", fontsize=11)
ax.set_xticks(POSITIONS)
ax.set_ylim(0.37, 1.05)
ax.grid(color="#e1e0d9", linewidth=0.5)
ax.set_axisbelow(True)
ax.tick_params(colors="#52514e")
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color("#c3c2b7")
ax.legend(loc="lower left", frameon=False, fontsize=10)

fig.tight_layout()
out = os.path.join(HERE, "rl_per_position.png")
fig.savefig(out, dpi=300)
print("wrote", out)
