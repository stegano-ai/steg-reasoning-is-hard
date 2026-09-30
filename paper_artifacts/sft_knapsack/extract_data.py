import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(EXPERIMENT_REPO, "experiments", "sft", "results")


# Qwen3.5-397B pair: identical configs except the side task (running sum mod 3
# vs the same nine values as a fixed message). qwen36_reasoning is the
# Qwen3.6-35B run of the same reasoning task (evaluated only every 100 steps).
RUNS = {
    "messaging": "steg_messaging/train_sft_steg_messaging_knapsack_2026-08-17_12-09-15",
    "reasoning": "steg_reasoning/train_sft_steg_reasoning_knapsack_summod3_2026-07-01_19-40-48",
    "qwen36_reasoning": "steg_reasoning/train_sft_steg_reasoning_knapsack_summod3_2026-06-21_17-20-10",
}

out = {}
for name, run_dir in RUNS.items():
    series = []
    for line in open(os.path.join(RESULTS, run_dir, "metrics.jsonl")):
        row = json.loads(line)
        if "eval/reasoning_correct" in row:
            series.append([row["step"], row["eval/reasoning_correct"], row["eval/reasoning_exact"]])
    out[name] = series
    print(name, "->", series)

with open(os.path.join(HERE, "data.json"), "w") as f:
    json.dump(out, f)
print("wrote", os.path.join(HERE, "data.json"))
