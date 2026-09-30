import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(EXPERIMENT_REPO, "experiments", "rl", "results")


RUNS = {
    "reasoning": "reasoning/train_grpo_reasoning_2026-07-15_09-31-17",
    "messaging": "messaging/train_grpo_messaging_2026-07-15_12-38-39",
    "wordprefix": "reasoning_wordprefix_process/train_grpo_reasoning_wordprefix_process_2026-07-15_08-54-22",
}
SKIP_PREFIXES = ("time/", "progress/", "step")
SKIP_SUBSTRINGS = ("tokens", "turns", "total_episodes")

out = {}
for run_name, run_dir in RUNS.items():
    series = {}
    for line in open(os.path.join(RESULTS, run_dir, "metrics.jsonl")):
        row = json.loads(line)
        step = row["step"]
        for key, value in row.items():
            if key.startswith(SKIP_PREFIXES) or any(s in key for s in SKIP_SUBSTRINGS):
                continue
            if isinstance(value, (int, float)):
                series.setdefault(key, []).append([step, value])
    out[run_name] = series
    print(run_name, "->", len(series), "metrics")

with open(os.path.join(HERE, "data.json"), "w") as f:
    json.dump(out, f)
print("wrote", os.path.join(HERE, "data.json"))
