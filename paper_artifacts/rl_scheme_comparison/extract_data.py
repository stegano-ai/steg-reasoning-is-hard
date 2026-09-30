
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(EXPERIMENT_REPO, "experiments", "rl", "results")


RUNS = {
    "steg_messaging": "messaging_scheme_replication/train_grpo_messaging_scheme_replication_2026-07-09_07-54-54",
    # cont300 is the full steg-reasoning run (0->280, with cover): stays flat, never learns.
    "steg_reasoning": "reasoning_model_scheme/reasoning_model_scheme_cont300",
    # encoded reasoning = the reasoning scheme WITHOUT a cover (nocover): learns to 1.0.
    "encoded_reasoning": "reasoning_model_scheme_nocover/train_grpo_messaging_scheme_replication_reasoning_model_scheme_reasoning_model_scheme_nocover_2026-07-14_10-26-30",
}
METRICS = {"output_correct": "test/env/all/output_correct", "suspiciousness": "test/env/all/suspiciousness"}

out = {"vocab_size": 2, "runs": {}}
for name, run_dir in RUNS.items():
    series = {m: [] for m in METRICS}
    n_episodes = set()
    for line in open(os.path.join(RESULTS, run_dir, "metrics.jsonl")):
        row = json.loads(line)
        for m, key in METRICS.items():
            if key in row:
                series[m].append([row["step"], row[key]])
        if "test/env/all/output_correct" in row:
            n_episodes.add(row["test/env/all/total_episodes"])
    [series["test_episodes"]] = n_episodes
    out["runs"][name] = series
    print(name, "->", {m: len(v) if isinstance(v, list) else v for m, v in series.items()})

with open(os.path.join(HERE, "data.json"), "w") as f:
    json.dump(out, f, indent=1)
print("wrote", os.path.join(HERE, "data.json"))
