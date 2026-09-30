import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(EXPERIMENT_REPO, "experiments", "rl", "results")

ONLYOUTPUT = "knapsack_handholding_onlyoutput/train_grpo_knapsack_handholding_knapsack_handholding_onlyoutput_knapsack_handholding_onlyoutput"
RUNS = {
    "process_35b": "knapsack_handholding/train_grpo_knapsack_handholding_2026-07-28_16-53-06",
    "outcome_35b": f"{ONLYOUTPUT}_2026-07-29_08-58-46",
    "outcome_397b": f"{ONLYOUTPUT}_2026-07-29_13-33-53",
}
# Test evals run only every 20 steps, too sparse for a line, so we plot the
# per-step training metrics (env/all/*).
METRICS = {"reasoning_correct": "env/all/reasoning_correct", "output_correct": "env/all/output_correct", "suspiciousness": "env/all/suspiciousness"}


out = {"metric_source": "train"}
for run_name, run_dir in RUNS.items():
    series = {m: [] for m in METRICS}
    for line in open(os.path.join(RESULTS, run_dir, "metrics.jsonl")):
        row = json.loads(line)
        for m, key in METRICS.items():
            if key in row:
                series[m].append([row["step"], row[key]])
    model = json.load(open(os.path.join(RESULTS, run_dir, "config.json")))["model_name"]
    out[run_name] = {"run_dir": run_dir, "model": model, **series}
    print(run_name, model, "->", {m: len(v) for m, v in series.items()})

with open(os.path.join(HERE, "data.json"), "w") as f:
    json.dump(out, f, indent=1)
print("wrote", os.path.join(HERE, "data.json"))
