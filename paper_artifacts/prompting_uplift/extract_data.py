import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(EXPERIMENT_REPO, "experiments", "prompting", "results", "knapsack")

# gpt-5.5, 24-step parity side task, n=200 each; same run as plots/knapsack_sweep.
CONDITIONS = {
    "cot": "knapsack_sweep_2026-07-10_12-53/eval_gpt_5_5_cot",
    "steg_reasoning": "knapsack_sweep_2026-07-10_12-53/eval_gpt_5_5_steg",
    "no_cot": "knapsack_sweep_2026-07-10_12-53/eval_gpt_5_5_direct",
}


def cell(run_dir):
    n = correct = 0
    for line in open(os.path.join(RESULTS, run_dir, "eval_completions.jsonl")):
        r = json.loads(line)
        n += 1
        correct += int(r["output_correct"])
    return {"n_total": n, "n_correct": correct}


out = {"model": "gpt_5_5", "chance": 0.5, "seq_length": 24, "conditions": {c: cell(rd) for c, rd in CONDITIONS.items()}}
with open(os.path.join(HERE, "data.json"), "w") as f:
    json.dump(out, f, indent=1)
print("wrote", os.path.join(HERE, "data.json"), {c: v["n_correct"] for c, v in out["conditions"].items()})
