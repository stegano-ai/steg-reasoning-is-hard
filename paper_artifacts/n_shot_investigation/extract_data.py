import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(EXPERIMENT_REPO, "experiments", "prompting", "results")

NSHOT = "n_shot_investigation/nshot_ball_gpt55_2026-07-14_10-54"
NO_COT = "side_task_baseline/frontier_sweep_2026-07-07_22-48/eval_gpt_5_5_direct_len10"
SHOTS = [0, 1, 2, 3, 4, 5]
VOCAB = 5
Z90 = 1.6449


def frac_stats(run_dir, field):  # mean + 90% CI half-width of a per-record fraction
    vals = [json.loads(l)[field] for l in open(os.path.join(run_dir, "eval_completions.jsonl"))]
    n = len(vals)
    m = sum(vals) / n
    var = sum((v - m) ** 2 for v in vals) / (n - 1)
    return {"mean": round(m, 4), "ci": round(Z90 * (var / n) ** 0.5, 4)}


def output_counts(run_dir):
    n = c = 0
    for line in open(os.path.join(run_dir, "eval_completions.jsonl")):
        n += 1
        c += int(round(json.loads(line)["output_correct"]))
    return {"n_total": n, "n_correct": c}


nc = output_counts(os.path.join(RESULTS, NO_COT))
out = {
    "shots": SHOTS,
    "chance": 1 / VOCAB,
    "no_cot_reasoning": round(nc["n_correct"] / nc["n_total"], 4),
    "lines": {
        "steg_messaging": {"metric": "encoding", "per_shot": [frac_stats(os.path.join(RESULTS, NSHOT, f"eval_steg_messaging_shot{s}"), "reasoning_correct") for s in SHOTS]},
        "encoded_reasoning": {"metric": "output", "per_shot": [output_counts(os.path.join(RESULTS, NSHOT, f"eval_encoded_reasoning_shot{s}")) for s in SHOTS]},
        "steg_reasoning": {"metric": "output", "per_shot": [output_counts(os.path.join(RESULTS, NSHOT, f"eval_steg_reasoning_shot{s}")) for s in SHOTS]},
    },
}
with open(os.path.join(HERE, "data.json"), "w") as f:
    json.dump(out, f, indent=1)
print("wrote", os.path.join(HERE, "data.json"), "| no_cot:", out["no_cot_reasoning"])
