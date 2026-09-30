import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(EXPERIMENT_REPO, "experiments", "prompting", "results", "knapsack")

SEQ_LEN = 24
SWEEP = "knapsack_sweep_2026-07-10_12-53"
SONNET = "knapsack_sonnet_2026-07-10_13-49"
# sonnet was run separately (forgotten in the main sweep); placed at rank 2, below gpt-5.5.
MODELS = ["gpt_5_5", "sonnet_5", "gemini_3_5_flash", "deepseek_v4_pro", "kimi_2_6", "glm_5_1", "qwen3_5_397b", "qwen3_6_35b_a3b", "qwen3_6_27b"]


def run_dir(model, cond):
    base = SONNET if model == "sonnet_5" else SWEEP
    return os.path.join(RESULTS, base, f"eval_{model}_{cond}")


def per_position(model, cond):
    n_correct = [0] * SEQ_LEN
    n = 0
    for line in open(os.path.join(run_dir(model, cond), "eval_completions.jsonl")):
        r = json.loads(line)
        exp, act = r["expected_reasoning"], r["actual_reasoning"] or []
        n += 1
        for i in range(SEQ_LEN):
            if i < len(act) and act[i] == exp[i]:
                n_correct[i] += 1
    return {"n_total": n, "n_correct": n_correct}


def output_acc(model, cond):
    n = correct = 0
    for line in open(os.path.join(run_dir(model, cond), "eval_completions.jsonl")):
        r = json.loads(line)
        n += 1
        correct += int(round(r["output_correct"]))
    return {"n_total": n, "n_correct": correct}


out = {"seq_len": SEQ_LEN, "chance": 0.5, "models": {}}
for model in MODELS:
    out["models"][model] = {
        "lines": {"messaging": per_position(model, "messaging"), "steg": per_position(model, "steg")},
        "bars": {c: output_acc(model, c) for c in ["cot", "steg", "direct"]},
    }
    print(model, "done")

with open(os.path.join(HERE, "data.json"), "w") as f:
    json.dump(out, f, indent=1)
print("wrote", os.path.join(HERE, "data.json"))
