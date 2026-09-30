import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
# Raw completions are gitignored; every non-text metric lives in this committed CSV.
# That is also why this pass has no decoded traces: decoding needs the completion text.
# Trace accuracies for the two survivor models are in ../replication_highn.
CSV = os.path.join(EXPERIMENT_REPO, "experiments", "prompting", "results_metrics", "paper_replication.csv")

MODELS = ["deepseek_v3_0324", "deepseek_v3_1", "qwen3_235b_a22b_2507", "kimi_k2_0905",
          "gpt_5_nano", "mistral_medium_3", "gemini_2_5_flash"]
CONDS = ["no_encoding", "restated", "first_letter", "word_count"]
LENGTHS = [3, 5, 8, 12]
PAIRS = [("first_letter", "no_encoding"), ("word_count", "no_encoding"),
         ("first_letter", "restated"), ("word_count", "restated"), ("restated", "no_encoding")]


rows = list(csv.DictReader(open(CSV)))


def cell(model, cond, ln):
    # sample_index is the pairing key: within a model, the same index is the same cover
    # question and the same hidden sequence in every arm. Verified by expected_reasoning
    # agreeing on every shared index. The two survivor models sit on a larger cover pool
    # than the other five, so pairing is only ever done within a model, never across.
    return {int(r["sample_index"]): int(round(float(r["output_correct"])))
            for r in rows if r["cell"].endswith(f"eval_{model}_{cond}_len{ln}") and r["output_correct"] != ""}


def traces(model, cond, ln):
    # post-hoc decode of the prompt scheme, written per row by extract_metrics; populated
    # only for the arms that were asked to encode. Chance is 0.5.
    return {int(r["sample_index"]): float(r["trace_correct"])
            for r in rows if r["cell"].endswith(f"eval_{model}_{cond}_len{ln}") and r["trace_correct"] != ""}


out = {"chance": 0.5,
       "lengths": LENGTHS, "conds": CONDS, "models": {}, "mcnemar": {}}

for model in MODELS:
    cells = {c: {ln: cell(model, c, ln) for ln in LENGTHS} for c in CONDS}
    # Restrict every length to the samples present in all four arms, so that one table row
    # reports four accuracies and their tests on the same underlying samples. For the two
    # survivor models the encoding arms hold more samples than this; the full-strength
    # contrast between them is in ../replication_highn.
    common = {ln: sorted(set.intersection(*(set(cells[c][ln]) for c in CONDS))) for ln in LENGTHS}
    tr = {c: {ln: traces(model, c, ln) for ln in LENGTHS} for c in ["first_letter", "word_count"]}
    out["models"][model] = {
        c: {f"len{ln}": {"n": len(common[ln]),
                         "n_correct": sum(cells[c][ln][i] for i in common[ln]),
                         **({"trace_sum": round(sum(tr[c][ln][i] for i in common[ln] if i in tr[c][ln]), 4),
                             "trace_n": sum(i in tr[c][ln] for i in common[ln])} if c in tr else {})}
            for ln in LENGTHS} for c in CONDS}
    out["mcnemar"][model] = {}
    for a, b in PAIRS:
        out["mcnemar"][model][f"{a}_vs_{b}"] = {
            f"len{ln}": {"b10": sum(cells[a][ln][i] > cells[b][ln][i] for i in common[ln]),
                         "b01": sum(cells[a][ln][i] < cells[b][ln][i] for i in common[ln]),
                         "n_pairs": len(common[ln])}
            for ln in LENGTHS}
    print(f"{model:22s} samples in all four arms per length:", [len(common[ln]) for ln in LENGTHS])

with open(os.path.join(HERE, "data.json"), "w") as f:
    json.dump(out, f, indent=1)
print("wrote", os.path.join(HERE, "data.json"))
