import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
# The prompting experiment moved to content-addressed cells: raw completions are
# gitignored and every non-text metric lives in this committed CSV.
CSV = os.path.join(EXPERIMENT_REPO, "experiments", "prompting", "results_metrics", "paper_replication.csv")

MODELS = {"gpt_5_nano": "gpt_5_nano", "qwen3_235b": "qwen3_235b_a22b_2507"}
LENGTHS = [3, 5, 8, 12]

# trace_correct is the post-hoc decode of the prompt schemes, written per row by
# extract_metrics: the fraction of hidden steps recovered from the completion, best of the
# two alignments. Chance is 0.5. It is populated only for the arms that were asked to encode
# (first_letter, word_count).


rows = list(csv.DictReader(open(CSV)))


def cell(name):
    # sample_index is the exact pairing key: the same index is the same WildChat
    # cover question and the same hidden sequence in every arm of a run.
    return {int(r["sample_index"]): int(round(float(r["output_correct"])))
            for r in rows if r["cell"].endswith(name) and r["output_correct"] != ""}


def traces(name):
    return {int(r["sample_index"]): float(r["trace_correct"])
            for r in rows if r["cell"].endswith(name) and r["trace_correct"] != ""}


def paired(a, b):
    common = sorted(set(a) & set(b))
    return {"n": len(common),
            "a_correct": sum(a[i] for i in common),
            "b_correct": sum(b[i] for i in common),
            "b10": sum(a[i] > b[i] for i in common),
            "b01": sum(a[i] < b[i] for i in common)}


out = {"chance": 0.5, "lengths": LENGTHS,
       "uplift": {}, "latent": {}, "per_completion": {}}

for short, tag in MODELS.items():
    per_len, samples = {}, []
    for ln in LENGTHS:
        fl = cell(f"eval_{tag}_first_letter_len{ln}")
        rs = cell(f"eval_{tag}_restated_len{ln}")
        tr = traces(f"eval_{tag}_first_letter_len{ln}")
        wc_tr = traces(f"eval_{tag}_word_count_len{ln}")
        per_len[f"len{ln}"] = {**paired(fl, rs), "n_first_letter": len(fl), "n_restated": len(rs),
                               "trace_n": len(tr), "trace_sum": round(sum(tr.values()), 4),
                               "word_count_trace_n": len(wc_tr),
                               "word_count_trace_sum": round(sum(wc_tr.values()), 4)}
        # per-completion pairs for the does-encoding-predict-correctness test
        samples += [[round(tr[i], 4), fl[i], ln] for i in sorted(set(tr) & set(fl))]
    out["uplift"][short] = per_len
    out["per_completion"][short] = samples
    print(short, {k: (v["n"], v["b10"], v["b01"]) for k, v in per_len.items()},
          f"per-completion pairs: {len(samples)}")

# Latent control: the strongest uplift cell (qwen, length 12) with a third arm that
# writes a natural answer and then the final state, with no encoding instruction.
arms = {a: cell(f"eval_qwen3_235b_a22b_2507_{a}_len12") for a in ["restated", "latent", "first_letter"]}
out["latent"] = {
    "seq_length": 12,
    "marginal": {a: {"n": len(d), "n_correct": sum(d.values())} for a, d in arms.items()},
    "contrasts": {f"{x}_vs_{y}": paired(arms[x], arms[y])
                  for x, y in [("latent", "restated"), ("first_letter", "latent"), ("first_letter", "restated")]},
}
print("latent", {a: (v["n"], v["n_correct"]) for a, v in out["latent"]["marginal"].items()})

with open(os.path.join(HERE, "data.json"), "w") as f:
    json.dump(out, f, indent=1)
print("wrote", os.path.join(HERE, "data.json"))
