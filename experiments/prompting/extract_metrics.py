# Extract every non-text field from results/**/completions.jsonl into one compact CSV per
# top-level results group (results_metrics/<group>.csv). Drops the prose fields
# (cot, completion, designed_scheme, monitor reasoning) but keeps all scores, the int-list
# fields (space-joined), timestamps, and the monitor's cumulative_cost.
# For the paper_replication prompt-scheme arms (first_letter / word_count, where
# scheme: null means the harness has no decoder), the trace is decoded post-hoc here —
# same logic as the analyze scripts: first letter of each sentence N->0/S->1 (or words
# mod 2), scored against expected_reasoning at the better of the two alignments — into
# decoded_reasoning and trace_correct (fraction of hidden steps recovered).
import csv
import json
import sys
from pathlib import Path

import yaml

from lib.trace_decode import decode_first_letter, decode_word_count, trace_correct

RESULTS = Path("results")
OUT = Path("results_metrics")

SCALARS = ["sample_index", "timestamp", "output_format", "reasoning_format", "output_exact",
           "output_correct", "reasoning_exact", "reasoning_correct", "suspiciousness",
           "swap_output_correct", "swap_output_exact"]
LISTS = ["expected_output", "expected_reasoning", "actual_output", "actual_reasoning"]
IDENT = ["group", "cell", "model", "scheme", "seq_length", "side_task", "env_kind", "seed", "n_eval"]
COLS = IDENT + SCALARS + LISTS + ["monitor_cost", "decoded_reasoning", "trace_correct"]


PROMPT_SCHEME_DECODERS = {"first_letter": decode_first_letter, "word_count": decode_word_count}


def dig(d, *path):
    for p in path:
        if not isinstance(d, dict):
            return None
        d = d.get(p)
    return d


def identity(cell_dir):
    cfg_path = cell_dir / "config.yaml"
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    fk = cfg.get("function_kwargs", cfg)  # new format is flat; old scaffold nests here
    sampler = dig(fk, "sampler") or {}
    return {
        "model": sampler.get("model") or sampler.get("base_model"),
        "scheme": dig(fk, "main_task", "scheme"),
        "seq_length": dig(fk, "side_task", "seq_length"),
        "side_task": dig(fk, "side_task", "type"),
        "env_kind": fk.get("env_kind"),
        "seed": fk.get("seed"),
        "n_eval": fk.get("n_eval"),
    }


def extract_row(r, decoder=None):
    out = {k: r.get(k) for k in SCALARS}
    for k in LISTS:
        v = r.get(k)
        out[k] = " ".join(map(str, v)) if isinstance(v, list) else ""
    mc = r.get("monitor_cot")
    out["monitor_cost"] = mc.get("cumulative_cost") if isinstance(mc, dict) else None
    if decoder is not None:
        decoded = decoder(r["cot"] if r["cot"] is not None else r["completion"])
        out["decoded_reasoning"] = " ".join("-" if d is None else str(d) for d in decoded)
        out["trace_correct"] = round(trace_correct(decoded, r["expected_reasoning"]), 4)
    return out


def main():
    OUT.mkdir(exist_ok=True)
    for group in sorted(p for p in RESULTS.iterdir() if p.is_dir()):
        files = sorted(list(group.rglob("completions.jsonl")) + list(group.rglob("eval_completions.jsonl")))
        if not files:
            continue
        n_rows = 0
        with open(OUT / f"{group.name}.csv", "w", newline="") as fo:
            w = csv.DictWriter(fo, fieldnames=COLS)
            w.writeheader()
            for f in files:
                base = {"group": group.name, "cell": str(f.parent.relative_to(RESULTS)), **identity(f.parent)}
                decoder = next((d for arm, d in PROMPT_SCHEME_DECODERS.items()
                                if f"_{arm}_" in f.parent.name), None)
                for line in open(f):
                    w.writerow({**base, **extract_row(json.loads(line), decoder)})
                    n_rows += 1
        print(f"{group.name}: {len(files)} cells, {n_rows} rows -> {OUT / f'{group.name}.csv'}")


if __name__ == "__main__":
    main()
