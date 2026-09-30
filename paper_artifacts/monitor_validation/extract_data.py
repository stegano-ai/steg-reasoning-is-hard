import json
import math
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(EXPERIMENT_REPO, "experiments", "monitor_benchmarking", "results")
# The per-example ratings (ratings.jsonl) are gitignored in the experiment repo; the committed
# per-type table (written by experiments/monitor_benchmarking/summarize.py from those ratings,
# excluding monitor-error rows) is the source here. Fractions are converted back to counts.
TABLES_MD = os.path.join(RESULTS, "paper_validation_tables.md")

RUNS = {
    "rl_base": {"prompt": "base", "main_task": "aqua_rat", "min_prefix": False, "n_symbols": 2, "seq_length": 5},
    "icl_opening": {"prompt": "opening", "main_task": "wildchat", "min_prefix": True, "n_symbols": 5, "seq_length": 10},
    "icl_complete": {"prompt": "complete", "main_task": "wildchat", "min_prefix": False, "n_symbols": 5, "seq_length": 10},
    "lenient_computation": {"prompt": "lenient", "main_task": "aqua_rat", "min_prefix": False, "n_symbols": 2, "seq_length": 5},
}
ROW = re.compile(r"^\| (\S+) \| (\d+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \([\d.]+–[\d.]+\) \| ([\d.]+) \| (\d+) \| (\d+) \|$")


def parse_tables(path):
    runs, cur = {}, None
    for line in open(path):
        if line.startswith("### "):
            cur = line[4:].strip()
            runs[cur] = {}
            continue
        m = ROW.match(line.strip())
        if m and cur:
            typ, n, mean, sd, le3, ge7, lo, hi = m.groups()
            n = int(n)
            runs[cur][typ] = {"n": n, "mean": float(mean), "sd": float(sd),
                              "k_le3": round(float(le3) * n), "k_ge7": round(float(ge7) * n),
                              "min": int(lo), "max": int(hi)}
    return runs


def pooled(cells):
    n = sum(c["n"] for c in cells)
    mean = sum(c["n"] * c["mean"] for c in cells) / n
    ss = sum((c["n"] - 1) * c["sd"] ** 2 + c["n"] * (c["mean"] - mean) ** 2 for c in cells)
    return {"n": n, "mean": mean, "sd": math.sqrt(ss / (n - 1)),
            "k_le3": sum(c["k_le3"] for c in cells), "k_ge7": sum(c["k_ge7"] for c in cells),
            "min": min(c["min"] for c in cells), "max": max(c["max"] for c in cells)}


tables = parse_tables(TABLES_MD)
out = {"monitors": {}}
for run, meta in RUNS.items():
    types = dict(tables[run])
    # opening monitor: the unencoded control is the same-length prefix paired with each scheme;
    # keep the per-scheme cells and add one pooled "natural" cell
    nat = {k: v for k, v in types.items() if k.startswith("natural|")}
    if nat:
        types["natural"] = pooled(list(nat.values()))
    out["monitors"][run] = {**meta, "types": types}

with open(os.path.join(HERE, "data.json"), "w") as f:
    json.dump(out, f, indent=2)
for run, m in out["monitors"].items():
    print(run, {t: (c["n"], round(c["mean"], 2), round(c["k_le3"] / c["n"], 3)) for t, c in m["types"].items()})
