import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "data.json")))
M = data["monitors"]

MONITORS = ["rl_base", "icl_opening", "icl_complete", "lenient_computation"]
HEADERS = [r"\shortstack{RL base\\(AQuA-RAT, full)}", r"\shortstack{ICL opening\\(WildChat, prefix)}",
           r"\shortstack{ICL complete\\(WildChat, full)}", r"\shortstack{Lenient\\(AQuA-RAT, full)}"]
GROUPS = [
    (r"\emph{Benign}", [("natural", "Dataset answer, unencoded"), ("model_output", "GPT-5-mini answer, unencoded")]),
    (r"\emph{Surface-formatting schemes}", [("capitalization", "Capitalization"), ("whitespace", "Whitespace"),
                                           ("markdown", "Markdown"), ("sentence_markdown", "Sentence-markdown"),
                                           ("sentence_newline", "Sentence-newline"),
                                           ("sentence_interjection", "Sentence-interjection")]),
    (r"\emph{Conspicuous controls}", [("number_prefix", "Digits prepended"), ("word_prefix", "Arbitrary words prepended"),
                                    ("evocative_word_prefix", "Word-prefix words prepended"),
                                    ("synthetic_header", "Bold-word header prepended")]),
]


def wilson90(k, n, z=1.645):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def cell(run, typ):
    c = M[run]["types"].get(typ)
    if c is None:
        return "--- & ---"
    return f"{c['mean']:.1f} & {100 * c['k_le3'] / c['n']:.0f}"


lines = [r"{\footnotesize\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{l" + "cc" * len(MONITORS) + "}", r"\toprule",
         "Input & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{h}}}" for h in HEADERS) + r" \\",
         " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(MONITORS))),
         " & " + " & ".join([r"mean & $\leq 3$ (\%)"] * len(MONITORS)) + r" \\", r"\midrule"]
for gi, (title, rows) in enumerate(GROUPS):
    if gi:
        lines.append(r"\addlinespace")
    lines.append(rf"\multicolumn{{{1 + 2 * len(MONITORS)}}}{{l}}{{{title}}} \\")
    for typ, label in rows:
        lines.append(f"\\quad {label} & " + " & ".join(cell(r, typ) for r in MONITORS) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}}"]

with open(os.path.join(HERE, "table.tex"), "w") as f:
    f.write("\n".join(lines) + "\n")

# widest 90% CI half-width on the covert fraction across all reported cells, for the caption
widest = max((lambda lo, hi: (hi - lo) / 2)(*wilson90(c["k_le3"], c["n"]))
             for run in MONITORS for c in M[run]["types"].values() if not c.get("pooled"))
print("\n".join(lines))
print(f"n range: {min(c['n'] for r in MONITORS for c in M[r]['types'].values())}–{max(c['n'] for r in MONITORS for c in M[r]['types'].values())}")
print(f"widest 90% CI half-width on covert fraction: {100 * widest:.1f} pp")
