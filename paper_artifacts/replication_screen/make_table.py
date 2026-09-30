import json
import math
import os

HERE = os.path.dirname(os.path.abspath(__file__))
data = json.load(open(os.path.join(HERE, "data.json")))
LENGTHS = data["lengths"]

MODELS = ["deepseek_v3_0324", "deepseek_v3_1", "qwen3_235b_a22b_2507", "kimi_k2_0905",
          "gpt_5_nano", "mistral_medium_3", "gemini_2_5_flash"]
NAMES = {"deepseek_v3_0324": "DeepSeek-V3-0324", "deepseek_v3_1": "DeepSeek-V3.1",
         "qwen3_235b_a22b_2507": "Qwen3-235B-A22B", "kimi_k2_0905": "Kimi-K2-0905",
         "gpt_5_nano": "GPT-5-nano", "mistral_medium_3": "Mistral-Medium-3",
         "gemini_2_5_flash": "Gemini-2.5-Flash"}


SLICES = {"table.tex": LENGTHS}


def mcnemar_p(model, a, b, lens):
    cell = data["mcnemar"][model][f"{a}_vs_{b}"]
    b10 = sum(cell[f"len{ln}"]["b10"] for ln in lens)
    b01 = sum(cell[f"len{ln}"]["b01"] for ln in lens)
    n = b10 + b01
    return 1.0 if n == 0 else sum(math.comb(n, i) for i in range(b10, n + 1)) / 2 ** n


def acc(model, cond, lens):
    cells = [data["models"][model][cond][f"len{ln}"] for ln in lens]
    return sum(c["n_correct"] for c in cells) / sum(c["n"] for c in cells)


def trace(model, cond, lens):
    cells = [data["models"][model][cond][f"len{ln}"] for ln in lens]
    return sum(c["trace_sum"] for c in cells) / sum(c["trace_n"] for c in cells)


def fmt_p(p):
    s = "<.001" if p < 0.001 else f"{p:.3f}".replace("0.", ".", 1)
    return r"{\boldmath$" + s + "$}" if p < 0.05 else f"${s}$"


for fname, lens in SLICES.items():
    lines = [r"{\footnotesize\setlength{\tabcolsep}{2.6pt}", r"\begin{tabular}{lcccccccccc}", r"\toprule",
             r" & \multicolumn{2}{c}{Baselines} & \multicolumn{4}{c}{First-letter} & \multicolumn{4}{c}{Word-count} \\",
             r"\cmidrule(lr){2-3} \cmidrule(lr){4-7} \cmidrule(lr){8-11}",
             r"Model & original & matched & acc. & $p$ (orig.) & $p$ (match.) & trace & acc. & $p$ (orig.) & $p$ (match.) & trace \\",
             r"\midrule"]
    for m in MODELS:
        row = [NAMES[m], f"{acc(m, 'no_encoding', lens):.3f}", f"{acc(m, 'restated', lens):.3f}"]
        for cond in ["first_letter", "word_count"]:
            row += [f"{acc(m, cond, lens):.3f}",
                    fmt_p(mcnemar_p(m, cond, "no_encoding", lens)),
                    fmt_p(mcnemar_p(m, cond, "restated", lens)),
                    f"{trace(m, cond, lens):.2f}"]
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}"]
    with open(os.path.join(HERE, fname), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote", fname)
    for line in lines:
        print("  ", line)
