import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
RESULTS = os.path.join(EXPERIMENT_REPO, "experiments", "sft", "results")


# Each condition is a list of (run_dir, step_offset) segments; cont10x runs resume
# from the 2x run's final checkpoint, so their offset is that run's final step.
# Qwen runs are the paper's main-line runs; the mod-3 reasoning offsets are the
# resumed checkpoints' batches (3500, then 3500 + 3916).
SWEEP = {
    "mod3": {
        "inkling": {
            "encoded": [("model_sweep/encoded_reasoning/train_sft_encoded_reasoning_inkling_2026-08-15_13-56-56", 0)],
            "messaging": [("model_sweep/steg_messaging/train_sft_steg_messaging_inkling_2026-08-15_13-56-56", 0)],
            "reasoning": [("model_sweep/steg_reasoning/train_sft_steg_reasoning_inkling_2026-08-16_18-13-10", 0)],
        },
        "qwen": {
            "encoded": [("steg_reasoning/train_sft_steg_reasoning_word_prefix_2026-06-17_16-29-59", 0)],
            "messaging": [("steg_messaging/train_sft_steg_messaging_markdown_2026-07-31_16-02-29", 0)],
            "reasoning": [
                ("steg_reasoning/train_sft_steg_reasoning_markdown_summod3_2026-07-07_16-30-35", 0),
                ("steg_reasoning/train_sft_steg_reasoning_markdown_summod3_2026-07-08_13-40-29", 3500),
                ("steg_reasoning/train_sft_steg_reasoning_markdown_summod3_2026-07-10_17-36-18", 7416),
            ],
        },
        "nemotron_super": {
            "encoded": [("model_sweep/encoded_reasoning/train_sft_encoded_reasoning_nemotron3_super_2026-08-13_20-44-44", 0)],
            "messaging": [("model_sweep/steg_messaging/train_sft_steg_messaging_nemotron3_super_2026-08-13_20-44-44", 0)],
            "reasoning": [
                ("model_sweep/steg_reasoning/train_sft_steg_reasoning_nemotron3_super_2026-08-13_22-27-47", 0),
                ("model_sweep/steg_reasoning/train_sft_steg_reasoning_nemotron3_super_cont10x_2026-08-14_05-58-56", 1400),
            ],
        },
        "gptoss": {
            "encoded": [("model_sweep/encoded_reasoning/train_sft_encoded_reasoning_gptoss_120b_2026-08-13_21-00-13", 0)],
            "messaging": [("model_sweep/steg_messaging/train_sft_steg_messaging_gptoss_120b_2026-08-13_22-20-31", 0)],
            "reasoning": [
                ("model_sweep/steg_reasoning/train_sft_steg_reasoning_gptoss_120b_2026-08-13_23-02-29", 0),
                ("model_sweep/steg_reasoning/train_sft_steg_reasoning_gptoss_120b_cont10x_2026-08-14_05-58-56", 2000),
            ],
        },
        "inkling_small": {
            "encoded": [("model_sweep/encoded_reasoning/train_sft_encoded_reasoning_inkling_small_2026-08-13_21-13-21", 0)],
            "messaging": [("model_sweep/steg_messaging/train_sft_steg_messaging_inkling_small_closed_2026-08-14_15-17-18", 0)],
            "reasoning": [
                ("model_sweep/steg_reasoning/train_sft_steg_reasoning_inkling_small_2026-08-15_13-56-56", 0),
                ("model_sweep/steg_reasoning/train_sft_steg_reasoning_inkling_small_cont10x_2026-08-16_18-13-10", 2200),
            ],
        },
        "lightning": {
            "encoded": [("model_sweep/encoded_reasoning/train_sft_encoded_reasoning_nemotron35_lightning_2026-08-13_21-25-32", 0)],
            "messaging": [("model_sweep/steg_messaging/train_sft_steg_messaging_nemotron35_lightning_2026-08-13_22-20-31", 0)],
            "reasoning": [
                ("model_sweep/steg_reasoning/train_sft_steg_reasoning_nemotron35_lightning_2026-08-13_23-36-49", 0),
                ("model_sweep/steg_reasoning/train_sft_steg_reasoning_nemotron35_lightning_cont10x_2026-08-14_05-58-56", 4200),
            ],
        },
    },
    "a5ws": {
        "qwen": {
            "encoded": [("model_sweep/encoded_reasoning_a5ws/train_sft_encoded_reasoning_a5ws_qwen36_2026-08-17_11-42-13", 0)],
            "messaging": [("model_sweep/steg_messaging_a5ws/train_sft_steg_messaging_a5ws_qwen36_2026-08-17_11-42-13", 0)],
            "reasoning": [("model_sweep/steg_reasoning_a5ws/train_sft_steg_reasoning_a5ws_qwen36_2026-08-17_12-30-55", 0)],
        },
        "nemotron_super": {
            "encoded": [("model_sweep/encoded_reasoning_a5ws/train_sft_encoded_reasoning_a5ws_nemotron3_super_2026-08-15_16-02-36", 0)],
            "messaging": [("model_sweep/steg_messaging_a5ws/train_sft_steg_messaging_a5ws_nemotron3_super_2026-08-15_16-00-36", 0)],
            "reasoning": [
                ("model_sweep/steg_reasoning_a5ws/train_sft_steg_reasoning_a5ws_nemotron3_super_2026-08-15_18-06-23", 0),
                ("model_sweep/steg_reasoning_a5ws/train_sft_steg_reasoning_a5ws_nemotron3_super_cont10x_2026-08-21_11-00-57", 1000),
            ],
        },
        "gptoss": {
            "encoded": [("model_sweep/encoded_reasoning_a5ws/train_sft_encoded_reasoning_a5ws_gptoss_120b_2026-08-15_16-02-36", 0)],
            "messaging": [("model_sweep/steg_messaging_a5ws/train_sft_steg_messaging_a5ws_gptoss_120b_2026-08-15_16-00-36", 0)],
            "reasoning": [
                ("model_sweep/steg_reasoning_a5ws/train_sft_steg_reasoning_a5ws_gptoss_120b_2026-08-15_18-06-23", 0),
                ("model_sweep/steg_reasoning_a5ws/train_sft_steg_reasoning_a5ws_gptoss_120b_cont10x_2026-08-21_11-00-56", 1600),
            ],
        },
        "inkling_small": {
            "encoded": [("model_sweep/encoded_reasoning_a5ws/train_sft_encoded_reasoning_a5ws_inkling_small_2026-08-15_16-02-36", 0)],
            "messaging": [("model_sweep/steg_messaging_a5ws/train_sft_steg_messaging_a5ws_inkling_small_2026-08-15_16-00-36", 0)],
            "reasoning": [
                ("model_sweep/steg_reasoning_a5ws/train_sft_steg_reasoning_a5ws_inkling_small_2026-08-15_18-06-23", 0),
                ("model_sweep/steg_reasoning_a5ws/train_sft_steg_reasoning_a5ws_inkling_small_cont10x_2026-08-21_11-00-56", 600),
            ],
        },
        "lightning": {
            "encoded": [("model_sweep/encoded_reasoning_a5ws/train_sft_encoded_reasoning_a5ws_nemotron35_lightning_2026-08-15_16-02-36", 0)],
            "messaging": [("model_sweep/steg_messaging_a5ws/train_sft_steg_messaging_a5ws_nemotron35_lightning_2026-08-15_17-08-01", 0)],
            "reasoning": [
                ("model_sweep/steg_reasoning_a5ws/train_sft_steg_reasoning_a5ws_nemotron35_lightning_2026-08-15_18-23-21", 0),
                ("model_sweep/steg_reasoning_a5ws/train_sft_steg_reasoning_a5ws_nemotron35_lightning_cont10x_2026-08-21_11-00-56", 3800),
            ],
        },
    },
}

out = {}
for variant, models in SWEEP.items():
    out[variant] = {}
    for model, conditions in models.items():
        out[variant][model] = {}
        for cond, segments in conditions.items():
            series = []
            for run_dir, offset in segments:
                for line in open(os.path.join(RESULTS, run_dir, "metrics.jsonl")):
                    row = json.loads(line)
                    if "eval/reasoning_correct" not in row:
                        continue
                    if offset and row["step"] == 0:
                        continue  # duplicate eval of the checkpoint the segment resumed from
                    series.append([row["step"] + offset, row["eval/reasoning_correct"], row["eval/reasoning_exact"]])
            out[variant][model][cond] = series
        print(variant, model, "->", {c: len(s) for c, s in out[variant][model].items()})

with open(os.path.join(HERE, "data.json"), "w") as f:
    json.dump(out, f)
print("wrote", os.path.join(HERE, "data.json"))
