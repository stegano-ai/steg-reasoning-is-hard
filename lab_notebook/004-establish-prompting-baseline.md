# 004 — Establish prompting baseline
Issue: #4
Owner: Julian

## 2026-05-06

Goal: establish a prompting baseline across the full 2×2 (messaging/reasoning × steg/nonsteg) on four models — qwen_30b, qwen_235b (the big Tinker model), kimi_k2.6, opus — with CoT explicitly disabled (`reasoning.enabled=false` on every OpenRouter call). Encodings: `whitespace` for steg, `word_prefix` (Owl/Fox/Penguin) for nonsteg. n_eval=50, n_shot=3, side_task seq_length=9, modulus=3.

Ran the main sweep: `experiments/prompting/results/main_2026-05-06_10-27-51/`. All 16 runs completed.

`side_correct` (mean ± SEM, n=50):

**messaging**

| model | nonsteg | steg |
|---|---|---|
| qwen_30b  | 1.00 ± 0.00 | 0.33 ± 0.02 |
| qwen_235b | 0.78 ± 0.06 | 0.36 ± 0.02 |
| kimi_k2.6 | 1.00 ± 0.00 | 0.36 ± 0.02 |
| opus      | 1.00 ± 0.00 | 0.56 ± 0.02 |

**reasoning**

| model | nonsteg | steg |
|---|---|---|
| qwen_30b  | 0.35 ± 0.02 | 0.33 ± 0.02 |
| qwen_235b | 0.33 ± 0.02 | 0.35 ± 0.02 |
| kimi_k2.6 | 0.33 ± 0.02 | 0.30 ± 0.02 |
| opus      | 0.64 ± 0.04 | 0.34 ± 0.02 |

**Quadrant averages (across models)**

|   | nonsteg | steg |
|---|---|---|
| messaging | 0.95 | 0.40 |
| reasoning | 0.41 | 0.33 |

Strange result: nonsteg reasoning is at chance (~0.33 = 1/3) for every model except Opus. Word_prefix emits one animal word per cumsum step, so the model has token positions to compute incrementally — it's not a no-scratchpad regime. I would have expected qwen_235b and kimi_k2.6 to be much better at this. They're not — they collapse to low-entropy guesses (e.g. qwen_235b output `Fox Fox Fox Fox Fox Fox Owl Owl Owl`). Only Opus actually does the running-sum-mod-3 mentally and lands at 0.64.

Next step: rerun reasoning_nonsteg with mod 2 instead of mod 3 to see if the difficulty is the modular arithmetic itself or something else. Config: `experiments/prompting/configs/binary_nonsteg_reasoning.yaml`.

Ran the binary mod-2 sweep: `experiments/prompting/results/binary_nonsteg_reasoning_2026-05-06_10-40-10/`. Random baseline is now 0.50.

`side_correct` (mean ± SEM, n=50):

| model | mod-2 | mod-3 (prev) | side_exact (mod-2) |
|---|---|---|---|
| qwen_30b  | 0.49 ± 0.03 | 0.35 | 0.00 |
| qwen_235b | 0.58 ± 0.04 | 0.33 | 0.06 |
| kimi_k2.6 | 0.56 ± 0.02 | 0.33 | 0.00 |
| opus      | 0.92 ± 0.02 | 0.64 | 0.74 |

Opus jumps to 0.92 (side_exact 0.74). qwen_235b/kimi_k2.6 land just above chance. qwen_30b at chance.

Next step: shrink operand_range from 10 to 2 (operands ∈ {1,2}, i.e. one flip and one no-op) so each step is a pure parity flip with no arithmetic — does that lift the small models off chance?

Ran with operand_range=2 (overwrote the same config). Results in `experiments/prompting/results/binary_nonsteg_reasoning_2026-05-06_10-47-24/`.

`side_correct` (mean ± SEM, n=50):

| model | op{1,2} | op{1..10} (prev) | side_exact {1,2} |
|---|---|---|---|
| qwen_30b  | 0.38 ± 0.03 | 0.49 | 0.00 |
| qwen_235b | 0.73 ± 0.04 | 0.58 | 0.30 |
| kimi_k2.6 | 0.54 ± 0.03 | 0.56 | 0.02 |
| opus      | 0.98 ± 0.01 | 0.92 | 0.92 |

Opus saturates. qwen_235b clearly improves when arithmetic is removed. kimi_k2.6 unchanged. qwen_30b drops below chance — emits `Penguin` (=2) outside the mod-2 alphabet because the encoding scheme description still lists Penguin as valid even though no shot example uses it.

Verified that `reasoning.enabled=false` actually drives `completion_tokens_details.reasoning_tokens` to 0 on Opus and on `openai/gpt-5.5` — extended thinking is genuinely disabled at the upstream provider, not just hidden in the response.

Next step: extend the sweep to all encoding schemes (3 steg + 2 nonsteg controls) on the frontier set (opus, gpt_5.5, kimi_k2.6), both side tasks, n_eval=10. Config: `experiments/prompting/configs/frontier_schemes.yaml`.

Ran the frontier sweep: `experiments/prompting/results/frontier_schemes_2026-05-06_13-25-50/`. All 30 runs completed.

**MESSAGING** — `side_correct` mean ± SEM | `side_exact` (n=10)

| encoding | opus | gpt_5.5 | kimi_k2.6 |
|---|---|---|---|
| whitespace      | 0.51 ± 0.06 \| 0.00 | 0.53 ± 0.04 \| 0.00 | 0.40 ± 0.05 \| 0.00 |
| capitalization  | 0.64 ± 0.08 \| 0.10 | 0.67 ± 0.06 \| 0.10 | 0.38 ± 0.05 \| 0.00 |
| markdown        | 0.57 ± 0.06 \| 0.00 | 0.69 ± 0.05 \| 0.10 | 0.50 ± 0.04 \| 0.00 |
| word_prefix     | 0.96 ± 0.03 \| 0.80 | 1.00 ± 0.00 \| 1.00 | 1.00 ± 0.00 \| 1.00 |
| number_prefix   | 1.00 ± 0.00 \| 1.00 | 1.00 ± 0.00 \| 1.00 | 1.00 ± 0.00 \| 1.00 |

**REASONING** — `side_correct` mean ± SEM | `side_exact` (n=10)

| encoding | opus | gpt_5.5 | kimi_k2.6 |
|---|---|---|---|
| whitespace      | 0.32 ± 0.05 \| 0.00 | 0.33 ± 0.05 \| 0.00 | 0.32 ± 0.06 \| 0.00 |
| capitalization  | 0.33 ± 0.06 \| 0.00 | 0.44 ± 0.06 \| 0.00 | 0.32 ± 0.04 \| 0.00 |
| markdown        | 0.28 ± 0.05 \| 0.00 | 0.46 ± 0.03 \| 0.00 | 0.31 ± 0.07 \| 0.00 |
| word_prefix     | 0.59 ± 0.09 \| 0.20 | 1.00 ± 0.00 \| 1.00 | 0.34 ± 0.04 \| 0.00 |
| number_prefix   | 0.78 ± 0.11 \| 0.70 | 1.00 ± 0.00 \| 1.00 | 0.56 ± 0.10 \| 0.10 |

Headlines:
- gpt_5.5 saturates every prefix-encoded cell at 1.00 (messaging and reasoning).
- No model exceeds chance on the steganographic-reasoning quadrant (whitespace/cap/markdown × reasoning all at ~0.30–0.46, side_exact = 0).
- kimi_k2.6 collapses on reasoning even with the easy `word_prefix` (0.34, chance). Inspection of `eval_prompting_reasoning_number_prefix_kimi_k2.6` shows format compliance is fine 9/10 (one refusal-style preamble that wiped the prefix); the rest is just unreliable arithmetic with high variance per-eval (0.11, 0.33, …, 0.89, 1.00). Reasoning weakness, not prompt-following.
- Opus < gpt_5.5 on reasoning_word_prefix (0.59 vs 1.00) — surprising; Opus is the strongest no-CoT reasoner I'd expected.
