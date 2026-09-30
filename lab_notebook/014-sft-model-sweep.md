# 014 — SFT model sweep

## Goal

Test whether the paper's SFT result — encoded reasoning and steg messaging are easy to
train, steg reasoning is ~10x harder — generalizes beyond Qwen3.6-35B-A3B. Sweep the
three conditions over cheap Tinker models (Nemotron-3-Super-120B, gpt-oss-120b,
Inkling-Small, Nemotron-3.5-Lightning-30B) and frontier models (Kimi-K2.6, Inkling),
using early stopping so messaging/encoded runs stop once they work, then steg reasoning
gets 2-3x that budget to show it does *not* work in the same budget.

## 2026-08-13 — Infrastructure

Ported from steg-transfer-learning: cost tracking (`lib/cost_estimate.py`,
`experiments/sft/functions/costing.py`, `-p` price-gate in `main.py`, `token_stats.json`
written at dataset build, in-run COST ceiling print) and early stopping
(`early_stop_metric/threshold/min_steps` config keys, `EarlyStopEvaluator` raising
`ThresholdReached`). Tests in `test/test_cost_estimate.py`, `test/test_cost_gate.py`.

Cost findings that reshaped defaults in `experiments/sft/configs/base.yaml`:
- At paper settings a messaging run ceiling was $61, of which **$48 was NLL eval**
  (3 x n_test datums bill at the training rate every `eval_every`). Reduced
  `n_test` 1000→200 and `eval_every` 50→250 → $16/run.
- Dependency changes for the new models: tinker-cookbook 0.3.1→0.5.5 (`recipe_name` now
  required by `train.Config`), torch 2.9.1→2.10.0 (tml-renderers stable-ABI),
  `lib/inkling_renderer.py` registers `tml_v0_no_thinking` (+ custom prefill support).

Per-model LRs scale the tuned Qwen anchor (1.6e-4, hidden 2048) by 2048/hidden_size:
Nemotron-Super 8.0e-5, gpt-oss 1.138e-4, Inkling-Small 8.0e-5, Lightning 1.219e-4,
Kimi-K2.6 4.571e-5. (The Inkling values later proved wrong — see the TML section.)
Renderers: each model's thinking-disabled variant. Configs under
`experiments/sft/configs/model_sweep/`. Debug runs for all six models passed.

## 2026-08-13/14 — Mod-3 sweep (markdown + running-sum-mod-3, mirroring the paper runs)

Early stop at `eval/reasoning_correct >= 0.95`, 40 eval prompts, batch 16.

| Model | encoded (word_prefix) | messaging (markdown, min_prefix) | steg reasoning @2x | @10x |
|---|---|---|---|---|
| Nemotron-3-Super | 1.00 @ 100 | 0.96 @ 700 | 0.37 / exact 0.0 @ 1400 | **0.969 / 0.85 @ cum 6800 = 9.7x** |
| gpt-oss-120b | 0.975 @ 200 | 0.95 @ 1000 | 0.36 / 0.0 @ 2000 | 0.39 / 0.0 @ cum 10000 — never |
| Lightning-30B | 0.99 @ 200 | 0.95 @ 2100 | 0.40 / 0.0 @ 4200 | 0.52 / 0.0 @ cum 21000 — never |
| Inkling-Small | 0.95 @ 200 | see TML section | 0.90 / 0.75 @ 2200 = ~2x, mid-transition | **0.961 / 0.90 @ cum 3000 = 2.7x** (continuation, 2026-08-16) |

- 2x runs: `experiments/sft/results/model_sweep/steg_reasoning/` (2026-08-13 dirs);
  10x continuations resumed the 2x final checkpoints (weights only):
  [nemotron cont10x](../experiments/sft/results/model_sweep/steg_reasoning/train_sft_steg_reasoning_nemotron3_super_cont10x_2026-08-14_05-58-56),
  [gptoss cont10x](../experiments/sft/results/model_sweep/steg_reasoning/train_sft_steg_reasoning_gptoss_120b_cont10x_2026-08-14_05-58-56),
  [lightning cont10x](../experiments/sft/results/model_sweep/steg_reasoning/train_sft_steg_reasoning_nemotron35_lightning_cont10x_2026-08-14_05-58-56).
- Nemotron-Super reproduces the Qwen story almost exactly: ratio 9.7x at the 95% level
  (13-14x at the 80/90% levels), with the sharp exact-match phase transition
  (0.0 until ~cum 5000, then 0.85).
- gpt-oss and Lightning do NOT learn it even at 10x (Lightning drifts 0.40→0.52; gpt-oss flat).

Frontier phase 1: Kimi-K2.6 encoded **1.00 @ 100**, messaging 0.969 / exact 0.875 at its
2500 cap ([run](../experiments/sft/results/model_sweep/steg_messaging/train_sft_steg_messaging_kimi_k26_2026-08-14_06-21-23));
Inkling encoded 0.975 @ 200, messaging 0.961 @ 1000 (2026-08-15). Kimi 2x reasoning not
run (deferred). Frontier Inkling 2x steg reasoning (2026-08-16): **0.358 / exact 0.0 at
the 2000-step cap — chance-flat throughout**
([run](../experiments/sft/results/model_sweep/steg_reasoning/train_sft_steg_reasoning_inkling_2026-08-16_18-13-10)).
Striking within-family contrast: Inkling-Small crossed at 2.7x, the AA-41 flagship shows
nothing at 2x — capability rank does not predict the steg-reasoning breakthrough multiple
even within one family. Nemotron-3-Ultra was prepared (debug passed, lr 4.0e-5) but cancelled
~$4 in; configs remain in `configs/model_sweep/singles/`.

### The TML/Inkling saga

Inkling-Small messaging failed identically at three LRs (8e-5, 4e-5, 1.6e-4): climb to
~0.6-0.8 then collapse from ~step 1400, capping at 0.42-0.51. Eval completions showed
corrupted turn structure: leaked `<|message_model|><|content_text|>` headers and
unstoppable `#### D` repetition. Root cause: our min_prefix datums train **unclosed**
prefill-style targets; TML models degrade under thousands of never-closed-message steps
(other families tolerate it). The steg-transfer-learning repo's successful Inkling run
(branch `transfer-learning`) used **closed** messages and lr 1.6e-4 unscaled.

Fix: `closed_prefix_targets` option in `train_sft.py` (min_prefix covers built as closed
assistant messages) + unscaled lr. Fourth attempt converged cleanly:
0.953 / exact 0.875 @ 1100
([run](../experiments/sft/results/model_sweep/steg_messaging/train_sft_steg_messaging_inkling_small_closed_2026-08-14_15-17-18)).
Also learned: tinker_cookbook's `get_lr` has calibrated formulas only for Llama/Qwen —
none of our six models — and the Qwen hidden-size exponent is near-flat (0.0775), i.e.
our 1/hidden downscaling was much steeper than any calibrated family.

## 2026-08-15 — a5ws variant (ball-and-cups task + whitespace-5 scheme), cheap tier

Same protocol, side task → ball_and_cups (5 states, 9 instructions), scheme →
whitespace n_symbols 5. Debug: whitespace-5 roundtrips 48/48 on all four tokenizers.
Configs `configs/model_sweep/*_a5ws.yaml`.

| Model | encoded | messaging | steg reasoning @2x |
|---|---|---|---|
| Nemotron-3-Super | 1.00 @ 100 | 0.989 @ 500 | 0.589 / 0.125 @ 1000 — climbing |
| gpt-oss-120b | 0.95 @ 100 | 0.975 @ 800 | **0.931 / 0.825 @ 1600 — near-cross** |
| Inkling-Small | 0.975 @ 100 | 0.958 @ 300 | 0.669 / 0.125 @ 600 — climbing |
| Lightning-30B | 1.00 @ 100 | 0.95 @ 1900 | 0.361 / 0.075 @ 3800 — chance-flat |
| Qwen3.6-35B *(added 2026-08-17)* | 1.00 @ 100 | 0.986 @ 700 | 0.397 / 0.075 @ 1400 — climbing, no cross |

Results in `experiments/sft/results/model_sweep/{encoded_reasoning,steg_messaging,steg_reasoning}_a5ws/`.

**Key finding of the sweep so far:** whether a model learns steg reasoning quickly is a
model x task/scheme interaction, not a model property. gpt-oss was chance-flat at 10x on
mod-3 but nearly converged at 2x on a5ws; Inkling-Small the reverse (fast on mod-3,
middling on a5ws); Lightning fails both; Nemotron-Super slow on both. Where it works,
cost is ~2-10x messaging with a sharp exact-match transition.

Operational notes: total spend ~$370 across ~45 runs (early stopping + the cost gate
kept it far under naive ceilings). The 5GB root overlay filled twice — run logs now go
to `tmp/runlogs/` on /workspace, never /tmp.

## 2026-08-16 — min_prefix concern → "base SFT" (no-min-prefix) direction

Concern: min_prefix truncation concentrates all loss on value-carrying tokens — an
implicit curriculum ("handholding") that may explain the noisy 2x-10x spread; the paper's
claim is about *normal SFT*. Note encoded reasoning never used min_prefix, and Qwen
messaging without min_prefix did converge (0.98 @ ~2400 vs ~1200-1500 truncated,
[run](../experiments/sft/results/steg_messaging/train_sft_steg_messaging_markdown_2026-07-30_18-43-42)),
so only messaging + steg reasoning cells change.

Plan (gated, ~$450 budget): (1) no-min-prefix messaging for the two fast-reasoning
models (Inkling-Small on mod-3, gpt-oss on a5ws) — if messaging itself slows ~3x, slower
reasoning wouldn't count as evidence; (2) if messaging is fine, their no-min-prefix
reasoning runs; (3) if those look good, the remaining models' base messaging then base
reasoning; 10x base runs deferred. Hypothesis: without the truncation crutch, ~10x
everywhere.

### Outcome: ABANDONED same day (~$150 spent) — no clean factor; appendix material at most

What ran before the direction was killed (results dirs under
`experiments/sft/results/model_sweep/{steg_messaging,steg_reasoning}{,_a5ws}_base/`):

- Base messaging slows a modest, consistent **1.2-2.3x** vs min_prefix (well under the
  3x disqualifier): gpt-oss a5ws 0.969@1200 (1.5x), lightning a5ws 0.956@2400 (1.26x),
  inkling mod-3 0.897 cap@2500 (2.3x), lightning mod-3 ~0.88 cap@2500 (1.2x),
  nemotron mod-3 0.928 near-cross when killed. Bonus: full covers need no TML
  special-casing (inkling converged with zero turn-structure pathology), further
  confirming the closed-message diagnosis.
- Base steg reasoning did NOT produce the hoped-for clean "~10x everywhere":
  **gpt-oss a5ws crossed at 1000 steps — FASTER than its min_prefix twin** (0.983 vs
  0.93@1600 capped), directly refuting the handholding hypothesis for that cell; while
  inkling mod-3 lagged its twin (0.56@2700 vs ~0.75 at matched steps, killed mid-run).
  I.e. removing min_prefix moves different cells in different directions — the
  model x task interaction stays noisy, it is not a truncation artifact.

Decision: the min_prefix protocol stays the paper's main setup; this ablation is
documented here (and worth at most an appendix mention as "truncated vs full-cover
targets change convergence by <~2x in either direction and do not explain the
model-dependence"). Incidental fixes kept: RendererError-tolerant eval scoring in
`GenerationEvaluator` (a nemotron full-cover eval produced an unparseable completion
that killed a run deterministically at temp 0).

## 2026-08-17 — Baseline-model a5ws row + knapsack messaging appendix cell

- Qwen3.6-35B was missing from the a5ws grid (the variant sweep covered only the six new
  models). Filled in under the current protocol: encoded 1.00 @ 100, messaging 0.986 @
  700; steg reasoning capped at 2x with 0.397 / exact 0.075 — climbing but no cross.
  Contrast with the July markdown-a5 result (~99% in ~1200 steps): same model and task,
  different scheme — another instance of scheme-dependence.
- Knapsack appendix: the kept knapsack steg-REASONING run
  ([knapsack_summod3 2026-07-01](../experiments/sft/results/steg_reasoning/train_sft_steg_reasoning_knapsack_summod3_2026-07-01_19-40-48),
  Qwen3.5-397B, binary_2 knapsack covers, min_prefix false) needed ~50-100 steps; its
  exact messaging analog (side task -> raw messaging, vocab 3, all else identical) hits
  **1.00 correct / 1.00 exact at step 25** — the first eval —
  ([run](../experiments/sft/results/steg_messaging/train_sft_steg_messaging_knapsack_2026-08-17_12-09-15)).
  Even in the easiest regime found anywhere in this project, reasoning costs a multiple
  of messaging.

### Error-injection / swap checks for the sweep's crossers (2026-08-17; CORRECTED 2026-08-21)

Added the two model-sweep crossers to the `steg_reasoning_check` harness
(`experiments/prompting/configs/steg_reasoning_check/`; crossing state checkpoints
converted to sampler weights; results content-addressed under
`experiments/prompting/results/steg_reasoning_check/eval_sft_{nemotron_super,inkling_small}_*`).
n_eval 20/arm (errinj: 20 per rotated position = 180 rows/model), temp 1.0.

NOTE: the first committed version of this section (be3e3f4) had the errinj columns
mislabeled; corrected here. The harness scores errinj rows against the ORIGINAL
sequence; per-row injected delta = actual[error_position] - expected[error_position].
Post-error positions decompose into "ignores" (matches original = recomputed from
input) vs "tracks" (matches original+delta = read the corrupted trace and propagated):

| Model (mod-3 crosser) | normal rc / exact | IGNORES error (recomputes) | TRACKS error (reads trace) | neither |
|---|---|---|---|---|
| Qwen (paper ckpt, ~10x) | 0.917 / 0.60 | **0.52** | 0.36 | 0.11 |
| Nemotron-Super (9.7x) | 0.906 / 0.75 | **0.43** | 0.28 | 0.29 |
| Inkling-Small (2.7x) | 0.939 / 0.75 | 0.24 | **0.53** | 0.24 |

(3-way chance = 0.33 per column.)

Reading: the mechanism is heterogeneous and correlates with breakthrough speed. The two
slow (~10x) learners predominantly recompute each value from the prompt's input and
ignore a corrupted trace; the fast (2.7x) learner Inkling-Small predominantly READS its
own trace and propagates the injected error — a genuinely sequential, trace-dependent
algorithm, despite no training pressure to read the trace (targets never included a
`#### <final state>` commitment; output_format is 0.0 everywhere, and the swap arm is
therefore vacuous for these models). A sharper version of this test would retrain with
an answer-committing target (`#### <final state>` appended), making errinj outcome-level
rather than trace-level.

## 2026-08-21 — a5ws 10x continuations + error injection on the a5ws crossers

10x continuations of the a5ws (ball-and-cups × whitespace-5) steg-reasoning runs
crossed for three of four models: gptoss cum 1900 = **2.4x**, Inkling-Small cum 900 =
**3.0x**, Nemotron-Super cum 2300 = **4.6x**. Lightning: **NO CROSS at 10x** — capped
at cum 19000 with 0.500 correct / 0.075 exact
([run](../experiments/sft/results/model_sweep/steg_reasoning_a5ws/train_sft_steg_reasoning_a5ws_nemotron35_lightning_cont10x_2026-08-21_11-00-56)).
Not chance-flat: it climbed off chance (~0.36) to a ~0.5 plateau by ~4.7x and sat there
through the cap, exact never above 0.1 — the same partial-learner signature as its
mod-3 10x run (0.519 / 0.0). Lightning is a two-variant negative; the errinj cells
below therefore cover only the three crossers.

Ran the `steg_reasoning_check` harness on the three crossers' crossing checkpoints
(early-stop states converted to sampler weights, `errinj_a5ws_*_cross`; cells under
`experiments/prompting/results/steg_reasoning_check/eval_sft_a5ws_*`). Same protocol
as mod-3: n_eval 20/arm, errinj rotate = 180 rows/model, temp 1.0, injected value =
(true + 1) mod 5.

Decomposition semantics differ from mod-3: ball-and-cups errors don't propagate as a
constant additive delta. The "tracks" reference is computed by replaying the sample's
true instruction sequence (Rotate / Double-Swap) from the injected wrong state
(`tmp/analyze_errinj_a5ws.py`). Both instructions are permutations, so the wrong-state
trajectory never re-merges with the original — every post-injection position cleanly
discriminates tracks vs ignores.

| Model (a5ws crosser) | normal rc / exact | IGNORES error (recomputes) | TRACKS error (reads trace) | neither |
|---|---|---|---|---|
| gptoss (2.4x) | 0.944 / 0.85 | 0.16 | **0.42** | 0.43 |
| Nemotron-Super (4.6x) | 0.811 / 0.40 | 0.30 | 0.33 | 0.37 |
| Inkling-Small (3.0x) | 0.978 / 0.95 | 0.17 | **0.57** | 0.26 |

Row-level (all 8 post positions consistent): gptoss 44 tracked / 8 original,
Nemotron-Super 40 / 16, Inkling-Small 80 / 14 (of 180; the rest mixed, flip-missing,
or position-8 injections with no post positions). Chance for any single agreement
column is 0.2 (5 states); "neither" also absorbs baseline decode errors — visible in
Nemotron-Super's weak normal arm (exact 0.40).

Reading: on the NC1-complete task, **tracking dominates wherever the model actually
learned the task** — gptoss and Inkling-Small, the two strong normal-arm models, both
read their trace and propagate the injected state (0.42 / 0.57 vs ignore 0.16 / 0.17).
This inverts the mod-3 picture for the slow learners: with running-sum-mod-3, each
value is recomputable from the prompt's prefix cheaply, and slow learners recompute;
ball position has no such shortcut (A5 word problem), so genuine learners must — and
do — carry state through the trace. Inkling-Small tracks on both tasks. Nemotron-Super
is split, consistent with its noisier decode. As with mod-3, targets never included an
answer commitment, so the swap arm stays vacuous (output_format 0.0).
