# 002 — Port SFT experiment
Issue: #2
Owner: Julian

## 2026-05-04

Started making clean implementaion of steg/nonsteg messaging/reasoning for SFT. I start with a small training sweep usint the 30B qwen on messaging.

non steg messaging works within 100 steps perfectly
experiments/sft/results/nonsteg_messaging/train_sft_nonsteg_messaging_2026-05-04_14-44-58

nonste reasoning also works perfectly within the first 100 steps

experiments/sft/results/nonsteg_reasoning/train_sft_nonsteg_reasoning_2026-05-04_14-53-35

steg messaging makes some progress, the loss diff between trajectories and shuffled trajectories is just starting to go above 0, but we need more training data for this to actually work

experiments/sft/results/steg_messaging/train_sft_steg_messaging_2026-05-04_14-46-07

Steg reasoning, as expectd is not learned at all: loss diff is around 0
experiments/sft/results/steg_reasoning/train_sft_steg_reasoning_2026-05-04_14-53-38


Next step: redo the steg runs with twice as much data, to hopefully get a clean learning of steg messaging but not of steg reasoning 
experiments/sft/results/steg_messaging/train_sft_steg_messaging_2026-05-04_17-18-12
steg messaging is on it's way up. it got up to 60% accyray

experiments/sft/results/steg_reasoning/train_sft_steg_reasoning_2026-05-04_17-18-17
steg reasoning is still around 0.

To get a clearer result, I will have to mak the run longer any probably also increase the eval set, so you can more clearly follow the eval correctness development.

Doubled n_eval to 40 and doubled steg n_train again (12800 → 25600). Both steg quadrants relaunched:
- `experiments/sft/results/steg_messaging/train_sft_steg_messaging_2026-05-04_21-37-54/`
- `experiments/sft/results/steg_reasoning/train_sft_steg_reasoning_2026-05-04_21-37-54/`

## 2026-05-05

Steg messaging still not quite at perfect correctness. Bumped n_train to 40000 (≈ 2500 steps) on both steg quadrants for a cleaner curve:
- `experiments/sft/results/steg_messaging/train_sft_steg_messaging_2026-05-05_07-13-47/`
- `experiments/sft/results/steg_reasoning/train_sft_steg_reasoning_2026-05-05_07-13-47/`

Steg reasoning won't get there with more data alone. Want to try the minimal-prefix training (port of `encoded-reasoning/experiments/sft/functions/generate_aqua_short.py`): truncate the natural rationale to the shortest word-prefix that still roundtrip-encodes the values, so every encoding position is load-bearing. Implemented as `min_prefix: bool` in `train_sft.py`; new config at `experiments/sft/configs/steg_reasoning/truncated_reasoning.yaml`. First run:
- `experiments/sft/results/steg_reasoning/train_sft_steg_reasoning_2026-05-05_07-50-58/`
