# 005 — Knapsack-coupling investigation
Branch: 005-knapsack-investigation
Owner: Julian

## 2026-05-07 — knapsack `column_a`, closed models

`experiments/prompting_combined/results/main_2026-05-07_12-45-02/`

n_eval=20, n_shot=3, n_candidates=9, side_task=rolling_mod_sum (operand_range=2, modulus=2), same/change rendering, `reasoning.enabled=false`.

| model | side_correct | side_exact |
|---|---|---|
| gpt-5.5 | 1.000 | 1.000 |
| opus-4.7 | 0.783 | 0.200 |
| sonnet-4.6 | 0.772 | 0.300 |
| qwen3-235b-a22b-2507 | 0.644 | 0.050 |
| kimi-k2.6 | 0.622 | 0.050 |
| qwen3-30b-a3b-instruct-2507 | 0.506 | 0.000 |

models are capable of encoding thier reasoning in this simple side task using the knapstak encoding
## 2026-05-07 — knapsack `column_a`, open models

`experiments/prompting_combined/results/open_models_2026-05-07_12-46-07/`

Same setup as above.

| model | side_correct | side_exact |
|---|---|---|
| qwen3.5-397b-a17b | 0.822 | 0.600 |
| qwen3.6-35b-a3b | 0.639 | 0.250 |
| qwen3.6-27b | 0.589 | 0.000 |
| llama-3.3-70b-instruct | 0.456 | 0.000 |
quen 3.6 is also capable of doing this, wich would be a great model to start RL with, espsecially given it's decen size of 3b activa paramets
## 2026-05-07 — knapsack `binary_2` × `limited_arithmetic`

`experiments/prompting_combined/results/limited_arithmetic_2026-05-07_12-46-59/`

n_eval=20, n_shot=3, n_items=6, n_candidates=3, side_task=limited_arithmetic (num_steps=3, max_value=4), `reasoning.enabled=false`. Random baseline 1/4 = 0.25 per value.

| model | side_correct | side_exact |
|---|---|---|
| opus-4.6 | 1.000 | 1.000 |
| gpt-5.5 | 0.950 | 0.850 |
| qwen3.6-27b | 0.383 | 0.100 |

a bit of a more complicated setup: models do continuios calculation of small numbers encoding them in a binary way in each knapsack line. 
setup is more evocative, as model can also use this to do arbitrary calculation and encode intermediate result 

## 2026-05-07 — dialogue `speaker`, closed models

`experiments/prompting_combined/results/dialogue_main_2026-05-07_13-02-24/`

n_eval=20, n_shot=3, n_lines=10 (1 anchor + 9 data lines), corpora=[eng, am, shake], side_task=rolling_mod_sum (operand_range=2, modulus=2), same/change rendering, `reasoning.enabled=false`.

| model | side_correct | side_exact |
|---|---|---|
| gpt-5.5 | 0.894 | 0.800 |
| sonnet-4.6 | 0.656 | 0.050 |
| kimi-k2.6 | 0.606 | 0.050 |
| opus-4.7 | 0.583 | 0.050 |
| qwen3-235b-a22b-2507 | 0.567 | 0.000 |
| qwen3-30b-a3b-instruct-2507 | 0.489 | 0.000 |
dialogue format is also a valid encoding scheme for binary state trackingj
## 2026-05-07 — dialogue `speaker`, open models

`experiments/prompting_combined/results/dialogue_open_models_2026-05-07_13-04-14/`

Same setup as above.

| model | side_correct | side_exact |
|---|---|---|
| qwen3.6-35b-a3b | 0.628 | 0.100 |
| llama-3.3-70b-instruct | 0.594 | 0.000 |
| qwen3.5-397b-a17b | 0.561 | 0.000 |
| qwen3.6-27b | 0.544 | 0.050 |
and some open source models are capable of doing it. 
