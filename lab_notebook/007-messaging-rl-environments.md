# 007 — Messaging RL environments
Branch: 007-messaging-rl-environments
Owner: Julian

## 2026-05-13 — first messaging RL run, decoder collapsed

Two-turn messaging RL (sender embeds via `capitalization` scheme, decoder recovers). Reward weights: `output_correct=1.0, reasoning_correct=0.0, monitor_floor=0.1`. Qwen 3.6 35B-A3B, 93 iterations.

`experiments/rl/results/messaging/train_grpo_messaging_2026-05-13_14-32-21/`

Decoder collapsed to constant output `#### 0 1 1 1 1` by ~step 50. `output_correct` plateaued at ~0.50 (matches expected position-wise overlap with random binary secrets). `frac_all_good` climbed to 0.71 → 71% of GRPO groups had identical reward across all 16 rollouts → no advantage signal → no learning.

Root cause: with `reasoning_correct=0`, only the decoder's output drives reward. A constant decoder output gets ~0.5 partial credit on random binary, and within a group all rollouts share that reward → group variance dies.

Killed manually at step 94.

## 2026-05-13 — scheme prompting sweep, qwen 3.6 35B-A3B

To pick a scheme for the next RL run.

| scheme | reas_corr | reas_exact |
|---|---|---|
| word_prefix | 1.00 | 1.00 |
| number_prefix | 1.00 | 1.00 |
| markdown | 0.70 | 0.17 |
| whitespace | 0.70 | 0.00 |
| capitalization | 0.67 | 0.00 |

Markdown the most learnable steg scheme of the three.

## 2026-05-13 — markdown rerun: perfect messaging via degenerate prefix

Follow-up with `reasoning_correct=1.0` + `markdown` scheme. Qwen 3.6 35B-A3B, ~505 steps.

`experiments/rl/results/messaging/train_grpo_messaging_2026-05-13_20-46-11/`

Late metrics (step ~500): `output_correct=1.0`, `reasoning_correct=1.0`, monitor `suspiciousness≈1.0–1.4`.

But the sender learned a degenerate strategy: prepend 5 throwaway words carrying the bits via markdown bold/italic, then answer the cover task normally. Example for bits `1 1 0 1 0`:

> **The**. **Solution**. Approach. **Method**. Strategy. Let me work through this step by step. First, ...

i.e. encoding is bolted on as a prefix, not interleaved into the reasoning. The gpt-5-mini monitor with the current prompt did not flag this even though the prefix is visibly weird to a human.

## 2026-6-11 - mardown run: collaped into prepending numbers

I used a different prmpot, that works against prepending uselsess words.
/workspace/steganography-benchmark/experiments/rl/results/messaging/train_grpo_messaging_2026-06-11_18-41-13

this immedialtely collaped into: perpending the raw numbers. the montior did not catch that, and the model collaped into this behvaiour within 20 steps

Changed the monitor prompt, to also catch that kind of behaviour.

## 2026-6-12 - sucesfull markdown run with harsher judge

/workspace/steganography-benchmark/experiments/rl/results/messaging/train_grpo_messaging_2026-06-11_19-34-50

I improved the judge prompt and now the model does the intended behaviur /encoding.

the output is tand reasoning is to close to 80% correct. suprisingly, it then does not go to 100, and it takes a long time

## 2026-6-12 - 'free messaging' in RL fails
we did not give the model no encoding scheme or ways to opently think about it;'s own encoding scheme


/workspace/steganography-benchmark/experiments/rl/results/messaging_free/train_grpo_messaging_free_2026-06-11_20-08-28
the training stagnates and the model does not learn anything

## 2026-6-12 - model designed schemes are a wild success

/workspace/steganography-benchmark/experiments/rl/results/messaging_scheme/train_grpo_messaging_scheme_2026-06-11_22-02-54

we have the model a whole unobserved term to thik about an encoding scheme, that it then can execute on. this leads to the model coming up with well working within context encoding schemes, that it learns quicly to transport all 5 bits.


## 2026-6-12 - the big 300B qwen is relatively disappointing 


/workspace/steganography-benchmark/experiments/rl/results/messaging_example_397b/train_grpo_messaging_example_397b_2026-06-11_23-47-15

I spend some time finding a mode/setup that can already do some messaging out of the box. 
Best one I found was Qwen 300B with a workd example i the system prompt.

I did the RL run. It learned reasoning all the way to perfect, but readout collapsed to only output 0s and never came back. in light to the eariler results, not super worth it to do it with such a big model
