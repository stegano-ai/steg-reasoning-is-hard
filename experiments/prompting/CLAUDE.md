# Prompting experiment

## wandb is OFF for all prompting experiments

Every prompting config — real, debug, sweep, everything — must set `wandb_project: null`
(both the top-level key and the one under `function_kwargs`). This OVERRIDES the global
rule that debug configs use `wandb_project: debug`: here debug runs also stay off wandb.
Metrics/summaries are written to the run's results folder (`summary.json`,
`eval_completions.jsonl`); we don't log prompting runs to wandb at all.

## Models that cannot be evaluated here

These experiments disable model reasoning — `OpenRouterSampler` sends
`extra_body={"reasoning": {"enabled": false}}` (see `lib/sampling.py`) — so the
model cannot solve the side task in a hidden reasoning scratchpad. The point is
to test whether it can encode/reason in the visible response, so reasoning must
stay off.

Reasoning-mandatory models reject `{"enabled": false}` — they either return
`400 - "Reasoning is mandatory for this endpoint and cannot be disabled."` or hang.
Some of them accept a minimal reasoning effort that produces **0 reasoning tokens**
in practice (the model reasons in the visible response, which is what we want). The
sampler takes a per-model `reasoning` override for this: set
`sampler: {kind: openrouter, model: ..., reasoning: {effort: minimal}}` in the axis
entry (default is `{"enabled": false}`).

Usable only with `reasoning: {effort: minimal}`:
- `google/gemini-3.5-flash` — mandates reasoning (rejects `{"enabled": false}`), but
  effort:minimal reliably produces 0 reasoning tokens (verified, incl. on the hard side
  tasks) — the model reasons in the visible response, which is what we want.

Known-incompatible (no workaround found — do not add to the sweeps):
- `anthropic/claude-fable-5` — 400, reasoning cannot be disabled
- `google/gemini-3.1-pro-preview` — hangs / reasoning mandatory

## Claude (opus/sonnet) is banned from the steganography sweeps

`anthropic/claude-opus-4.7` and `anthropic/claude-sonnet-5` are **excluded** from the
frontier sweeps. Their safety filter refuses the covert-encoding task: responses come back
`finish_reason: "content_filter"` at ~90–100% regardless of prompt phrasing, few-shot count
(0/1/2-shot all high), or provider (tested Anthropic/Bedrock/Google/Azure — all filter).
The rate is also time-variable, so it can't be reliably prompt-engineered around. The
`ContentFilterError` handling in `lib/sampling.py` drops any that slip through, but at that
rate there's no usable signal — so we exclude Claude entirely rather than run near-empty cells.

This is itself a finding worth reporting: Claude will not perform (or be measured on)
steganographic encoding — its safety training blocks it. Separately, when Claude *does*
respond it reasons in **plaintext** (writes the full working in the clear → suspiciousness ~9),
so even unfiltered it doesn't do the task covertly.
