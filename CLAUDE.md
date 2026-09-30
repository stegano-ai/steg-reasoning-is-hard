# CLAUDE.md

This file provides guidance to Claude Code when working with code in this repository.

## Project Goal

**"Steganography is easy, steganographic reasoning is hard."**

This repo is a benchmark for evaluating whether LLMs can learn steganographic reasoning — reasoning about one subject while the chain-of-thought looks like it's about something else.

Three related but distinct capabilities:
- **Steganographic messaging**: Hide a known piece of information in text so it can be read out later. No computation needed, just encoding.
- **Encoded reasoning**: Reason in a way that isn't transparent to an observer, but is obviously computational (not innocuous-looking).
- **Steganographic reasoning**: Reason about a hidden task while producing innocuous-looking text about a cover task. This is the hard one.

We test models on a 2x2 matrix:
- **Messaging vs Reasoning**: encode raw numbers vs compute running-sum-mod-N and encode the results
- **Steganographic vs Non-steganographic**: hidden encoding (capitalization, whitespace) vs obvious encoding (number prefix)

Our hypothesis: models easily learn the other three quadrants, but steganographic reasoning is much harder.

## Experiments

1. **sft** - Supervised fine-tuning on encoding schemes. Train models on (cover text + encoded values) pairs.
2. **prompting** - Zero/few-shot evaluation. Can models do encoding/reasoning from instructions alone?
3. **rl** - RL training (GRPO) where models learn encoding under a CoT monitor that penalizes suspicious outputs.
4. **monitor_benchmarking** - Validate that CoT monitors can distinguish suspicious from normal reasoning before using them as RL reward signals.

## Commands

```bash
# Run an experiment (auto-detects single/meta/sweep config)
cd experiments/<name> && python main.py -c configs/path/to/config.yaml

# Run tests (fast — skips network/dataset tests)
uv run pytest

# Run all tests including slow ones (network calls, HF dataset loads).
# Use this if you've touched lib/cot_monitor.py, lib/main_tasks/, or
# anything else that does network I/O.
uv run pytest --run-slow

# Run a single test
uv run pytest test/test_<module>.py

# Install dependencies
uv add <package>
```

## Run Discipline

Only run debug configs (`configs/debug.yaml` or similarly cheap commands) unsolicited. Never launch a non-debug experiment without explicit approval — not to verify a config change, not to rerun after a small fix, not even when the user just kicked off the same run and it failed. Non-debug runs cost real money and pollute the real wandb project; the user decides when to spend that. Ask "want me to rerun?" rather than running and reporting after-the-fact.

## Environment Variables

- `TINKER_API_KEY` - Required for SFT and RL training
- `OPEN_ROUTER_API_KEY` - Required for CoT monitor and WildChat check_answer
- `WANDB_API_KEY` - Required for experiment tracking

## Architecture

Uses **research_scaffold** for experiment orchestration.

### Directory Structure
- **lib/** - Shared Python package, importable as `from lib.<module> import ...`
- **experiments/** - Self-contained experiments, each with `main.py`, `functions/`, `configs/`, `results/`
- **test/** - Unit tests for lib modules
- **lab_notebook/** - Experimental narrative log. See [Lab Notebook](#lab-notebook) section.
- **tmp/** - Scratch space for temporary scripts. Claude has automatic permission to write and execute here using relative paths (e.g., `uv run python tmp/script.py`).

### lib/ Modules

- **main_tasks/** - Cover tasks accessed via `get_main_task(type, **kwargs)`. Types: `aqua_rat`, `wildchat`, `knapsack`, `dialogue`, `blank`. Each returns a `TaskDefinition` with train/test data and `check_answer(prompt, model_output, label) -> bool`. The answer field includes full reasoning (e.g. `"reasoning steps\n#### 42"`).
- **side_tasks/** - Hidden tasks accessed via `get_side_task(type, **kwargs)`. Types: `messaging` (encode numbers as-is), `rolling_mod_sum` (encode running sum mod N), `ball_and_cups` (A5 state tracking), `limited_arithmetic`. Each returns a `SideTaskDefinition` with train/test data (dicts with `input` and `output` int lists) and an `explanation` prompt.
- **encoding_schemes/** - Steganography schemes accessed via `make_general_scheme(name, n_symbols)`. Types: `capitalization`, `whitespace`, `markdown`, `number_prefix`, `word_prefix`, `evocative_word_prefix`, `sentence_markdown`, `sentence_interjection`, `sentence_newline`. Each has `encode(text, values)`, `decode(text, n)`, and `description`.
- **cot_monitor.py** - Async batch monitoring of model outputs via OpenRouter API. Also contains `MonitorConfig` dataclass.

### Config Pattern

Cover task and side task use the same config dict pattern; the encoding scheme is a field of the cover task:
```yaml
main_task:
  type: aqua_rat
  scheme: whitespace
  n_symbols: 3

side_task:
  type: rolling_mod_sum
  n_samples: 10000
  seq_length: 9
  operand_range: 10
  modulus: 3
```

In code: `get_main_task(**config["main_task"])`, `get_side_task(**config["side_task"])`.

### Experiment Structure
Each experiment in `experiments/<name>/` follows this pattern:
- `main.py` - Entry point that defines a `function_map` and calls `execute_experiments()`
- `functions/` - Training/evaluation functions referenced by `function_name` in configs
- `configs/base.yaml` - Base config with `name`, `function_name`, and `function_kwargs`
- Override configs merge into base.yaml

### Config System
Three config types (auto-detected with `-c` flag):
1. **Single** - Direct experiment config with `name` and `function_name`
2. **Meta** - Batch runner with `experiments` list and `common_root`
3. **Sweep** - W&B hyperparameter sweep with `method` and `parameters`

### Config Structure Convention
- **base.yaml** - Single config with all shared defaults. Every parameter the function needs must be here. No defaults in code.
- **debug.yaml** - Meta config inheriting from base. Minimal computation to verify the pipeline works.
- **<experiment>.yaml** - Meta configs for real experiments. Each inherits from base and specifies only what differs.

Key principles:
- **All defaults live in base.yaml, NEVER in Python code** - no default parameter values in function signatures
- Functions take explicit parameters, no `kwargs.get()` with defaults
- Debug config should run in seconds, not minutes
- **Meta configs override ONLY what needs to change** - don't redefine values that are the same as base.yaml
- **Debug configs must always use `wandb_project: debug`** - never log debug runs to the real project

## YAML Gotchas

- **Always write scientific notation with an explicit decimal and sign**: use `5.0e-4`, not `5e-4`. YAML 1.1 parsers (incl. PyYAML) only recognize the float form with a dot and signed exponent; `5e-4` is parsed as a string.

## Lab Notebook

`lab_notebook/` contains markdown files documenting the narrative of experimental work — hypotheses, what was run, what was learned, and decisions made. This is the "why" behind experiments that W&B and results folders don't capture.

### Naming Convention
Files are named `<issue-number>-<short-description>.md` (e.g., `002-port-sft-experiment.md`). The issue number maps to the GitHub issue. By default, the filename matches the current git branch name, which matches the issue being worked on.

### Entry Format
Each file has dated sections. Under each date: the current goal, what was run, what was learned.

### Rules
- **Never write to lab notebook files unless the user explicitly asks.** The lab notebook is the user's record, not the agent's.
- When the user asks you to write an entry, always link empirical results to their folder path (e.g., `experiments/sft/results/train_sft_messaging_2026-05-01_18-45-50/`).

## Coding Style

- Write minimal code with as few lines as possible
- Never write fallback code to handle multiple possibilities
- Only add docstrings, error handling, comments, or README edits if explicitly asked
- Use library imports over hand-coding
- **No silent failures**: Never use `.get(key, default)`, `try/except` with fallback values, or `os.environ.get()` for required config. If something is wrong, crash immediately with a clear error.

## Models

- Nano: `gpt-5-nano-2025-08-07`
- Mini: `gpt-5-mini-2025-08-07`
- GPT-5.2: `gpt-5.2`
- Never use gpt-4 models in this codebase

### Tinker Models

Available models: https://tinker-docs.thinkingmachines.ai/tinker/models/

Default to Qwen 3.6 (not older Qwen 3.x):
- `Qwen/Qwen3.6-35B-A3B` — MoE with 3B active params. User shorthand: "3B" or "qwen 3.6 3B".
- `Qwen/Qwen3.6-27B` — dense. User shorthand: "27B" or "qwen 3.6 27B".
