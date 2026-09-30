# Learning Steganography Is Easy, Learning Steganographic Reasoning Is Hard

Code, data and lab notebooks accompanying the paper

> **Learning Steganography Is Easy, Learning Steganographic Reasoning Is Hard**
> Julian Schulz, Lukas Fülle, Rieke Fruengel
> NeurIPS 2026 Workshop AI4GOOD: Trustworthy AI for Good
> [arXiv:XXXX.XXXXX](https://arxiv.org/abs/XXXX.XXXXX)

## Abstract

Chain-of-thought monitoring as an approach for AI oversight and control is threatened by the possibility of steganographic reasoning, where LLMs conceal their reasoning inside innocuous-looking text.
Two neighbouring capabilities, steganographic messaging (passing a concealed message) and encoded reasoning (reasoning in an illegible but unconcealed format), have already been shown to emerge under training pressures that occur in real pipelines, such as reinforcement learning against monitors.
This suggests that steganographic reasoning too might arise as an unintended side effect of training.
Here, we compare how easily models learn steganographic reasoning and these two neighbouring capabilities across three elicitation methods: reinforcement learning, in-context learning, and supervised fine-tuning (SFT).

For most tasks, models learn steganographic reasoning only under SFT, while they learn steganographic messaging and encoded reasoning under all three elicitation methods.
Even under SFT, steganographic reasoning requires at least twice as much training as messaging, and for several model-task combinations it is not learned at all.
However, on a cover task that makes hiding information especially convenient, steganographic reasoning can be successfully learned under all three elicitation methods.
Steganographic reasoning is thus much harder than steganographic messaging and encoded reasoning, and learning the latter two does not imply learning the former.
Yet it lies within reach: an easy version is learned under every elicitation method, when the cover task is convenient for hiding information.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and Python ≥ 3.11.

```bash
uv sync
export OPEN_ROUTER_API_KEY=...   # prompted models, CoT monitor
export TINKER_API_KEY=...        # SFT and RL (https://thinkingmachines.ai/tinker)
export WANDB_API_KEY=...         # experiment tracking
uv run pytest                    # fast tests; --run-slow adds network and dataset tests
```

`.env` sets the Hugging Face `trust_remote_code` flags some tokenizers need. Load it with
[direnv](https://direnv.net/) (`.envrc`) or `set -a; source .env; set +a`.

## Running experiments

Each experiment in `experiments/` (`prompting`, `sft`, `rl`, `monitor_benchmarking`) is a
[research_scaffold](https://github.com/jr-brown/research-scaffold) entry point driven by YAML configs:

```bash
cd experiments/<experiment> && uv run python main.py -c configs/<path>.yaml
```

Every config has a cheap `debug` sibling; run that first. The non-debug configs call paid APIs
(OpenRouter, Tinker). Every run writes a folder under `experiments/<experiment>/results/` holding
the exact `config.yaml` it ran with, its metrics and its transcripts.

## Reproducing the figures and tables

`paper_artifacts/` has one folder per figure or table. Each folder has an `extract_data.py`, which
reads the committed results into `data.json`, and a plotting or table script. To regenerate all of them:

```bash
uv run --with playwright playwright install chromium   # Figure 1 is rendered from HTML
paper_artifacts/reproduce.sh
```

| Paper | `paper_artifacts/` | Results it reads (under `experiments/`) |
|---|---|---|
| Figure 1 | `figure_1/` | `prompting/results/n_shot_investigation/`, plus the data of Figure 3 and Figure S12 |
| Table 1 | — (written by hand) | scheme definitions in `lib/encoding_schemes/` |
| Figure 2 | `prompting_steg_comparison/pareto_simple.png` | `prompting/results_metrics/frontier_model_sweep.csv` |
| Figure 3 | `rl_comparison/rl_main_stacked.png` | `rl/results/{messaging,reasoning,reasoning_wordprefix_process}/`, `rl/results/knapsack_handholding/` |
| Table 2 | — (written by hand) | `sft/results/model_sweep/` (same runs as Figures S12–S13) |
| Table S1 | — (written by hand) | model identifiers in the configs under `experiments/*/configs/` |
| Table S2 | `monitor_validation/table.tex` | `monitor_benchmarking/results/` |
| Figure S1 | `prompting_steg_comparison/pareto.png` | `prompting/results_metrics/frontier_model_sweep.csv` |
| Figures S2, S3 | `prompting_steg_comparison/{a5,binary}.png` | `prompting/results_metrics/frontier_model_sweep.csv` |
| Figures S4, S5 | `side_task_performance/` | `prompting/results/{side_task_baseline,best_steg_reasoning}/`, `prompting/results_metrics/frontier_model_sweep.csv` |
| Figure S6 | `n_shot_investigation/` (a), `prompting_uplift/` (b) | `prompting/results/{n_shot_investigation,side_task_baseline,knapsack}/` |
| Figure S7 | `rl_comparison/rl_per_position.png` | `rl/results/{messaging,reasoning,reasoning_wordprefix_process}/` |
| Figure S8 | `rl_comparison/rl_suspiciousness.png` | `rl/results/{messaging,reasoning}/` |
| Figure S9 | `rl_knapsack/rl_knapsack_suspiciousness.png` | `rl/results/knapsack_handholding/` |
| Figure S10 | `rl_scheme_comparison/` | `rl/results/{messaging_scheme_replication,reasoning_model_scheme,reasoning_model_scheme_nocover}/` |
| Figure S11 | `rl_knapsack/rl_knapsack_outcome.png` | `rl/results/knapsack_handholding{,_onlyoutput}/` |
| Table S3 | — (written by hand) | the `config.yaml` in each RL run folder above |
| Figures S12, S13 | `sft_model_sweep/` | `sft/results/{model_sweep,steg_messaging,steg_reasoning}/` |
| Table S4 | — (written by hand) | the `config.yaml` in each SFT run folder above |
| Figure S14 | `sft_knapsack/` | `sft/results/{steg_messaging,steg_reasoning}/` (knapsack runs) |
| Figure S15 | `error_injection/` | `prompting/results/steg_reasoning_check/` |
| Figure S16 | `knapsack_sweep/` | `prompting/results/knapsack/` |
| Table S5 | — (written by hand) | `prompting/configs/paper_replication/` |
| Figure S17, Tables S6, S7 | `replication_highn/`, `replication_screen/` | `prompting/results_metrics/paper_replication.csv` |

Each `extract_data.py` names the exact run folders it reads at the top of the file.

## Repository layout

```
lib/                 shared code: cover tasks, side tasks, encoding schemes, monitor, scoring, sampling
experiments/
  prompting/         in-context learning: frontier model sweeps, knapsack, n-shot, Zolkowski et al. replication
  sft/               supervised fine-tuning on Tinker
  rl/                GRPO on Tinker against a CoT monitor
  monitor_benchmarking/  validating the CoT monitor before using it as a reward
paper_artifacts/     scripts that turn committed results into the paper's figures and tables
lab_notebook/        the research log, dead ends included
test/                unit tests
```

## Lab notebook and dead ends

`lab_notebook/` is the research log we kept while doing this work: hypotheses, what was run,
what went wrong and why each decision was made. Results folders that no figure reads are the
dead ends and preliminary runs it describes (for example the early RL reward variants in
`experiments/rl/results/train_grpo_2026-05-*`, and monitor prompt iterations v1–v5 in
`experiments/monitor_benchmarking/results/`). We kept them on purpose.

`CLAUDE.md` holds the conventions we gave our coding agent while doing this work. It also
works as a short guide to the code.

## Citation

```bibtex
@inproceedings{schulz2026steganographic,
  title     = {Learning Steganography Is Easy, Learning Steganographic Reasoning Is Hard},
  author    = {Schulz, Julian and F{\"u}lle, Lukas and Fruengel, Rieke},
  booktitle = {NeurIPS 2026 Workshop AI4GOOD: Trustworthy AI for Good},
  year      = {2026},
  url       = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

## License

MIT, see [LICENSE](LICENSE).
