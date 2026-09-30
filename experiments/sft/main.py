from research_scaffold.config_tools import execute_experiments
from research_scaffold.argparsing import get_base_argparser, process_base_args
from functions import train_sft
from lib.cost_estimate import (approve_or_exit, format_grand_total,
                               format_run_cost, resolve_configs, run_total)

function_map = {
    "train_sft": train_sft,
}

if __name__ == "__main__":
    parser = get_base_argparser()
    parser.add_argument("-p", "--price", action="store_true",
                        help="print each run's measured cost ceiling and ask y/N before "
                             "anything starts; nothing is created until approved")
    args = parser.parse_args()
    config_path, meta_config_path, sweep_config_path = process_base_args(args)
    if args.price:
        # Sampled dataset builds in a temp dir — nothing exists yet: the scaffold has
        # opened no wandb run, no Tinker client, no results directory.
        from functions.costing import stage_estimates
        configs = resolve_configs(config_path or meta_config_path or sweep_config_path)
        if configs:
            per_run = {c.name: stage_estimates(c) for c in configs}
            print("\n\n".join(format_run_cost(name, cfg.function_kwargs["model_name"], ests)
                              for (name, ests), cfg in zip(per_run.items(), configs)))
            if len(per_run) > 1:
                print("\n" + format_grand_total({n: run_total(e) for n, e in per_run.items()}))
        else:
            print("Cost estimate unavailable: wandb sweeps have no fixed experiment list.")
        if not args.dry_run:
            approve_or_exit()
    execute_experiments(
        function_map=function_map,
        config_path=config_path,
        meta_config_path=meta_config_path,
        sweep_config_path=sweep_config_path,
        dry_run=args.dry_run,
    )
