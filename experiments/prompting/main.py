import asyncio
import sys
from pathlib import Path

import yaml
from research_scaffold.argparsing import get_base_argparser, process_base_args
from research_scaffold.config_tools import (
    detect_config_type,
    execute_experiments,
    get_time_stamp,
    load_meta_config,
    process_meta_config,
    resolve_run_names,
    substitute_placeholders,
)

from functions import generate
from functions.generate import _generate
import lib.sampling
import lib.inkling_renderer  # noqa: F401  registers tml_v0_no_thinking for TML samplers

# Sync entry points: single configs and W&B sweeps go through research_scaffold unchanged.
function_map = {"generate": generate}
# Async entry points: a meta-config's cells are driven directly (below) so the whole sweep
# shares one event loop, and thus the sampler's global concurrency cap (lib.sampling).
async_function_map = {"generate": _generate}


async def _run_meta_bounded(meta_config_path: str) -> None:
    """Run every cell of a prompting meta-config concurrently in a single event loop.

    research_scaffold would otherwise fork one unbounded process per cell; here the cells
    share one loop, so the sampler's global semaphore bounds requests across the entire
    sweep. Cells are isolated — one that raises is reported, not allowed to cancel the rest.
    """
    cells = process_meta_config(load_meta_config(meta_config_path))
    launch = get_time_stamp()
    total = len(cells)
    done = 0
    print(f"Running {total} cells in one event loop, capped at {lib.sampling.MAX_OPENROUTER_CONCURRENCY} "
          f"concurrent requests.", flush=True)

    async def run_cell(cell):
        nonlocal done
        names = resolve_run_names(
            name=cell.name,
            time_stamp_name=cell.time_stamp_name,
            time_stamp_group=cell.time_stamp_group,
            wandb_group=cell.wandb_group,
            launch_time_stamp=launch,
        )
        kwargs = substitute_placeholders(cell.function_kwargs, names)
        if cell.save_config_path:
            path = Path(substitute_placeholders(cell.save_config_path, names))
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(yaml.safe_dump(substitute_placeholders(cell.d, names), sort_keys=False))
        try:
            await async_function_map[cell.function_name](**kwargs)
            err = None
        except Exception as e:
            err = e
        done += 1
        tag = "ok" if err is None else f"FAILED {type(err).__name__}: {err}"
        print(f"[{done}/{total}] {cell.name}  {tag}", flush=True)
        return err

    errors = await asyncio.gather(*(run_cell(c) for c in cells))
    n_failed = sum(e is not None for e in errors)
    print(f"\nSweep complete: {total - n_failed}/{total} cells succeeded.", flush=True)


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)  # live output even when redirected to a file
    args = get_base_argparser().parse_args()
    config_path, meta_config_path, sweep_config_path = process_base_args(args)
    path = config_path or meta_config_path or sweep_config_path
    raw = yaml.safe_load(open(path)) if path else {}
    if path and detect_config_type(raw) == "meta" and not args.dry_run:
        if raw.get("max_concurrency"):  # spread requests when a provider rate-limits (e.g. Crusoe)
            lib.sampling.MAX_OPENROUTER_CONCURRENCY = raw["max_concurrency"]
        asyncio.run(_run_meta_bounded(path))
    else:
        execute_experiments(
            function_map=function_map,
            config_path=config_path,
            meta_config_path=meta_config_path,
            sweep_config_path=sweep_config_path,
            dry_run=args.dry_run,
        )
