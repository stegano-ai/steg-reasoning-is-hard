#!/bin/bash
# Regenerates every figure and table of the paper from the committed results.
set -e
cd "$(dirname "$0")"
uv run python error_injection/build_a5_pool.py
for d in */; do
  if [ -f "$d/extract_data.py" ] && [ "$d" != figure_1/ ]; then uv run python "$d/extract_data.py"; fi
done
uv run python figure_1/extract_data.py
for f in */plot_*.py */make_table.py; do uv run python "$f"; done
uv run --with playwright python figure_1/generate_png.py
