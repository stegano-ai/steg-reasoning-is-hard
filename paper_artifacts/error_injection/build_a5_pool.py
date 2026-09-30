# Rebuilds the 200-sample eval pool of the a5ws error-injection cells (pool_size 200,
# seed 0, whitespace/5 on AQuA-RAT, ball_and_cups 10000x9) with the experiment repo's
# own dataset code, so extract_data.py can replay each sample's instruction sequence.
# Run from the experiment repo: uv run python <this file>   (it is not stdlib-only).
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_REPO = os.path.normpath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, EXPERIMENT_REPO)
from lib.main_tasks import get_main_task  # noqa: E402
from lib.side_tasks import get_side_task  # noqa: E402
from lib.encoded_dataset import EncodedDataset  # noqa: E402

main = get_main_task(type="aqua_rat", scheme="whitespace", n_symbols=5)
side = get_side_task(type="ball_and_cups", n_samples=10000, seq_length=9)
pool = EncodedDataset(main, side, min_prefix=True).test(limit=200)
random.Random(0).shuffle(pool)
out = [{"instructions": list(s.side_input), "expected_reasoning": list(s.expected_reasoning)} for s in pool]
json.dump(out, open(os.path.join(HERE, "a5_pool.json"), "w"))
print("wrote", len(out), "samples")
