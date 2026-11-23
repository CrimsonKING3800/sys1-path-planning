import math
import random

import numpy as np
from openpyxl import Workbook

from map_gen import generate_map
from vlm_rrt import Environment2D, VLMRRTConfig, VLMRRTPlanner

SEEDS = range(1, 3)          # one map per seed
USE_MOCK = False               # False = call the real VLM
OUT_FILE = "vlm_rrt_results.xlsx"

config = VLMRRTConfig(
    gamma=0.85,
    prompt_mode="few_shot",
    model_name="nvidia/nemotron-3.5-super-vl-120b-a12b",
    disable_thinking=True,
)

wb = Workbook()
ws = wb.active
ws.title = "Results"
ws.append(["Seed", "Success", "Iterations", "Path length (m)"])

for seed in SEEDS:
    random.seed(seed)
    np.random.seed(seed)

    env = Environment2D(**generate_map(seed=seed))
    path, iters = VLMRRTPlanner(env, config, use_mock_vlm=USE_MOCK).plan()

    length = None
    if path:
        length = round(
            sum(math.dist(path[k], path[k + 1]) for k in range(len(path) - 1)), 2
        )
    ws.append([seed, "Yes" if path else "No", iters, length])
    wb.save(OUT_FILE)  # save after every seed so nothing is lost if a run stops

print(f"Saved {OUT_FILE}")