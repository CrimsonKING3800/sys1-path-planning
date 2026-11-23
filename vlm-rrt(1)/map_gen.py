"""
map_gen.py - random map generator for the VLM-RRT planner.

Returns plain data (width, height, start, goal, obstacles) so it does not
depend on your planner file. Obstacles are (x, y, w, h) tuples, the same
format your Environment2D uses.

Usage in your planner file:

    from map_gen import generate_map
    env = Environment2D(**generate_map(seed=1))

For a benchmark over many maps:

    for seed in range(50):
        env = Environment2D(**generate_map(seed=seed))
        ...

Guarantees for every returned map:
  * start and goal are at least `clearance` metres away from every obstacle
  * obstacles stay inside the map
  * start and goal are at least `min_start_goal_dist` apart
  * a collision-free path exists (checked with a grid BFS), so a failed
    planner run is the planner's fault, not an unsolvable map
"""

import math
import random
from collections import deque
from typing import Dict, List, Optional, Tuple

Obstacle = Tuple[float, float, float, float]  # (x, y, w, h)


def _rect_distance_to_point(ob: Obstacle, px: float, py: float) -> float:
    """Distance from a point to a rectangle (0 if inside)."""
    ox, oy, ow, oh = ob
    dx = max(ox - px, 0.0, px - (ox + ow))
    dy = max(oy - py, 0.0, py - (oy + oh))
    return math.hypot(dx, dy)


def _rects_overlap(a: Obstacle, b: Obstacle, gap: float) -> bool:
    """True if rectangles a and b are closer than `gap` metres."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return not (
        ax + aw + gap <= bx
        or bx + bw + gap <= ax
        or ay + ah + gap <= by
        or by + bh + gap <= ay
    )


def _is_solvable(
    width: float,
    height: float,
    start: Tuple[float, float],
    goal: Tuple[float, float],
    obstacles: List[Obstacle],
    cell: float = 5.0,
    inflate: float = 5.0,
) -> bool:
    """Grid BFS on obstacles inflated by `inflate` metres."""
    nx, ny = int(width // cell), int(height // cell)
    blocked = [[False] * ny for _ in range(nx)]
    for i in range(nx):
        cx = (i + 0.5) * cell
        for j in range(ny):
            cy = (j + 0.5) * cell
            for ob in obstacles:
                if _rect_distance_to_point(ob, cx, cy) <= inflate:
                    blocked[i][j] = True
                    break

    def to_cell(p):
        return (
            min(max(int(p[0] // cell), 0), nx - 1),
            min(max(int(p[1] // cell), 0), ny - 1),
        )

    s, g = to_cell(start), to_cell(goal)
    blocked[s[0]][s[1]] = False
    blocked[g[0]][g[1]] = False

    seen = {s}
    q = deque([s])
    while q:
        cx, cy = q.popleft()
        if (cx, cy) == g:
            return True
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1),
                       (1, 1), (1, -1), (-1, 1), (-1, -1)):
            x, y = cx + dx, cy + dy
            if 0 <= x < nx and 0 <= y < ny and not blocked[x][y] and (x, y) not in seen:
                seen.add((x, y))
                q.append((x, y))
    return False


def generate_map(
    seed: Optional[int] = None,
    width: float = 500.0,
    height: float = 500.0,
    num_obstacles: Optional[int] = None,      # None -> random in num_obstacles_range
    num_obstacles_range: Tuple[int, int] = (8, 14),
    size_range: Tuple[float, float] = (30.0, 90.0),   # obstacle width/height
    clearance: float = 20.0,                  # min gap from start/goal to any obstacle
    obstacle_gap: float = 10.0,               # min gap between obstacles
    min_start_goal_dist: float = 300.0,
    corner_region: float = 120.0,             # start/goal sampled within this corner box
    max_attempts: int = 500,
) -> Dict:
    """
    Generate one random map. Start is sampled in the bottom-left corner
    region and goal in the top-right corner region (like the paper's
    figures). Returns a dict you can splat into Environment2D(**map).
    """
    rng = random.Random(seed)

    for _ in range(max_attempts):
        start = (
            rng.uniform(20.0, corner_region),
            rng.uniform(20.0, corner_region),
        )
        goal = (
            rng.uniform(width - corner_region, width - 20.0),
            rng.uniform(height - corner_region, height - 20.0),
        )
        if math.hypot(goal[0] - start[0], goal[1] - start[1]) < min_start_goal_dist:
            continue

        n = num_obstacles if num_obstacles is not None else rng.randint(*num_obstacles_range)
        obstacles: List[Obstacle] = []
        tries = 0
        while len(obstacles) < n and tries < n * 60:
            tries += 1
            w = rng.uniform(*size_range)
            h = rng.uniform(*size_range)
            x = rng.uniform(0.0, width - w)
            y = rng.uniform(0.0, height - h)
            cand = (x, y, w, h)

            if _rect_distance_to_point(cand, *start) < clearance:
                continue
            if _rect_distance_to_point(cand, *goal) < clearance:
                continue
            if any(_rects_overlap(cand, ob, obstacle_gap) for ob in obstacles):
                continue
            obstacles.append(cand)

        if len(obstacles) < max(1, n // 2):
            continue  # couldn't place enough, try a new start/goal
        if not _is_solvable(width, height, start, goal, obstacles):
            continue

        return {
            "width": width,
            "height": height,
            "start": (round(start[0], 2), round(start[1], 2)),
            "goal": (round(goal[0], 2), round(goal[1], 2)),
            "obstacles": [tuple(round(v, 2) for v in ob) for ob in obstacles],
        }

    raise RuntimeError(
        "Could not generate a valid map; try fewer obstacles or a smaller size_range."
    )


if __name__ == "__main__":
    # Preview a few maps: python map_gen.py
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for seed, ax in enumerate(axes.flat):
        m = generate_map(seed=seed)
        ax.set_xlim(0, m["width"])
        ax.set_ylim(0, m["height"])
        ax.set_aspect("equal")
        ax.set_title(f"seed={seed}  obstacles={len(m['obstacles'])}")
        for ox, oy, ow, oh in m["obstacles"]:
            ax.add_patch(patches.Rectangle((ox, oy), ow, oh, color="black"))
        ax.scatter(*m["start"], color="green", s=60, zorder=5)
        ax.scatter(*m["goal"], color="red", s=60, zorder=5)
    plt.tight_layout()
    plt.savefig("map_preview.png", dpi=100)
    print("Saved map_preview.png")