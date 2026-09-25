"""Shared helper: roll a policy forward for roughly half of the shortest-
path distance from its start to the target, so sampled "current state"
snapshots are usually still mid-episode (not trivially already at the
goal) but have accumulated some real steps/obstacle hits.
"""
from __future__ import annotations

import numpy as np

import gridworld as g


def sample_cells(n: int, seed: int = 42, min_dist: int = 3) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    free = [c for c in g.all_free_cells()
            if c != g.TARGET and g.BFS_DIST_TO_TARGET.get(c, 0) >= min_dist]
    idx = rng.choice(len(free), size=n, replace=False)
    return [free[i] for i in idx]


def partial_rollout(action_probs_fn, start: tuple[int, int], rng: np.random.Generator,
                     target: tuple[int, int] = g.TARGET) -> dict:
    """Advance `action_probs_fn(cell) -> probs` from `start` for about half
    the optimal distance to `target` (at least 1, at most 6 steps)."""
    budget = int(np.clip(g.BFS_DIST_TO_TARGET.get(start, 4) // 2, 1, 6))
    cur = start
    hits, steps = 0, 0
    for _ in range(budget):
        if cur == target:
            break
        probs = action_probs_fn(cur)
        a = rng.choice(g.ACTIONS, p=probs)
        nxt, hit = g.step(cur, a)
        hits += int(hit)
        steps += 1
        cur = nxt
    return {"start": start, "cur": cur, "steps_taken": steps, "obstacle_hits": hits}
