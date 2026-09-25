"""Shared helper: roll a policy forward for roughly half of the shortest-
path distance from its start to the target, so sampled "current state"
snapshots are usually still mid-episode (not trivially already at the
goal) but have accumulated some real steps/obstacle hits.

To get a statistically reasonable sample size out of a small map (16
grid cells are far enough from the target to be non-trivial), each
eligible cell is rolled out several times with independent random
draws ("replicates"), rather than sampling a handful of cells once.
"""
from __future__ import annotations

import numpy as np

import gridworld as g


def eligible_cells(min_dist: int = 3) -> list[tuple[int, int]]:
    return [c for c in g.all_free_cells()
            if c != g.TARGET and g.BFS_DIST_TO_TARGET.get(c, 0) >= min_dist]


def sample_cells(n: int, seed: int = 42, min_dist: int = 3) -> list[tuple[int, int]]:
    rng = np.random.default_rng(seed)
    free = eligible_cells(min_dist)
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


def build_situations(action_probs_fn, replicates: int, seed_base: int,
                      min_dist: int = 3, target: tuple[int, int] = g.TARGET) -> list[dict]:
    """Every eligible cell, rolled out `replicates` times with independent
    RNG draws, giving len(eligible_cells) * replicates situations from a
    single fixed map -- the statistically reasonable substitute for a
    handful of once-sampled cells."""
    cells = eligible_cells(min_dist)
    situations = []
    for i, cell in enumerate(cells):
        for r in range(replicates):
            rng = np.random.default_rng(seed_base * 1000 + i * replicates + r)
            situations.append(partial_rollout(action_probs_fn, cell, rng, target))
    return situations
