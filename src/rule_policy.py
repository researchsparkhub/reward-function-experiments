"""E1's "software as policy": a hand-written rule that turns distance-to-
target into an action-probability distribution over the four moves, plus
the matching rule-based rubric used as the software-side ground truth in
every experiment.

Policy: softmax over the *reduction* in shortest-path distance to the
target that each of the four actions would produce. An action that walks
into a wall gets a large penalty (never preferred). Temperature controls
how peaked the resulting distribution is.
"""
from __future__ import annotations

import numpy as np

import gridworld as g

TEMPERATURE = 0.7
WALL_PENALTY = 3.0  # extra "distance" charged for bumping a wall


def action_logits(cell: tuple[int, int], target: tuple[int, int] | None = None) -> np.ndarray:
    target = target or g.TARGET
    dist = g.bfs_distances(target) if target != g.TARGET else g.BFS_DIST_TO_TARGET
    cur_dist = dist.get(cell, WALL_PENALTY * g.SIZE)
    logits = np.zeros(len(g.ACTIONS))
    for i, a in enumerate(g.ACTIONS):
        nxt, hit = g.step(cell, a)
        if hit:
            nxt_dist = cur_dist + WALL_PENALTY
        else:
            nxt_dist = dist.get(nxt, cur_dist + WALL_PENALTY)
        progress = cur_dist - nxt_dist  # positive = gets closer
        logits[i] = progress / TEMPERATURE
    return logits


def action_probs(cell: tuple[int, int], target: tuple[int, int] | None = None) -> np.ndarray:
    logits = action_logits(cell, target)
    logits = logits - logits.max()
    exp = np.exp(logits)
    return exp / exp.sum()


def rollout(start: tuple[int, int], target: tuple[int, int] | None = None,
            max_steps: int = 40, rng: np.random.Generator | None = None) -> g.Trajectory:
    target = target or g.TARGET
    rng = rng or np.random.default_rng(0)
    traj = g.Trajectory(start=start, target=target, positions=[start])
    cell = start
    for _ in range(max_steps):
        probs = action_probs(cell, target)
        a = rng.choice(g.ACTIONS, p=probs)
        nxt, hit = g.step(cell, a)
        traj.actions.append(a)
        traj.positions.append(nxt)
        traj.obstacle_hits += int(hit)
        cell = nxt
        if cell == target:
            traj.reached = True
            break
    return traj


MAX_DIST = max(g.BFS_DIST_TO_TARGET.values())


def state_rubric(start: tuple[int, int], cur: tuple[int, int], steps_taken: int,
                  obstacle_hits: int, target: tuple[int, int] | None = None) -> dict:
    """Software-computed rubric for an *in-progress* situation (start cell,
    current cell, steps taken, obstacles hit so far) — the same quantity the
    LLM judge is asked to estimate from the map/instruction alone."""
    target = target or g.TARGET
    dist_to_target = g.bfs_distances(target)
    dist_from_start = g.bfs_distances(start)
    reached = float(cur == target)
    d_cur = dist_to_target.get(cur, MAX_DIST)
    closeness = 1.0 if reached else max(0.0, 1 - d_cur / MAX_DIST)
    avoided_damage = max(0.0, 1 - obstacle_hits / 5.0)
    optimal_to_cur = dist_from_start.get(cur, steps_taken)
    path_efficiency = 1.0 if steps_taken == 0 else min(1.0, optimal_to_cur / max(steps_taken, 1))
    return {
        "reached_target": round(reached, 4),
        "closeness": round(float(closeness), 4),
        "avoided_damage": round(float(avoided_damage), 4),
        "path_efficiency": round(float(path_efficiency), 4),
    }


def rubric_scores(traj: g.Trajectory) -> dict:
    """Programmatic ("software") rubric, on the same 0-1 scale the LLM
    judge is asked to use, for the same trajectory."""
    dist = g.bfs_distances(traj.target)
    start_dist = dist.get(traj.start, g.SIZE * 2)
    end_dist = dist.get(traj.positions[-1], start_dist)
    closeness = 1.0 if traj.reached else max(0.0, 1 - end_dist / max(start_dist, 1))
    shortest = dist.get(traj.start, 0)
    efficiency = 1.0 if not traj.reached else min(1.0, shortest / max(traj.path_len(), 1))
    damage = max(0.0, 1 - traj.obstacle_hits / 5.0)
    return {
        "reached_target": float(traj.reached),
        "closeness": round(float(closeness), 4),
        "avoided_damage": round(float(damage), 4),
        "path_efficiency": round(float(efficiency), 4),
    }


if __name__ == "__main__":
    for cell in [g.START, g.TARGET, (3, 3)]:
        print(cell, action_probs(cell).round(3))
