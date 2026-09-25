"""A small 2-D gridworld shared by all four experiments (E1-E4).

The map is an 8x8 grid (64 cells) with a border wall plus interior wall
segments, one target object and two distractor objects, following the
"registered one-wall map" layout used throughout this project. The action
space is the four cardinal moves {up, right, down, left}; an action that
would leave the grid or enter a wall cell is a no-op and counts as the
robot "hitting an obstacle".
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

SIZE = 8
ACTIONS = ["up", "right", "down", "left"]
ACTION_DELTA = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}

# Interior wall rows (0-indexed), each with a single one-cell gap, mirroring
# the reference "registered one-wall map-use control" figure: horizontal
# wall bands with a small opening, repeated down the map.
_WALL_ROWS = {
    2: {1, 2, 4, 5, 6},  # gap at column 3
    5: {1, 2, 3, 5, 6},  # gap at column 4
}


def _build_walls() -> np.ndarray:
    walls = np.zeros((SIZE, SIZE), dtype=bool)
    walls[0, :] = True
    walls[SIZE - 1, :] = True
    walls[:, 0] = True
    walls[:, SIZE - 1] = True
    for row, cols in _WALL_ROWS.items():
        for c in cols:
            walls[row, c] = True
    return walls


WALLS = _build_walls()

TARGET = (4, 4)          # "sofa" - the goal object
DISTRACTORS = [(1, 6), (6, 6)]  # "chair", "cube"
OBJECT_NAMES = {TARGET: "sofa", DISTRACTORS[0]: "chair", DISTRACTORS[1]: "cube"}

START = (1, 1)


def in_bounds(cell: tuple[int, int]) -> bool:
    r, c = cell
    return 0 <= r < SIZE and 0 <= c < SIZE


def is_free(cell: tuple[int, int]) -> bool:
    r, c = cell
    return in_bounds(cell) and not WALLS[r, c]


def all_free_cells() -> list[tuple[int, int]]:
    return [(r, c) for r in range(SIZE) for c in range(SIZE) if is_free((r, c))]


def step(cell: tuple[int, int], action: str) -> tuple[tuple[int, int], bool]:
    """Apply an action; returns (new_cell, hit_obstacle)."""
    dr, dc = ACTION_DELTA[action]
    nxt = (cell[0] + dr, cell[1] + dc)
    if is_free(nxt):
        return nxt, False
    return cell, True


def manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def bfs_distances(target: tuple[int, int]) -> dict[tuple[int, int], int]:
    """Shortest-path distance (in moves, respecting walls) from every free
    cell to `target`."""
    from collections import deque

    dist = {target: 0}
    q = deque([target])
    while q:
        cur = q.popleft()
        for a in ACTIONS:
            nxt, hit = step(cur, a)
            if not hit and nxt not in dist:
                dist[nxt] = dist[cur] + 1
                q.append(nxt)
    return dist


BFS_DIST_TO_TARGET = bfs_distances(TARGET)


@dataclass
class Trajectory:
    start: tuple[int, int]
    target: tuple[int, int] = TARGET
    positions: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    obstacle_hits: int = 0
    reached: bool = False

    def path_len(self) -> int:
        return len(self.actions)


def render_ascii(agent: Optional[tuple[int, int]] = None) -> str:
    """Text rendering of the map used both for the report and as the LLM
    judge's input."""
    rows = []
    for r in range(SIZE):
        row_chars = []
        for c in range(SIZE):
            cell = (r, c)
            if WALLS[r, c]:
                row_chars.append("#")
            elif agent is not None and cell == agent:
                row_chars.append("A")
            elif cell == TARGET:
                row_chars.append("T")
            elif cell in DISTRACTORS:
                row_chars.append("D")
            else:
                row_chars.append(".")
        rows.append("".join(row_chars))
    return "\n".join(rows)


def scene_description() -> dict:
    return {
        "size": SIZE,
        "walls": WALLS.astype(int).tolist(),
        "target": {"cell": list(TARGET), "name": "sofa"},
        "distractors": [
            {"cell": list(d), "name": OBJECT_NAMES[d]} for d in DISTRACTORS
        ],
        "actions": ACTIONS,
    }


if __name__ == "__main__":
    print(render_ascii(START))
    print(json.dumps(scene_description(), indent=2)[:500])
