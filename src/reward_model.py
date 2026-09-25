"""E4: replace the LLM judge with a learned reward network.

A small MLP regressor is distilled from the (state -> rubric) pairs the
LLM judge produced in E2/E3: it predicts the same four rubric fields
directly from grid features, with no further API calls. Action
probabilities are then derived the same way the rule-based policy derives
them in E1, but using the *learned* closeness prediction in place of the
true BFS distance.
"""
from __future__ import annotations

import numpy as np
from sklearn.neural_network import MLPRegressor

import gridworld as g

RUBRIC_FIELDS = ["reached_target", "closeness", "avoided_damage", "path_efficiency"]
TEMPERATURE = 0.15


def features(cell: tuple[int, int], obstacle_hits: int = 0, steps_taken: int = 0,
             target: tuple[int, int] = g.TARGET) -> np.ndarray:
    r, c = cell
    tr, tc = target
    return np.array([
        r / g.SIZE, c / g.SIZE,
        (tr - r) / g.SIZE, (tc - c) / g.SIZE,
        min(obstacle_hits, 5) / 5.0,
        min(steps_taken, 40) / 40.0,
    ])


def build_dataset(records: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """records: list of {"cell": (r,c), "obstacle_hits": int, "steps_taken": int,
    "judge_out": {rubric fields...}}"""
    X, Y = [], []
    for rec in records:
        X.append(features(tuple(rec["cell"]), rec.get("obstacle_hits", 0),
                            rec.get("steps_taken", 0)))
        Y.append([rec["judge_out"][k] for k in RUBRIC_FIELDS])
    return np.array(X), np.array(Y)


class RewardNetwork:
    def __init__(self, seed: int = 0):
        self.model = MLPRegressor(hidden_layer_sizes=(32, 16), activation="tanh",
                                    max_iter=4000, random_state=seed, alpha=1e-3)

    def fit(self, X: np.ndarray, Y: np.ndarray):
        self.model.fit(X, Y)
        return self

    def predict_rubric(self, cell: tuple[int, int], obstacle_hits: int = 0,
                        steps_taken: int = 0) -> dict:
        x = features(cell, obstacle_hits, steps_taken).reshape(1, -1)
        y = np.clip(self.model.predict(x)[0], 0.0, 1.0)
        return dict(zip(RUBRIC_FIELDS, y.tolist()))

    def action_probs(self, cell: tuple[int, int], obstacle_hits: int = 0,
                      steps_taken: int = 0) -> np.ndarray:
        cur = self.predict_rubric(cell, obstacle_hits, steps_taken)["closeness"]
        logits = np.zeros(len(g.ACTIONS))
        for i, a in enumerate(g.ACTIONS):
            nxt, hit = g.step(cell, a)
            if hit:
                logits[i] = -5.0
                continue
            nxt_close = self.predict_rubric(nxt, obstacle_hits, steps_taken + 1)["closeness"]
            logits[i] = (nxt_close - cur) / TEMPERATURE
        logits -= logits.max()
        p = np.exp(logits)
        return p / p.sum()
