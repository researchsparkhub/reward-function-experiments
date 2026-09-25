"""E3's "neural network as policy": a tiny 2-layer MLP (position -> hidden
tanh layer -> softmax over the 4 moves), trained with REINFORCE against a
dense distance-shaped reward. This mirrors the pedagogical NN sim already
in the project (rl_nn_policy_sim.html) but in numpy, sized for this map.
"""
from __future__ import annotations

import numpy as np

import gridworld as g

HIDDEN = 16
LR = 0.05
GAMMA = 0.95
EPISODES = 1500
MAX_STEPS = 40
STEP_COST = -0.02
PROGRESS_COEF = 0.2
SUCCESS_BONUS = 2.0


def featurize(cell: tuple[int, int]) -> np.ndarray:
    r, c = cell
    tr, tc = g.TARGET
    return np.array([r / g.SIZE, c / g.SIZE, (tr - r) / g.SIZE, (tc - c) / g.SIZE], dtype=np.float64)


class MLPPolicy:
    def __init__(self, seed: int = 0):
        rng = np.random.default_rng(seed)
        n_in, n_out = 4, len(g.ACTIONS)
        self.W1 = rng.normal(0, 0.5, (n_in, HIDDEN))
        self.b1 = np.zeros(HIDDEN)
        self.W2 = rng.normal(0, 0.5, (HIDDEN, n_out))
        self.b2 = np.zeros(n_out)

    def forward(self, x: np.ndarray):
        h = np.tanh(x @ self.W1 + self.b1)
        logits = h @ self.W2 + self.b2
        logits = logits - logits.max()
        p = np.exp(logits) / np.exp(logits).sum()
        return h, p

    def action_probs(self, cell: tuple[int, int]) -> np.ndarray:
        _, p = self.forward(featurize(cell))
        return p

    def train(self, rng: np.random.Generator | None = None, verbose: bool = False):
        rng = rng or np.random.default_rng(1)
        free_cells = g.all_free_cells()
        history = []
        for ep in range(EPISODES):
            start = free_cells[rng.integers(len(free_cells))]
            cell = start
            states, hs, acts, rewards = [], [], [], []
            dist = g.BFS_DIST_TO_TARGET
            prev_d = dist.get(cell, g.SIZE * 2)
            for t in range(MAX_STEPS):
                x = featurize(cell)
                h, p = self.forward(x)
                a_idx = rng.choice(len(g.ACTIONS), p=p)
                a = g.ACTIONS[a_idx]
                nxt, hit = g.step(cell, a)
                d = dist.get(nxt, prev_d + 1)
                r = STEP_COST + PROGRESS_COEF * (prev_d - d)
                reached = nxt == g.TARGET
                if reached:
                    r += SUCCESS_BONUS
                states.append(x); hs.append(h); acts.append(a_idx); rewards.append(r)
                cell = nxt
                prev_d = d
                if reached:
                    break
            # discounted returns + baseline
            G, returns = 0.0, []
            for r in reversed(rewards):
                G = r + GAMMA * G
                returns.insert(0, G)
            returns = np.array(returns)
            baseline = returns.mean()
            adv = returns - baseline

            for x, h, a_idx, A in zip(states, hs, acts, adv):
                logits_grad = -self.forward(x)[1]
                logits_grad[a_idx] += 1
                dW2 = np.outer(h, logits_grad) * A
                db2 = logits_grad * A
                dh = (self.W2 @ logits_grad) * (1 - h ** 2) * A
                dW1 = np.outer(x, dh)
                db1 = dh
                self.W2 += LR * dW2
                self.b2 += LR * db2
                self.W1 += LR * dW1
                self.b1 += LR * db1

            history.append(cell == g.TARGET)
            if verbose and (ep + 1) % 300 == 0:
                rate = np.mean(history[-300:])
                print(f"episode {ep+1}: success rate (last 300) = {rate:.2f}")
        return history


def rollout(policy: MLPPolicy, start: tuple[int, int], max_steps: int = 40,
            rng: np.random.Generator | None = None) -> g.Trajectory:
    rng = rng or np.random.default_rng(0)
    traj = g.Trajectory(start=start, positions=[start])
    cell = start
    for _ in range(max_steps):
        p = policy.action_probs(cell)
        a = rng.choice(g.ACTIONS, p=p)
        nxt, hit = g.step(cell, a)
        traj.actions.append(a)
        traj.positions.append(nxt)
        traj.obstacle_hits += int(hit)
        cell = nxt
        if cell == g.TARGET:
            traj.reached = True
            break
    return traj


if __name__ == "__main__":
    pol = MLPPolicy()
    hist = pol.train(verbose=True)
    print("final success rate:", np.mean(hist[-300:]))
