"""The LLM-as-judge reward (E1-E3): a real call to Claude that reads the
map (as text + the literal start/goal PNG images), optionally a natural-
language instruction, and returns a structured rubric plus an action-
probability distribution over the four moves.

Requires ANTHROPIC_API_KEY in the environment.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict

import numpy as np
from anthropic import Anthropic

import gridworld as g
import viz

MODEL = "claude-haiku-4-5-20251001"

_client = None


def client() -> Anthropic:
    global _client
    if _client is None:
        _client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    return _client


RUBRIC_TOOL = {
    "name": "rubric_judgment",
    "description": "Report the rubric-based judgment of the robot's situation and the "
                    "probability the robot's policy should assign to each of the four moves.",
    "input_schema": {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Brief reasoning: how close the robot is to the target, "
                                "whether it has been taking a sensible route, and any damage.",
            },
            "reached_target": {"type": "number", "minimum": 0, "maximum": 1},
            "closeness": {
                "type": "number", "minimum": 0, "maximum": 1,
                "description": "1.0 = at the target, 0.0 = as far as possible from it.",
            },
            "avoided_damage": {
                "type": "number", "minimum": 0, "maximum": 1,
                "description": "1.0 = never hit a wall/obstacle, 0.0 = hit obstacles repeatedly.",
            },
            "path_efficiency": {
                "type": "number", "minimum": 0, "maximum": 1,
                "description": "1.0 = shortest possible route so far, 0.0 = very wasteful.",
            },
            "action_probs": {
                "type": "object",
                "description": "Probability (summing to 1) that each move is the right next "
                                "action for the robot to take right now.",
                "properties": {
                    "up": {"type": "number", "minimum": 0, "maximum": 1},
                    "right": {"type": "number", "minimum": 0, "maximum": 1},
                    "down": {"type": "number", "minimum": 0, "maximum": 1},
                    "left": {"type": "number", "minimum": 0, "maximum": 1},
                },
                "required": ["up", "right", "down", "left"],
            },
        },
        "required": ["reasoning", "reached_target", "closeness", "avoided_damage",
                      "path_efficiency", "action_probs"],
    },
}


def _scene_text(cell: tuple[int, int], target: tuple[int, int], obstacle_hits: int,
                 steps_taken: int, instruction: str | None) -> str:
    lines = [
        "You are judging a robot in an 8x8 grid world (# = wall, . = free cell, "
        "A = robot, T = target object 'sofa', D = distractor object).",
        "Map:",
        g.render_ascii(cell),
        f"Robot cell (row,col): {cell}. Target cell: {target}.",
        f"Steps taken so far: {steps_taken}. Times it has bumped a wall: {obstacle_hits}.",
        "Two images are attached: the robot's current view (start) and what the grid looks "
        "like when the robot is exactly at the target (goal).",
    ]
    if instruction:
        lines.append(f"Natural-language instruction given to the robot: \"{instruction}\"")
    else:
        lines.append("No natural-language instruction is given; judge using the map alone.")
    lines.append(
        "Score the rubric fields and give an action_probs distribution over "
        "{up, right, down, left} for what the robot should do next."
    )
    return "\n".join(lines)


def judge(cell: tuple[int, int], target: tuple[int, int] = g.TARGET,
          instruction: str | None = None, obstacle_hits: int = 0, steps_taken: int = 0,
          feedback: str | None = None, use_images: bool = True, max_retries: int = 4) -> dict:
    """One call to the LLM judge. Returns the parsed tool input plus timing."""
    start_png, end_png = viz.start_end_images(cell, target) if use_images else (None, None)
    text = _scene_text(cell, target, obstacle_hits, steps_taken, instruction)
    if feedback:
        text += "\n\nFeedback on your previous answer for this same situation:\n" + feedback

    content = [{"type": "text", "text": text}]
    if use_images:
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                        "data": start_png}})
        content.append({"type": "image", "source": {"type": "base64", "media_type": "image/png",
                                                        "data": end_png}})

    required = ["reasoning", "reached_target", "closeness", "avoided_damage",
                "path_efficiency", "action_probs"]
    last_err = None
    for attempt in range(max_retries):
        try:
            t0 = time.time()
            resp = client().messages.create(
                model=MODEL,
                max_tokens=1000,
                tools=[RUBRIC_TOOL],
                tool_choice={"type": "tool", "name": "rubric_judgment"},
                messages=[{"role": "user", "content": content}],
            )
            latency = time.time() - t0
            out = None
            for block in resp.content:
                if block.type == "tool_use":
                    out = dict(block.input)
                    break
            if out is None:
                raise RuntimeError("no tool_use block in response")
            missing = [k for k in required if k not in out]
            if missing:
                raise RuntimeError(
                    f"incomplete tool_use output (stop_reason={resp.stop_reason}, "
                    f"missing={missing})")
            if not isinstance(out.get("action_probs"), dict) or \
                    set(out["action_probs"]) < set(g.ACTIONS):
                raise RuntimeError(f"malformed action_probs: {out.get('action_probs')!r}")
            out["_latency_s"] = latency
            return out
        except Exception as e:  # noqa: BLE001
            last_err = e
            time.sleep(min(2 ** attempt, 8))
    raise RuntimeError(f"LLM judge call failed after {max_retries} attempts: {last_err}")


def probs_array(judge_out: dict) -> np.ndarray:
    p = judge_out["action_probs"]
    arr = np.array([p[a] for a in g.ACTIONS], dtype=float)
    s = arr.sum()
    return arr / s if s > 0 else np.ones(4) / 4


def js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-9, 1); p = p / p.sum()
    q = np.clip(q, 1e-9, 1); q = q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: np.sum(a * np.log(a / b))
    return float(0.5 * kl(p, m) + 0.5 * kl(q, m))


@dataclass
class IterationRecord:
    iteration: int
    judge_out: dict
    js_div: float
    latency_s: float


def iterative_judge(cell: tuple[int, int], reference_probs: np.ndarray,
                     target: tuple[int, int] = g.TARGET, instruction: str | None = None,
                     obstacle_hits: int = 0, steps_taken: int = 0,
                     max_iters: int = 4, tol: float = 0.05, use_images: bool = True) -> list:
    """Repeatedly ask the LLM judge for this same situation, each time telling
    it how far its action distribution is from the reference policy's, and
    see how many rounds it takes to converge (E2/E3's 'iterations' metric)."""
    records = []
    feedback = None
    for it in range(1, max_iters + 1):
        out = judge(cell, target, instruction, obstacle_hits, steps_taken, feedback,
                    use_images=use_images)
        p = probs_array(out)
        div = js_divergence(reference_probs, p)
        records.append(IterationRecord(it, out, div, out.get("_latency_s", 0.0)))
        if div <= tol:
            break
        ref_str = ", ".join(f"{a}={reference_probs[i]:.2f}" for i, a in enumerate(g.ACTIONS))
        feedback = (
            f"Your action_probs were {out['action_probs']}. After simulating many robots with "
            f"this kind of situation, the move that actually reduces distance to the target and "
            f"respects the instruction has empirical preference ({ref_str}). Your distribution's "
            f"Jensen-Shannon divergence from that empirical preference is {div:.3f} "
            f"(target <= {tol:.2f}). Revise your reasoning and action_probs to move closer to it, "
            f"without simply copying the numbers verbatim."
        )
    return records


if __name__ == "__main__":
    out = judge(g.START, instruction=None)
    print(json.dumps(out, indent=2))
