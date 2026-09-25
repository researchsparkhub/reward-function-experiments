"""E1 - rule-based ("software") policy vs. LLM judge, no natural-language
instruction. For a sample of grid cells, roll the rule-based policy
forward a few steps, then ask both the software rubric and a real LLM
judge to score that same situation and propose an action-probability
distribution over the four moves. Compare the two.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

import gridworld as g
import rule_policy as rp
import llm_judge as lj
import metrics as met
import viz
import state_sampler as ss

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "e1")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "report", "figures")
N_CELLS = 10
RUBRIC_FIELDS = ["closeness", "avoided_damage", "path_efficiency"]


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    cells = ss.sample_cells(N_CELLS)
    rng = np.random.default_rng(7)

    records = []
    for start in cells:
        state = ss.partial_rollout(rp.action_probs, start, rng)
        cur, steps, hits = state["cur"], state["steps_taken"], state["obstacle_hits"]

        rule_probs = rp.action_probs(cur)
        sw_rubric = rp.state_rubric(start, cur, steps, hits)

        llm_out = lj.judge(cur, g.TARGET, instruction=None, obstacle_hits=hits,
                            steps_taken=steps)
        llm_probs = lj.probs_array(llm_out)

        rec = {
            "start": start, "cur": cur, "steps_taken": steps, "obstacle_hits": hits,
            "rule_probs": rule_probs.tolist(), "llm_probs": llm_probs.tolist(),
            "sw_rubric": sw_rubric, "llm_rubric": {k: llm_out[k] for k in
                ["reached_target", "closeness", "avoided_damage", "path_efficiency"]},
            "llm_reasoning": llm_out["reasoning"], "latency_s": llm_out["_latency_s"],
            "js_div": met.js_divergence(rule_probs, llm_probs),
            "cosine_sim": met.cosine_sim(rule_probs, llm_probs),
            "argmax_agree": met.agreement_rate(rule_probs, llm_probs),
            "rubric_mae": met.rubric_mae(sw_rubric, llm_out, RUBRIC_FIELDS),
        }
        records.append(rec)
        print(f"cell={cur} steps={steps} hits={hits} JS={rec['js_div']:.3f} "
              f"agree={rec['argmax_agree']} rubric_MAE={rec['rubric_mae']:.3f}")

    summary = {
        "n_cells": N_CELLS,
        "mean_js_div": float(np.mean([r["js_div"] for r in records])),
        "mean_cosine_sim": float(np.mean([r["cosine_sim"] for r in records])),
        "argmax_agreement_rate": float(np.mean([r["argmax_agree"] for r in records])),
        "mean_rubric_mae": float(np.mean([r["rubric_mae"] for r in records])),
        "mean_llm_latency_s": float(np.mean([r["latency_s"] for r in records])),
    }
    print("SUMMARY", json.dumps(summary, indent=2))

    with open(os.path.join(OUT_DIR, "records.json"), "w") as f:
        json.dump(records, f, indent=2, default=str)
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # --- figures ---
    rule_field = {str(r["cur"]): r["rule_probs"] for r in records}
    llm_field = {str(r["cur"]): r["llm_probs"] for r in records}
    viz.action_field_plot(rule_field, llm_field, [r["cur"] for r in records],
                           "Rule-based policy", "LLM judge",
                           os.path.join(FIG_DIR, "e1_action_field.png"))

    labels = [str(r["cur"]) for r in records]
    viz.bar_overlay(labels, {
        "software closeness": [r["sw_rubric"]["closeness"] for r in records],
        "LLM closeness": [r["llm_rubric"]["closeness"] for r in records],
    }, "closeness score", "E1: software vs. LLM closeness rubric",
        os.path.join(FIG_DIR, "e1_rubric_closeness.png"))

    viz.bar_overlay(labels, {
        "software path_efficiency": [r["sw_rubric"]["path_efficiency"] for r in records],
        "LLM path_efficiency": [r["llm_rubric"]["path_efficiency"] for r in records],
    }, "path efficiency score", "E1: software vs. LLM path-efficiency rubric",
        os.path.join(FIG_DIR, "e1_rubric_efficiency.png"))

    return records, summary


if __name__ == "__main__":
    run()
