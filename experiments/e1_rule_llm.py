"""E1 - rule-based ("software") policy vs. LLM judge, no natural-language
instruction. Every eligible grid cell is rolled out several independent
times ("replicates") to reach a statistically reasonable sample size from
a small map, then both the software rubric and a real LLM judge score
that same situation and propose an action-probability distribution over
the four moves. Compare the two.
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
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
REPLICATES = 4  # 16 eligible cells x 4 replicates = 64 situations
RUBRIC_FIELDS = ["closeness", "avoided_damage", "path_efficiency"]


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    situations = ss.build_situations(rp.action_probs, replicates=REPLICATES, seed_base=1)
    print(f"E1: {len(situations)} situations "
          f"({len(ss.eligible_cells())} cells x {REPLICATES} replicates)")

    records = []
    n_failed = 0
    for k, state in enumerate(situations):
        cur, steps, hits = state["cur"], state["steps_taken"], state["obstacle_hits"]
        start = state["start"]

        rule_probs = rp.action_probs(cur)
        sw_rubric = rp.state_rubric(start, cur, steps, hits)

        try:
            llm_out = lj.judge(cur, g.TARGET, instruction=None, obstacle_hits=hits,
                                steps_taken=steps)
        except Exception as e:  # noqa: BLE001
            print(f"[{k+1}/{len(situations)}] SKIPPED (judge failed: {e})")
            n_failed += 1
            continue
        llm_probs = lj.probs_array(llm_out)

        rec = {
            "start": start, "cur": cur, "steps_taken": steps, "obstacle_hits": hits,
            "rule_probs": rule_probs.tolist(), "llm_probs": llm_probs.tolist(),
            "sw_rubric": sw_rubric, "llm_rubric": {k2: llm_out[k2] for k2 in
                ["reached_target", "closeness", "avoided_damage", "path_efficiency"]},
            "llm_reasoning": llm_out["reasoning"], "latency_s": llm_out["_latency_s"],
            "js_div": met.js_divergence(rule_probs, llm_probs),
            "cosine_sim": met.cosine_sim(rule_probs, llm_probs),
            "argmax_agree": met.agreement_rate(rule_probs, llm_probs),
            "rubric_mae": met.rubric_mae(sw_rubric, llm_out, RUBRIC_FIELDS),
        }
        records.append(rec)
        if (k + 1) % 8 == 0 or k == len(situations) - 1:
            print(f"[{k+1}/{len(situations)}] cell={cur} steps={steps} hits={hits} "
                  f"JS={rec['js_div']:.3f} agree={rec['argmax_agree']}")
            with open(os.path.join(OUT_DIR, "records.json"), "w") as f:
                json.dump(records, f, indent=2, default=str)

    if n_failed:
        print(f"{n_failed} situation(s) skipped after exhausting retries")
    n = len(records)
    n_agree = int(sum(r["argmax_agree"] for r in records))
    agree_lo, agree_hi = met.wilson_ci(n_agree, n)
    js_mean, js_lo, js_hi = met.mean_ci([r["js_div"] for r in records])
    mae_mean, mae_lo, mae_hi = met.mean_ci([r["rubric_mae"] for r in records])

    summary = {
        "n_situations": n, "n_failed": n_failed,
        "n_cells": len(ss.eligible_cells()), "replicates": REPLICATES,
        "mean_js_div": js_mean, "js_div_ci95": [js_lo, js_hi],
        "mean_cosine_sim": float(np.mean([r["cosine_sim"] for r in records])),
        "argmax_agreement_rate": n_agree / n, "argmax_agreement_ci95": [agree_lo, agree_hi],
        "mean_rubric_mae": mae_mean, "rubric_mae_ci95": [mae_lo, mae_hi],
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

    viz.scatter_calibration(
        [r["sw_rubric"]["closeness"] for r in records],
        [r["llm_rubric"]["closeness"] for r in records],
        "software closeness", "LLM closeness",
        f"E1: closeness rubric calibration (n={n})",
        os.path.join(FIG_DIR, "e1_rubric_closeness.png"))

    viz.scatter_calibration(
        [r["sw_rubric"]["path_efficiency"] for r in records],
        [r["llm_rubric"]["path_efficiency"] for r in records],
        "software path efficiency", "LLM path efficiency",
        f"E1: path-efficiency rubric calibration (n={n})",
        os.path.join(FIG_DIR, "e1_rubric_efficiency.png"))

    return records, summary


if __name__ == "__main__":
    run()
