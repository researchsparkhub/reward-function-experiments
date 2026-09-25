"""E2 - same rule-based ("software") policy as E1, but now the LLM judge
also receives a natural-language instruction, and is iteratively re-asked
(with feedback on how far its action distribution is from the reference
policy's) up to `MAX_ITERS` times. Tracks both judgment quality and how
many rounds of refinement it took to converge.
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
import instructions as instr
import metrics as met
import viz
import state_sampler as ss

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "e2")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "report", "figures")
N_CELLS = 10
MAX_ITERS = 4
TOL = 0.05
RUBRIC_FIELDS = ["closeness", "avoided_damage", "path_efficiency"]


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    cells = ss.sample_cells(N_CELLS, seed=43)
    rng = np.random.default_rng(11)

    records = []
    for i, start in enumerate(cells):
        state = ss.partial_rollout(rp.action_probs, start, rng)
        cur, steps, hits = state["cur"], state["steps_taken"], state["obstacle_hits"]
        rule_probs = rp.action_probs(cur)
        sw_rubric = rp.state_rubric(start, cur, steps, hits)
        instruction = instr.instruction_for(i)

        history = lj.iterative_judge(cur, rule_probs, g.TARGET, instruction, hits, steps,
                                       max_iters=MAX_ITERS, tol=TOL)
        final = history[-1]
        final_probs = lj.probs_array(final.judge_out)

        rec = {
            "start": start, "cur": cur, "steps_taken": steps, "obstacle_hits": hits,
            "instruction": instruction,
            "rule_probs": rule_probs.tolist(), "sw_rubric": sw_rubric,
            "iterations_used": final.iteration,
            "converged": final.js_div <= TOL,
            "js_div_per_iter": [h.js_div for h in history],
            "rubric_per_iter": [{k: h.judge_out[k] for k in
                ["reached_target", "closeness", "avoided_damage", "path_efficiency"]}
                for h in history],
            "final_probs": final_probs.tolist(),
            "final_rubric": {k: final.judge_out[k] for k in RUBRIC_FIELDS},
            "final_rubric_mae": met.rubric_mae(sw_rubric, final.judge_out, RUBRIC_FIELDS),
            "final_cosine_sim": met.cosine_sim(rule_probs, final_probs),
            "final_argmax_agree": met.agreement_rate(rule_probs, final_probs),
            "total_latency_s": float(sum(h.latency_s for h in history)),
        }
        records.append(rec)
        print(f"cell={cur} iters={rec['iterations_used']} converged={rec['converged']} "
              f"JS_per_iter={[round(x,3) for x in rec['js_div_per_iter']]}")

    summary = {
        "n_cells": N_CELLS, "max_iters": MAX_ITERS, "tol": TOL,
        "mean_iterations": float(np.mean([r["iterations_used"] for r in records])),
        "convergence_rate": float(np.mean([r["converged"] for r in records])),
        "mean_final_js_div": float(np.mean([r["js_div_per_iter"][-1] for r in records])),
        "mean_final_cosine_sim": float(np.mean([r["final_cosine_sim"] for r in records])),
        "final_argmax_agreement_rate": float(np.mean([r["final_argmax_agree"] for r in records])),
        "mean_final_rubric_mae": float(np.mean([r["final_rubric_mae"] for r in records])),
        "mean_total_latency_s": float(np.mean([r["total_latency_s"] for r in records])),
    }
    print("SUMMARY", json.dumps(summary, indent=2))

    with open(os.path.join(OUT_DIR, "records.json"), "w") as f:
        json.dump(records, f, indent=2, default=str)
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # --- figures ---
    max_len = max(len(r["js_div_per_iter"]) for r in records)
    iters = list(range(1, max_len + 1))
    mean_js_by_iter = []
    for k in range(max_len):
        vals = [r["js_div_per_iter"][k] for r in records if len(r["js_div_per_iter"]) > k]
        mean_js_by_iter.append(float(np.mean(vals)))
    viz.line_plot(iters, {"mean JS divergence to rule-based policy": mean_js_by_iter},
                  "refinement round", "Jensen-Shannon divergence",
                  "E2: LLM judge convergence toward the rule-based policy",
                  os.path.join(FIG_DIR, "e2_convergence.png"))

    labels = [str(r["cur"]) for r in records]
    viz.bar_overlay(labels, {"iterations to converge": [r["iterations_used"] for r in records]},
                     "iterations", "E2: rounds of refinement needed per situation",
                     os.path.join(FIG_DIR, "e2_iterations.png"))

    rule_field = {str(r["cur"]): r["rule_probs"] for r in records}
    llm_field = {str(r["cur"]): r["final_probs"] for r in records}
    viz.action_field_plot(rule_field, llm_field, [r["cur"] for r in records],
                           "Rule-based policy", "LLM judge (with instruction, converged)",
                           os.path.join(FIG_DIR, "e2_action_field.png"))

    return records, summary


if __name__ == "__main__":
    run()
