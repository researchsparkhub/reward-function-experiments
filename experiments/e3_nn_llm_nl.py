"""E3 - same protocol as E2 (natural-language instruction + iterative LLM
judge refinement), but the reference/base policy is now a trained neural
network instead of the hand-written rule, so the LLM is being asked to
track an imperfect, learned policy rather than a near-optimal one.
"""
from __future__ import annotations

import json
import os
import pickle
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

import gridworld as g
import nn_policy as nnp
import llm_judge as lj
import instructions as instr
import metrics as met
import viz
import state_sampler as ss
import rule_policy as rp  # only for the software rubric ground truth

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "e3")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "report", "figures")
N_CELLS = 10
MAX_ITERS = 4
TOL = 0.05
RUBRIC_FIELDS = ["closeness", "avoided_damage", "path_efficiency"]


def get_policy():
    ckpt = os.path.join(OUT_DIR, "nn_policy.pkl")
    if os.path.exists(ckpt):
        with open(ckpt, "rb") as f:
            return pickle.load(f)
    pol = nnp.MLPPolicy(seed=0)
    hist = pol.train(verbose=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(ckpt, "wb") as f:
        pickle.dump(pol, f)
    print("NN policy final success rate:", np.mean(hist[-300:]))
    return pol


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    pol = get_policy()
    cells = ss.sample_cells(N_CELLS, seed=44)
    rng = np.random.default_rng(23)

    records = []
    for i, start in enumerate(cells):
        state = ss.partial_rollout(pol.action_probs, start, rng)
        cur, steps, hits = state["cur"], state["steps_taken"], state["obstacle_hits"]
        nn_probs = pol.action_probs(cur)
        sw_rubric = rp.state_rubric(start, cur, steps, hits)  # software ground truth, for context
        instruction = instr.instruction_for(i)

        history = lj.iterative_judge(cur, nn_probs, g.TARGET, instruction, hits, steps,
                                       max_iters=MAX_ITERS, tol=TOL)
        final = history[-1]
        final_probs = lj.probs_array(final.judge_out)
        rule_probs = rp.action_probs(cur)  # for a three-way comparison in the report

        rec = {
            "start": start, "cur": cur, "steps_taken": steps, "obstacle_hits": hits,
            "instruction": instruction,
            "nn_probs": nn_probs.tolist(), "rule_probs": rule_probs.tolist(),
            "sw_rubric": sw_rubric,
            "iterations_used": final.iteration,
            "converged": final.js_div <= TOL,
            "js_div_per_iter": [h.js_div for h in history],
            "rubric_per_iter": [{k: h.judge_out[k] for k in
                ["reached_target", "closeness", "avoided_damage", "path_efficiency"]}
                for h in history],
            "final_probs": final_probs.tolist(),
            "final_rubric": {k: final.judge_out[k] for k in RUBRIC_FIELDS},
            "final_rubric_mae": met.rubric_mae(sw_rubric, final.judge_out, RUBRIC_FIELDS),
            "final_cosine_sim_vs_nn": met.cosine_sim(nn_probs, final_probs),
            "final_cosine_sim_vs_rule": met.cosine_sim(rule_probs, final_probs),
            "final_argmax_agree_vs_nn": met.agreement_rate(nn_probs, final_probs),
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
        "mean_final_cosine_sim_vs_nn": float(np.mean([r["final_cosine_sim_vs_nn"] for r in records])),
        "mean_final_cosine_sim_vs_rule": float(np.mean([r["final_cosine_sim_vs_rule"] for r in records])),
        "final_argmax_agreement_vs_nn": float(np.mean([r["final_argmax_agree_vs_nn"] for r in records])),
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
    viz.line_plot(iters, {"mean JS divergence to NN policy": mean_js_by_iter},
                  "refinement round", "Jensen-Shannon divergence",
                  "E3: LLM judge convergence toward the NN policy",
                  os.path.join(FIG_DIR, "e3_convergence.png"))

    nn_field = {str(r["cur"]): r["nn_probs"] for r in records}
    llm_field = {str(r["cur"]): r["final_probs"] for r in records}
    viz.action_field_plot(nn_field, llm_field, [r["cur"] for r in records],
                           "NN policy", "LLM judge (with instruction, converged)",
                           os.path.join(FIG_DIR, "e3_action_field.png"))

    return records, summary


if __name__ == "__main__":
    run()
