"""E2 - same rule-based ("software") policy as E1, but now the LLM judge
also receives a natural-language instruction, and is iteratively re-asked
(with feedback on how far its action distribution is from the reference
policy) up to `MAX_ITERS` times. Every eligible grid cell is rolled out
several independent times to reach a statistically reasonable sample
size. Tracks both judgment quality and how many rounds of refinement it
took to converge.
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
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "paper", "figures")
REPLICATES = 4  # 16 eligible cells x 4 replicates = 64 situations
MAX_ITERS = 4
TOL = 0.05
RUBRIC_FIELDS = ["closeness", "avoided_damage", "path_efficiency"]


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    situations = ss.build_situations(rp.action_probs, replicates=REPLICATES, seed_base=2)
    print(f"E2: {len(situations)} situations "
          f"({len(ss.eligible_cells())} cells x {REPLICATES} replicates)")

    records = []
    n_failed = 0
    for i, state in enumerate(situations):
        cur, steps, hits = state["cur"], state["steps_taken"], state["obstacle_hits"]
        start = state["start"]
        rule_probs = rp.action_probs(cur)
        sw_rubric = rp.state_rubric(start, cur, steps, hits)
        instruction = instr.instruction_for(i)

        try:
            history = lj.iterative_judge(cur, rule_probs, g.TARGET, instruction, hits, steps,
                                           max_iters=MAX_ITERS, tol=TOL)
        except Exception as e:  # noqa: BLE001
            print(f"[{i+1}/{len(situations)}] SKIPPED (judge failed: {e})")
            n_failed += 1
            continue
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
        if (i + 1) % 8 == 0 or i == len(situations) - 1:
            print(f"[{i+1}/{len(situations)}] cell={cur} iters={rec['iterations_used']} "
                  f"converged={rec['converged']}")
            with open(os.path.join(OUT_DIR, "records.json"), "w") as f:
                json.dump(records, f, indent=2, default=str)

    if n_failed:
        print(f"{n_failed} situation(s) skipped after exhausting retries")
    n = len(records)
    n_conv = int(sum(r["converged"] for r in records))
    conv_lo, conv_hi = met.wilson_ci(n_conv, n)
    n_agree = int(sum(r["final_argmax_agree"] for r in records))
    agree_lo, agree_hi = met.wilson_ci(n_agree, n)
    iters_mean, iters_lo, iters_hi = met.mean_ci([r["iterations_used"] for r in records])
    js_mean, js_lo, js_hi = met.mean_ci([r["js_div_per_iter"][-1] for r in records])
    mae_mean, mae_lo, mae_hi = met.mean_ci([r["final_rubric_mae"] for r in records])

    summary = {
        "n_situations": n, "n_failed": n_failed,
        "n_cells": len(ss.eligible_cells()), "replicates": REPLICATES,
        "max_iters": MAX_ITERS, "tol": TOL,
        "mean_iterations": iters_mean, "iterations_ci95": [iters_lo, iters_hi],
        "convergence_rate": n_conv / n, "convergence_ci95": [conv_lo, conv_hi],
        "mean_final_js_div": js_mean, "final_js_div_ci95": [js_lo, js_hi],
        "mean_final_cosine_sim": float(np.mean([r["final_cosine_sim"] for r in records])),
        "final_argmax_agreement_rate": n_agree / n,
        "final_argmax_agreement_ci95": [agree_lo, agree_hi],
        "mean_final_rubric_mae": mae_mean, "final_rubric_mae_ci95": [mae_lo, mae_hi],
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
    mean_js_by_iter, lo_by_iter, hi_by_iter = [], [], []
    for k in range(max_len):
        vals = [r["js_div_per_iter"][k] for r in records if len(r["js_div_per_iter"]) > k]
        m, lo, hi = met.mean_ci(vals)
        mean_js_by_iter.append(m); lo_by_iter.append(lo); hi_by_iter.append(hi)
    viz.line_plot(iters, {"mean JS divergence to rule-based policy": mean_js_by_iter},
                  "refinement round", "Jensen-Shannon divergence",
                  f"E2: LLM judge convergence toward the rule-based policy (n={n})",
                  os.path.join(FIG_DIR, "e2_convergence.png"),
                  band={"mean JS divergence to rule-based policy": (lo_by_iter, hi_by_iter)})

    viz.iteration_histogram([r["iterations_used"] for r in records], MAX_ITERS,
                             "iterations to converge", "number of situations",
                             f"E2: rounds of refinement needed (n={n})",
                             os.path.join(FIG_DIR, "e2_iterations.png"))

    rule_field = {str(r["cur"]): r["rule_probs"] for r in records}
    llm_field = {str(r["cur"]): r["final_probs"] for r in records}
    viz.action_field_plot(rule_field, llm_field, [r["cur"] for r in records],
                           "Rule-based policy", "LLM judge (with instruction, converged)",
                           os.path.join(FIG_DIR, "e2_action_field.png"))

    return records, summary


if __name__ == "__main__":
    run()
