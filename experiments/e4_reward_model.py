"""E4 - replace the LLM judge with a learned reward network. The network
is distilled from every (state, rubric) pair the real LLM judge produced
in E1-E3, then evaluated at inference time (no further API calls) against
a held-out set of situations that get a fresh, real LLM judge call as the
comparison target.
"""
from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np

import gridworld as g
import rule_policy as rp
import llm_judge as lj
import reward_model as rm
import metrics as met
import viz
import state_sampler as ss

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "e4")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "report", "figures")
BASE = os.path.join(os.path.dirname(__file__), "..", "results")
N_TEST = 8
RUBRIC_FIELDS = rm.RUBRIC_FIELDS


def load_training_records():
    train = []
    with open(os.path.join(BASE, "e1", "records.json")) as f:
        for r in json.load(f):
            train.append({"cell": tuple(r["cur"]), "obstacle_hits": r["obstacle_hits"],
                           "steps_taken": r["steps_taken"], "judge_out": r["llm_rubric"] |
                           {"reached_target": float(tuple(r["cur"]) == g.TARGET)}})
    for exp in ["e2", "e3"]:
        with open(os.path.join(BASE, exp, "records.json")) as f:
            for r in json.load(f):
                for rubric in r["rubric_per_iter"]:
                    train.append({"cell": tuple(r["cur"]), "obstacle_hits": r["obstacle_hits"],
                                   "steps_taken": r["steps_taken"], "judge_out": rubric})
    return train


def run():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    train_records = load_training_records()
    X, Y = rm.build_dataset(train_records)
    print(f"training reward network on {len(X)} (state, LLM-rubric) pairs distilled from E1-E3")

    net = rm.RewardNetwork(seed=0).fit(X, Y)

    # a few-fold check on the training distribution itself (in-sample fit quality)
    Y_hat_train = net.model.predict(X)
    train_mae = float(np.mean(np.abs(Y_hat_train - Y)))
    print(f"in-sample rubric MAE: {train_mae:.4f}")

    test_cells = ss.sample_cells(N_TEST, seed=99)
    rng = np.random.default_rng(31)

    records = []
    for start in test_cells:
        state = ss.partial_rollout(rp.action_probs, start, rng)
        cur, steps, hits = state["cur"], state["steps_taken"], state["obstacle_hits"]
        rule_probs = rp.action_probs(cur)

        t0 = time.time()
        llm_out = lj.judge(cur, g.TARGET, instruction=None, obstacle_hits=hits, steps_taken=steps)
        llm_latency = time.time() - t0
        llm_probs = lj.probs_array(llm_out)

        t0 = time.time()
        rm_rubric = net.predict_rubric(cur, hits, steps)
        rm_probs = net.action_probs(cur, hits, steps)
        rm_latency = time.time() - t0

        rec = {
            "cell": cur, "steps_taken": steps, "obstacle_hits": hits,
            "llm_rubric": {k: llm_out[k] for k in RUBRIC_FIELDS},
            "rm_rubric": rm_rubric,
            "rubric_mae_rm_vs_llm": met.rubric_mae(rm_rubric, llm_out, RUBRIC_FIELDS),
            "rule_probs": rule_probs.tolist(), "llm_probs": llm_probs.tolist(),
            "rm_probs": rm_probs.tolist(),
            "js_div_rm_vs_llm": met.js_divergence(rm_probs, llm_probs),
            "js_div_rm_vs_rule": met.js_divergence(rm_probs, rule_probs),
            "js_div_llm_vs_rule": met.js_divergence(llm_probs, rule_probs),
            "argmax_agree_rm_vs_llm": met.agreement_rate(rm_probs, llm_probs),
            "llm_latency_s": llm_latency, "rm_latency_s": rm_latency,
        }
        records.append(rec)
        print(f"cell={cur} MAE(rm,llm)={rec['rubric_mae_rm_vs_llm']:.3f} "
              f"JS(rm,llm)={rec['js_div_rm_vs_llm']:.3f} JS(llm,rule)={rec['js_div_llm_vs_rule']:.3f} "
              f"latency llm={llm_latency:.2f}s rm={rm_latency*1000:.2f}ms")

    summary = {
        "n_train_pairs": len(X), "in_sample_rubric_mae": train_mae, "n_test_cells": N_TEST,
        "mean_rubric_mae_rm_vs_llm": float(np.mean([r["rubric_mae_rm_vs_llm"] for r in records])),
        "mean_js_div_rm_vs_llm": float(np.mean([r["js_div_rm_vs_llm"] for r in records])),
        "mean_js_div_rm_vs_rule": float(np.mean([r["js_div_rm_vs_rule"] for r in records])),
        "mean_js_div_llm_vs_rule": float(np.mean([r["js_div_llm_vs_rule"] for r in records])),
        "argmax_agreement_rm_vs_llm": float(np.mean([r["argmax_agree_rm_vs_llm"] for r in records])),
        "mean_llm_latency_s": float(np.mean([r["llm_latency_s"] for r in records])),
        "mean_rm_latency_s": float(np.mean([r["rm_latency_s"] for r in records])),
        "speedup_x": float(np.mean([r["llm_latency_s"] for r in records]) /
                            max(np.mean([r["rm_latency_s"] for r in records]), 1e-9)),
    }
    print("SUMMARY", json.dumps(summary, indent=2))

    with open(os.path.join(OUT_DIR, "records.json"), "w") as f:
        json.dump(records, f, indent=2, default=str)
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # --- figures ---
    labels = [str(r["cell"]) for r in records]
    viz.bar_overlay(labels, {
        "LLM judge closeness": [r["llm_rubric"]["closeness"] for r in records],
        "reward network closeness": [r["rm_rubric"]["closeness"] for r in records],
    }, "closeness score", "E4: LLM judge vs. distilled reward network (held-out cells)",
        os.path.join(FIG_DIR, "e4_rubric_closeness.png"))

    viz.bar_overlay(["mean latency (log ms)"], {
        "LLM judge": [np.log10(summary["mean_llm_latency_s"] * 1000)],
        "reward network": [np.log10(summary["mean_rm_latency_s"] * 1000)],
    }, "log10(latency in ms)", "E4: inference latency, LLM judge vs. reward network",
        os.path.join(FIG_DIR, "e4_latency.png"))

    llm_field = {str(r["cell"]): r["llm_probs"] for r in records}
    rm_field = {str(r["cell"]): r["rm_probs"] for r in records}
    viz.action_field_plot(llm_field, rm_field, [r["cell"] for r in records],
                           "LLM judge", "Distilled reward network",
                           os.path.join(FIG_DIR, "e4_action_field.png"))

    return records, summary


if __name__ == "__main__":
    run()
