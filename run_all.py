"""Run E1-E4 in order. Each experiment writes its records/summary JSON to
results/<exp>/ and its figures to report/figures/. Requires ANTHROPIC_API_KEY
in the environment (every experiment makes real LLM judge calls)."""
import os
import runpy

HERE = os.path.dirname(__file__)

if __name__ == "__main__":
    assert os.environ.get("ANTHROPIC_API_KEY"), "Set ANTHROPIC_API_KEY before running."
    for script in ["e1_rule_llm.py", "e2_rule_llm_nl.py", "e3_nn_llm_nl.py", "e4_reward_model.py"]:
        path = os.path.join(HERE, "experiments", script)
        print(f"\n=== running {script} ===")
        runpy.run_path(path, run_name="__main__")
