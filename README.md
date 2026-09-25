# Reward Function Experiments (E1-E4)

Four small, real (not mocked) experiments comparing a hand-written
rule-based reward, a real LLM-as-judge reward (Claude Haiku 4.5), and a
reward network distilled from the LLM's judgments, on a shared 2-D
gridworld navigation task.

Full write-up: [`report/report.pdf`](report/report.pdf) (4-page IEEE-format
report, built from `report/report.tex`).

## What's here

- **E1** — rule-based policy vs. a single-shot LLM judge, no instruction.
- **E2** — same, plus a natural-language instruction and iterative
  refinement (the judge is told how far its action distribution is from
  the reference policy and re-asked, up to 4 rounds).
- **E3** — same protocol as E2, but the reference policy is a small
  REINFORCE-trained neural network (72% success rate) instead of the
  near-optimal rule.
- **E4** — a small MLP regressor distilled from every (situation, LLM
  rubric) pair produced in E1-E3, evaluated against fresh real LLM calls
  on held-out situations.

Every LLM call across E1-E4 is a real call to the Anthropic API
(`claude-haiku-4-5-20251001`) with a structured tool-use schema for the
rubric + action-probability output — there is no mocked judge in this
codebase.

## Layout

```
src/                  shared modules: gridworld, rule policy, NN policy,
                       LLM judge, reward network, plotting, metrics
experiments/          e1_rule_llm.py, e2_rule_llm_nl.py,
                       e3_nn_llm_nl.py, e4_reward_model.py
results/<e1..e4>/     records.json + summary.json from each run
report/               report.tex, references.bib, report.pdf, figures/
run_all.py            runs E1-E4 in order
```

## Running it

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...
python run_all.py            # or run experiments/e*.py individually
```

Each experiment script is self-contained: it samples a handful of grid
situations, computes the reference policy's action-probabilities and
rubric, calls the real LLM judge, saves `results/<exp>/{records,summary}.json`,
and regenerates the figures under `report/figures/` that the report cites.

Rebuilding the report after re-running the experiments:

```bash
cd report && latexmk -pdf report.tex
```

## Environment

One 8x8 grid, walls forming two one-cell-gap bands, a target object
("sofa") and two distractors ("chair", "cube"), four cardinal actions
{up, right, down, left}. See `src/gridworld.py`. This is a deliberately
lightweight stand-in for the fuller MiniGrid/PPO/sentence-transformer
pipeline described in the companion project draft — see the report's
Limitations section for why, and what a larger-scale follow-up would add
back.

## Results summary

| Exp. | Reference policy | Instruction | Mean iters. | Final JS div. | Argmax agreement | Rubric MAE |
|---|---|---|---|---|---|---|
| E1 | Rule-based | No | 1 (single-shot) | 0.151 | 0.80 | 0.097 |
| E2 | Rule-based | Yes | 1.3 / 4 | 0.019 | 1.00 | 0.052 |
| E3 | Neural network (72% success) | Yes | 1.8 / 4 | 0.043 | 0.70 | 0.118 |
| E4 | Distilled reward net vs. LLM | Yes (train.) | - | 0.188 | 0.25 | 0.076 |

E4's reward network also runs ~2300x faster than a real LLM judge call
(1.1ms vs. 2.56s) while reproducing its rubric scores closely but its
specific best-action choice only 25% of the time — see the report's
Discussion for why that gap matters.
