# Reward Function Experiments (E1-E4)

Four small, real (not mocked) experiments comparing a hand-written
rule-based reward, a real LLM-as-judge reward (Claude Haiku 4.5), and a
reward network distilled from the LLM's judgments, on a shared 2-D
gridworld navigation task.

Two write-ups of the same underlying study:
- [`report/report.pdf`](report/report.pdf) — a 4-page IEEE-format
  technical report, built from `report/report.tex`.
- [`paper/paper.pdf`](paper/paper.pdf) — a fuller 5-page paper (Abstract,
  Introduction, Previous Work, Novelty, Contribution, Experimental
  Setup, Results and Analysis, Conclusion), built from `paper/paper.tex`.

## What's here

- **E1** — rule-based policy vs. a single-shot LLM judge, no instruction.
  64 situations.
- **E2** — same, plus a natural-language instruction and iterative
  refinement (the judge is told how far its action distribution is from
  the reference policy and re-asked, up to 4 rounds). 64 situations.
- **E3** — same protocol as E2, but the reference policy is a small
  REINFORCE-trained neural network (72% success rate) instead of the
  near-optimal rule. 64 situations.
- **E4** — a small MLP regressor distilled from every (situation, LLM
  rubric) pair produced in E1-E3 (258 pairs), evaluated against 48
  fresh, held-out real LLM calls.

Every LLM call across E1-E4 is a real call to the Anthropic API
(`claude-haiku-4-5-20251001`) with a structured tool-use schema for the
rubric + action-probability output — there is no mocked judge in this
codebase. E1-E3 each evaluate every one of the map's 16 eligible grid
cells, replicated 3-4 times with independent stochastic rollouts (rather
than a handful of once-sampled cells), and every summary number is
reported with a 95% confidence interval — see `src/state_sampler.py` and
`src/metrics.py`.

## Layout

```
src/                  shared modules: gridworld, rule policy, NN policy,
                       LLM judge, reward network, plotting, metrics
experiments/          e1_rule_llm.py, e2_rule_llm_nl.py,
                       e3_nn_llm_nl.py, e4_reward_model.py
results/<e1..e4>/     records.json + summary.json from each run
report/               4-page IEEE report: report.tex, references.bib, report.pdf, figures/
paper/                5-page paper: paper.tex, references.bib, paper.pdf, figures/
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

n=64 situations for E1-E3, n=48 held-out situations for E4 (258 training
pairs pooled from E1-E3); brackets are 95% confidence intervals.

| Exp. | Reference policy | Instruction | Mean iters. | Final JS div. | Argmax agreement | Rubric MAE |
|---|---|---|---|---|---|---|
| E1 | Rule-based | No | 1 (single-shot) | 0.095 [0.06,0.13] | 0.78 [0.67,0.87] | 0.053 [0.04,0.06] |
| E2 | Rule-based | Yes | 1.34 / 4 | 0.017 [0.01,0.02] | 1.00 [0.94,1.00] | 0.056 [0.04,0.07] |
| E3 | Neural network (72% success) | Yes | 1.69 / 4 | 0.011 [0.01,0.01] | 0.86 [0.75,0.92] | 0.114 [0.09,0.14] |
| E4 | Distilled reward net vs. LLM | Yes (train.) | - | 0.232 [0.20,0.26] | 0.48 [0.34,0.62] | 0.058 [0.05,0.07] |

E4's reward network also runs ~1800x faster (median) than a real LLM
judge call (2.4ms vs. 2.44s) while reproducing its rubric scores closely
but its specific best-action choice only 48% of the time — up from 25%
in a smaller 41-pair pilot run, but still well short of parity — see the
paper's Discussion for why that gap matters.
