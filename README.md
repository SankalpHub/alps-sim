# ALPS-Inspired Adaptive Priority Scheduler — Simulator

OS course project. User-space discrete-event simulator comparing CFS, SRPT (oracle),
level-based ALPS, and a continuous-score ALPS variant on bursty serverless workloads.

## Environment
Python 3.13, venv at `.venv/`. Install: `pip install -r requirements.txt`

## Files
| File | Purpose |
|---|---|
| sched_sim.py | Simulator core, 5 schedulers, workload generator, metrics |
| run_experiments.py | 3 utils x N seeds x 6 scheds sweep → results.csv + results.png |
| run_workloads.py | Phase C: baseline / drift / overlap / heavy-tail / new-func |
| run_sweeps.py | Phase D: parameter sweeps (tau, hyst, window, levels, ...) |
| run_switch.py | Phase E: context-switch cost sensitivity |
| diagnose.py | Phase B: per-function p99, learned-level log, F5-style plot |
| collect_raw.py | Phase G: gather per-function / per-task / switch CSVs |
| make_figures.py | Phase G: regenerate F1-F9 from results/*.csv |
| tests/test_sim.py | 11 unit tests: SRPT traces, CFS fairness, invariants, switch cost |

## Key commands
- Smoke test:  `python sched_sim.py --cores 4 --util 0.95 --duration 30 --seed 1`
- Full sweep:  `python run_experiments.py --seeds 10 --duration 60`
- Tests:       `pytest -q`
- Figures:     `python collect_raw.py && python make_figures.py`

## Key results (util=0.95, 10 seeds)
- CFS short_avg = 55.0 ± 11.0, short_p99 = 299 ± 68
- SRPT (oracle) short_avg = 25.6 ± 0.6, short_p99 = 87.6 ± 2.6
- alps_levels short_avg = 43.5 ± 6.2, short_p99 = 739 ± 384  ← unstable learner
- alps_score short_avg = 29.6 ± 2.6, short_p99 = 173 ± 70  ← near-oracle at window=8000

## Findings (for the report)
1. Level-based ALPS reproduces paper trend but has a tail problem on mid-length functions (api).
2. Phase B diagnosis: learned quantile boundaries flip 20-30 times per run for auth/api/resize.
3. Phase D sweep: tau / hysteresis / predictor / slice-cap have ~zero effect. **Learning window is first-order** — window=8000ms makes alps_score match SRPT within 4ms on short p99.
4. Phase C: alps_score is robust to drift / overlap / heavy-tail / cold-start. alps_levels *loses to CFS* on heavy-tail workloads.
5. Phase E: under switch_cost=2ms, CFS collapses (short_avg 51→308, switches/s 460→721 — positive-feedback loop). SRPT invariant. ALPS's 40ms slice cap is what saves it.