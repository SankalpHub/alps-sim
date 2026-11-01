"""Supplementary Phase D experiment.

Q1: does window=8000 also fix the LEVELS approach (not just score)?
Q2: does window=8000 hurt adaptability on the Phase C drift workload?

40 runs (2 scheds x 2 windows x 2 workloads x 5 seeds), ~2-3 min.
"""
import statistics
from concurrent.futures import ProcessPoolExecutor
import sched_sim as S

SEEDS = [1, 2, 3, 4, 5]
DURATION, CORES, UTIL = 60, 4, 0.95


def run_one(job):
    sname, window, workload, seed = job
    drift = {"api": (30, 3.0)} if workload == "drift" else None
    base = S.gen_workload(DURATION, CORES, UTIL, seed, drift=drift)
    sched = (S.ALPS(CORES, window=window) if sname == "alps_levels"
             else S.ALPSScore(CORES, window=window))
    ts = [S.Task(t.tid, t.func, t.arrival, t.burst) for t in base]
    r = S.summarize(sname, S.simulate(ts, sched, CORES))
    return (sname, window, workload, seed, r)


def show(rows, workload, title):
    print(f"\n=== {title} ===")
    print(f"{'config':<32} {'short_avg':>16} {'short_p99':>16} {'avg_tat':>16}")
    entries = [(cfg, rs) for cfg, rs in rows.items() if cfg[2] == workload]
    entries.sort(key=lambda x: (x[0][0], x[0][1]))
    for (sname, window, _), rs in entries:
        cells = []
        for m in ("short_avg", "short_p99", "avg_tat"):
            xs = [r[m] for r in rs]
            mean = statistics.mean(xs)
            sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
            cells.append(f"{mean:>9.1f} ±{sd:>4.1f}")
        print(f"{sname} win={window}".ljust(32) + " ".join(cells))


def main():
    jobs = [(s, w, wl, seed)
            for s in ("alps_levels", "alps_score")
            for w in (2000, 8000)
            for wl in ("baseline", "drift")
            for seed in SEEDS]
    print(f"{len(jobs)} jobs (2 scheds x 2 windows x 2 workloads x 5 seeds)")
    with ProcessPoolExecutor() as ex:
        results = list(ex.map(run_one, jobs))

    rows = {}
    for sname, window, workload, seed, r in results:
        rows.setdefault((sname, window, workload), []).append(r)

    show(rows, "baseline", "baseline / util=0.95")
    show(rows, "drift", "drift (api x3 @ 30s) / util=0.95")

    print("\nFor reference (from Phase 0 / Phase C):")
    print("  CFS baseline:      short_avg=51.1, short_p99=261.0")
    print("  SRPT baseline:     short_avg=25.7, short_p99=86.2")
    print("  alps_levels win=2000 (baseline): short_avg=43.7, short_p99=710.6")
    print("  alps_score  win=2000 (baseline): short_avg=29.3, short_p99=156.2")
    print("  alps_levels win=2000 (drift, Phase C seed 1): short_avg=50.0, short_p99=1662")


if __name__ == "__main__":
    main()