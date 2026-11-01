#!/usr/bin/env python3
"""
Runs sched_sim.py schedulers over several seeds and utilization levels.
Prints mean +- std per (utilization, scheduler), saves results.csv and a plot.

Usage:  python run_experiments.py --seeds 5 --duration 60
Needs sched_sim.py in the same folder.
"""
import argparse, csv, statistics, sys
from concurrent.futures import ProcessPoolExecutor
from sched_sim import (gen_workload, simulate, summarize, Task,
                       CFS, SRPT, ALPS, ALPSAging, ALPSScore)

METRICS = ["avg_tat", "p99_tat", "short_avg", "short_p99",
           "long_avg", "long_max_slowdown", "avg_slowdown", "jain_slowdown"]


def make_scheds(n):
    # Add or edit variants here (name -> scheduler instance)
    return {
        "cfs": CFS(n),
        "srpt": SRPT(n),
        "alps_levels": ALPS(n),
        "alps_score": ALPSScore(n, tau=100),
        "alps_score_noage": ALPSScore(n, tau=1e12),   # ablation: no wait-time term
        "alps_score_ewma": ALPSScore(n, tau=100, predictor="ewma"),
    }


def run_one(args):
    util, seed, cores, duration = args
    base = gen_workload(duration, cores, util, seed)
    rows = []
    for name, s in make_scheds(cores).items():
        ts = [Task(t.tid, t.func, t.arrival, t.burst) for t in base]
        r = summarize(name, simulate(ts, s, cores))
        r.update(util=util, seed=seed)
        rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--utils", type=float, nargs="+", default=[0.7, 0.85, 0.95])
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--out", default="results.csv")
    ap.add_argument("--plot", default="results.png")
    a = ap.parse_args()

    jobs = [(u, s, a.cores, a.duration) for u in a.utils for s in range(1, a.seeds + 1)]
    with ProcessPoolExecutor() as ex:
        all_rows = [r for rows in ex.map(run_one, jobs) for r in rows]

    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["util", "seed", "sched"] + METRICS)
        w.writeheader()
        for r in all_rows:
            w.writerow({k: r[k] for k in w.fieldnames})

    # aggregate
    agg = {}
    for r in all_rows:
        agg.setdefault((r["util"], r["sched"]), []).append(r)

    shown = ["short_avg", "short_p99", "long_avg", "avg_tat", "long_max_slowdown"]
    for u in a.utils:
        print(f"\n=== utilization {u}  ({a.seeds} seeds, {a.duration}s each) ===")
        print(f"{'sched':>15}" + "".join(f"{m:>22}" for m in shown))
        for name in make_scheds(a.cores):
            rs = agg[(u, name)]
            cells = []
            for m in shown:
                xs = [r[m] for r in rs]
                sd = statistics.pstdev(xs) if len(xs) > 1 else 0
                cells.append(f"{statistics.mean(xs):>12.1f} ±{sd:>7.1f}")
            print(f"{name:>15}" + "".join(f"{c:>22}" for c in cells))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = list(make_scheds(a.cores))
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for ax, m in zip(axes, ["short_avg", "short_p99", "long_avg"]):
            for name in names:
                ys = [statistics.mean(r[m] for r in agg[(u, name)]) for u in a.utils]
                ax.plot(a.utils, ys, marker="o", label=name)
            ax.set_title(m + " (ms)")
            ax.set_xlabel("target utilization")
        axes[0].legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(a.plot, dpi=150)
        print("\nsaved", a.out, "and", a.plot)
    except ImportError:
        print("\nsaved", a.out, "(matplotlib missing, no plot)")


if __name__ == "__main__":
    main()
