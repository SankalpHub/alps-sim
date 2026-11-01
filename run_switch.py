"""Phase E: context-switch cost sensitivity.

Runs cfs, srpt, alps_levels, alps_score on the default workload
with switch_cost = 0, 1, 2 ms, 5 seeds each.
"""
import statistics
from concurrent.futures import ProcessPoolExecutor
import sched_sim as S

SEEDS = [1, 2, 3, 4, 5]
DURATION, CORES, UTIL = 60, 4, 0.95
SCHEDS = ["cfs", "srpt", "alps_levels", "alps_score"]
COSTS = [0, 1, 2]


def make_sched(name, n):
    if name == "cfs":
        return S.CFS(n)
    if name == "srpt":
        return S.SRPT(n)
    if name == "alps_levels":
        return S.ALPS(n)
    if name == "alps_score":
        return S.ALPSScore(n, tau=100)
    raise ValueError(name)


def run_one(job):
    sname, cost, seed = job
    base = S.gen_workload(DURATION, CORES, UTIL, seed)
    sched = make_sched(sname, CORES)
    ts = [S.Task(t.tid, t.func, t.arrival, t.burst) for t in base]
    tasks = S.simulate(ts, sched, CORES, switch_cost=cost)
    r = S.summarize(sname, tasks)
    r["n_switches"] = sched.n_switches
    r["switches_per_s"] = sched.n_switches / DURATION
    return (sname, cost, seed, r)


def main():
    jobs = [(s, c, seed) for s in SCHEDS for c in COSTS for seed in SEEDS]
    print(f"{len(jobs)} jobs (4 scheds x 3 costs x 5 seeds)\n")
    with ProcessPoolExecutor() as ex:
        results = list(ex.map(run_one, jobs))

    agg = {}
    for sname, cost, seed, r in results:
        agg.setdefault((sname, cost), []).append(r)

    metrics = ["short_avg", "short_p99", "avg_tat", "long_max_slowdown",
               "switches_per_s"]
    labels = ["short_avg", "short_p99", "avg_tat", "long_max_sd", "sw/s"]

    for cost in COSTS:
        print(f"\n=== switch_cost = {cost} ms ===")
        print(f"{'sched':>16}" + "".join(f"{l:>16}" for l in labels))
        for sname in SCHEDS:
            rs = agg[(sname, cost)]
            cells = []
            for m in metrics:
                xs = [r[m] for r in rs]
                mean = statistics.mean(xs)
                sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
                cells.append(f"{mean:>8.1f} ±{sd:>5.1f}")
            print(f"{sname:>16}" + " ".join(cells))


if __name__ == "__main__":
    main()