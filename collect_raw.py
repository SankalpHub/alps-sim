"""Phase G data collection for figures.

Produces (all in results/):
  per_function_095.csv — per-function stats, 4 scheds x 10 seeds @ util=0.95
  tasks_baseline.csv   — per-task records, 4 scheds x 3 seeds @ 0.95
  tasks_drift.csv      — same, drift workload (api x3 @ 30s)
  switch_cost.csv      — 4 scheds x 3 costs x 5 seeds @ 0.95
  level_log.csv        — learned levels over time, one run (for F5)

Run:  python collect_raw.py     (takes ~2-3 minutes)
"""
import csv
import os
import statistics
from concurrent.futures import ProcessPoolExecutor

import sched_sim as S

SEEDS = list(range(1, 11))
SEEDS_PER_TASK = [1, 2, 3]
SEEDS_SWITCH = [1, 2, 3, 4, 5]
CORES, DURATION, UTIL = 4, 60, 0.95
SCHEDS = ["cfs", "srpt", "alps_levels", "alps_score"]
SWITCH_COSTS = [0, 1, 2]


def make_sched(name, n):
    return {"cfs": S.CFS,
            "srpt": S.SRPT,
            "alps_levels": S.ALPS,
            "alps_score": lambda m: S.ALPSScore(m, tau=100)}[name](n)


def per_func_job(args):
    sname, seed = args
    base = S.gen_workload(DURATION, CORES, UTIL, seed)
    ts = [S.Task(t.tid, t.func, t.arrival, t.burst) for t in base]
    tasks = S.simulate(ts, make_sched(sname, CORES), CORES)
    rows = []
    for f in S.FUNCS:
        xs = [t.finish - t.arrival for t in tasks if t.func == f]
        if not xs:
            continue
        rows.append({"sched": sname, "func": f, "seed": seed,
                     "count": len(xs), "avg": statistics.mean(xs),
                     "p99": S.pct(xs, 99)})
    return rows


def per_task_job(args):
    sname, seed, workload = args
    drift = {"api": (30, 3.0)} if workload == "drift" else None
    base = S.gen_workload(DURATION, CORES, UTIL, seed, drift=drift)
    ts = [S.Task(t.tid, t.func, t.arrival, t.burst) for t in base]
    tasks = S.simulate(ts, make_sched(sname, CORES), CORES)
    return [{"sched": sname, "seed": seed, "workload": workload,
             "tid": t.tid, "func": t.func, "arrival": t.arrival,
             "burst": t.burst, "finish": t.finish,
             "turnaround": t.finish - t.arrival,
             "slowdown": (t.finish - t.arrival) / t.burst}
            for t in tasks]


def switch_job(args):
    sname, cost, seed = args
    base = S.gen_workload(DURATION, CORES, UTIL, seed)
    sched = make_sched(sname, CORES)
    ts = [S.Task(t.tid, t.func, t.arrival, t.burst) for t in base]
    tasks = S.simulate(ts, sched, CORES, switch_cost=cost)
    r = S.summarize(sname, tasks)
    return {"sched": sname, "cost": cost, "seed": seed,
            "short_avg": r["short_avg"], "short_p99": r["short_p99"],
            "avg_tat": r["avg_tat"], "switches": sched.n_switches}


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"saved {path}  ({len(rows)} rows)")


def level_log():
    """Run alps_levels once and dump (time_s, func, level) rows."""
    cores = CORES
    base = S.gen_workload(DURATION, cores, UTIL, seed=1)
    ts = [S.Task(t.tid, t.func, t.arrival, t.burst) for t in base]
    sched = S.ALPS(cores)
    S.simulate(ts, sched, cores)
    rows = []
    for now, prio in sched.level_log:
        for f, lv in prio.items():
            rows.append({"time_s": now / 1000.0, "func": f, "level": lv})
    return rows


def main():
    os.makedirs("results", exist_ok=True)

    per_func_jobs = [(s, seed) for s in SCHEDS for seed in SEEDS]
    per_task_jobs = ([(s, seed, "baseline") for s in SCHEDS
                      for seed in SEEDS_PER_TASK] +
                     [(s, seed, "drift") for s in SCHEDS
                      for seed in SEEDS_PER_TASK])
    switch_jobs = [(s, c, seed) for s in SCHEDS for c in SWITCH_COSTS
                   for seed in SEEDS_SWITCH]

    print(f"per-function: {len(per_func_jobs)} jobs")
    print(f"per-task:     {len(per_task_jobs)} jobs")
    print(f"switch:       {len(switch_jobs)} jobs")

    with ProcessPoolExecutor() as ex:
        per_func_rows = [r for rows in ex.map(per_func_job, per_func_jobs)
                         for r in rows]
        per_task_rows = [r for rows in ex.map(per_task_job, per_task_jobs)
                         for r in rows]
        switch_rows = list(ex.map(switch_job, switch_jobs))

    write_csv("results/per_function_095.csv", per_func_rows,
              ["sched", "func", "seed", "count", "avg", "p99"])

    tasks_baseline = [r for r in per_task_rows if r["workload"] == "baseline"]
    tasks_drift = [r for r in per_task_rows if r["workload"] == "drift"]
    task_fields = ["sched", "seed", "workload", "tid", "func", "arrival",
                   "burst", "finish", "turnaround", "slowdown"]
    write_csv("results/tasks_baseline.csv", tasks_baseline, task_fields)
    write_csv("results/tasks_drift.csv", tasks_drift, task_fields)

    write_csv("results/switch_cost.csv", switch_rows,
              ["sched", "cost", "seed", "short_avg", "short_p99",
               "avg_tat", "switches"])

    print("level log (single run)")
    write_csv("results/level_log.csv", level_log(),
              ["time_s", "func", "level"])


if __name__ == "__main__":
    main()