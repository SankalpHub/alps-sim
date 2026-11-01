"""Phase C: compare schedulers on harder workloads.

Runs CFS, SRPT, alps_levels, and alps_score on five workloads:
  baseline, drift, overlap, heavy_tail, new_func.
Prints one table per workload. Utilization target is approximate on the
extended workloads (drift/new_func change the mix over time), so offered
load is reported per workload.
"""
import sched_sim as S


SCHEDS = {
    "cfs":         lambda n: S.CFS(n),
    "srpt":        lambda n: S.SRPT(n),
    "alps_levels": lambda n: S.ALPS(n),
    "alps_score":  lambda n: S.ALPSScore(n, tau=100),
}
METRICS = ("short_avg", "short_p99", "long_avg", "avg_tat")


def run_one(base, cores, sched):
    ts = [S.Task(t.tid, t.func, t.arrival, t.burst) for t in base]
    return S.summarize(sched.name, S.simulate(ts, sched, cores))


def print_table(name, results):
    print(f"\n=== {name} ===")
    print(f"{'sched':>18}" + "".join(f"{m:>14}" for m in METRICS))
    for sname, r in results.items():
        row = f"{sname:>18}"
        for m in METRICS:
            row += f"{r[m]:>14.1f}"
        print(row)


def main():
    cores, duration, util, seed = 4, 60, 0.95, 1

    workloads = [
        ("baseline (easy)",       S.gen_workload(duration, cores, util, seed)),
        ("drift api x3 @30s",     S.gen_workload(duration, cores, util, seed,
                                                 drift={"api": (30, 3.0)})),
        ("overlap api sigma=1.0", S.gen_workload(duration, cores, util, seed,
                                                 sigma={"api": 1.0})),
        ("heavy tail etl+ml",     S.gen_workload(duration, cores, util, seed,
                                                 heavy_tail=["etl", "ml"])),
        ("new func @30s",         S.gen_workload(duration, cores, util, seed,
                                                 new_func=(30, "cold", 100, 0.05))),
    ]

    print(f"cores={cores}, duration={duration}s, target util={util}, seed={seed}")
    print("Offered load = sum(burst) / (cores * duration_ms). It may differ")
    print("from the target on drift/new_func workloads by design.\n")

    for wname, base in workloads:
        offered = sum(t.burst for t in base) / (cores * duration * 1000)
        print(f"\n### {wname}: {len(base)} tasks, offered load = {offered:.3f}")
        results = {name: run_one(base, cores, make(cores))
                   for name, make in SCHEDS.items()}
        print_table(wname, results)


if __name__ == "__main__":
    main()