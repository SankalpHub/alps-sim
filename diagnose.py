"""Phase B diagnosis: is the level-based failure caused by learned-level
instability, or by strict priority itself?

Produces:
  - diagnose_levels.png: learned level of each function over time (one run)
  - two tables: per-function p99 and average turnaround for
    cfs / srpt / alps_levels / alps_fixed4 / alps_fixed6 / alps_score
  - a level-change summary so you can see which functions flip levels

Run from the project root: python diagnose.py
"""
import statistics
from collections import Counter

import sched_sim as S


# True-runtime mappings from the runbook, page 69.
FIXED_4 = {"thumb": 0, "auth": 1, "api": 2, "resize": 3, "ml": 3, "etl": 3}
FIXED_6 = {"thumb": 0, "auth": 1, "api": 2, "resize": 3, "ml": 4, "etl": 5}


class FixedLevelALPS(S.ALPS):
    """ALPS with the learner disabled and levels hard-coded."""
    name = "alps_fixed"

    def __init__(self, ncores, mapping, levels, **kw):
        super().__init__(ncores, levels=levels, **kw)
        self.prio = dict(mapping)

    def tick(self, now):
        # Never re-learn. Levels stay at the fixed mapping.
        pass


def run_one(sched, base, cores):
    """Copy tasks (fresh state), run scheduler, return finished tasks."""
    ts = [S.Task(t.tid, t.func, t.arrival, t.burst) for t in base]
    return S.simulate(ts, sched, cores)


def per_func_p99(tasks, f):
    xs = [t.finish - t.arrival for t in tasks if t.func == f]
    return S.pct(xs, 99) if xs else float("nan")


def per_func_avg(tasks, f):
    xs = [t.finish - t.arrival for t in tasks if t.func == f]
    return statistics.mean(xs) if xs else float("nan")


def main():
    cores, duration, util, seed = 4, 60, 0.95, 1
    order = ["thumb", "auth", "api", "resize", "ml", "etl"]

    base = S.gen_workload(duration, cores, util, seed)
    print(f"{len(base)} tasks, {cores} cores, util {util}, seed {seed}, "
          f"{duration}s\n")

    # 1) Learned levels over time (alps_levels run)
    alps = S.ALPS(cores)
    out_alps = run_one(alps, base, cores)
    print(f"ALPS learner ran {len(alps.level_log)} times "
          f"(every {alps.learn_every} ms)")

    # 2) Same workload, other schedulers
    runs = [
        ("cfs", run_one(S.CFS(cores), base, cores)),
        ("srpt", run_one(S.SRPT(cores), base, cores)),
        ("alps_levels", out_alps),
        ("alps_fixed4", run_one(FixedLevelALPS(cores, FIXED_4, levels=4),
                                base, cores)),
        ("alps_fixed6", run_one(FixedLevelALPS(cores, FIXED_6, levels=6),
                                base, cores)),
        ("alps_score", run_one(S.ALPSScore(cores, tau=100), base, cores)),
    ]

    # 3) Per-function p99
    print("\nPer-function p99 turnaround (ms):")
    print(f"{'func':>8}" + "".join(f"{name:>14}" for name, _ in runs))
    for f in order:
        row = f"{f:>8}"
        for _, tasks in runs:
            row += f"{per_func_p99(tasks, f):>14.0f}"
        print(row)

    # 4) Per-function average
    print("\nPer-function average turnaround (ms):")
    print(f"{'func':>8}" + "".join(f"{name:>14}" for name, _ in runs))
    for f in order:
        row = f"{f:>8}"
        for _, tasks in runs:
            row += f"{per_func_avg(tasks, f):>14.0f}"
        print(row)

    # 5) How often do learned levels change?
    if alps.level_log:
        changes = {f: 0 for f in order}
        per_level = {f: Counter() for f in order}
        prev = {}
        for _, prio in alps.level_log:
            for f in order:
                lv = prio.get(f)
                if lv is None:
                    continue
                per_level[f][lv] += 1
                if f in prev and prev[f] != lv:
                    changes[f] += 1
                prev[f] = lv
        print("\nLearned-level changes over the run (more = more unstable):")
        for f in order:
            seen = dict(sorted(per_level[f].items()))
            print(f"  {f:>7}: {changes[f]:>3} changes   levels seen {seen}")

    # 6) Plot levels over time
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 4))
        times = [now / 1000 for now, _ in alps.level_log]
        for f in order:
            ys, last = [], None
            for _, prio in alps.level_log:
                lv = prio.get(f)
                if lv is not None:
                    last = lv
                ys.append(last)
            ax.step(times, ys, where="post", label=f)
        ax.set_xlabel("time (s)")
        ax.set_ylabel("learned level (0 = highest priority)")
        ax.set_title(f"Learned levels over time (util {util}, seed {seed})")
        ax.legend(loc="center right", fontsize=8)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("diagnose_levels.png", dpi=150)
        print("\nsaved diagnose_levels.png")
    except ImportError:
        print("\nmatplotlib missing; skipping plot")


if __name__ == "__main__":
    main()