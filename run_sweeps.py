"""Phase D: one-factor-at-a-time parameter sweeps for Phase D.

Each sweep runs 5 seeds (1-5) at the default config (4 cores, util 0.95,
default workload, 60 s), varying one parameter.

Usage:
  python run_sweeps.py --sweep tau          # first sweep, ~30-50s
  python run_sweeps.py --sweep all          # everything, ~20-30 min
  python run_sweeps.py --sweep grids        # only the two interaction grids

Writes results/sweep_{name}.csv and prints one table per sweep.
"""
import argparse, csv, os, statistics
from concurrent.futures import ProcessPoolExecutor

import sched_sim as S

SEEDS = [1, 2, 3, 4, 5]
DURATION = 60
DEFAULT_CORES = 4
DEFAULT_UTIL = 0.95
METRICS = ["short_avg", "short_p99", "long_avg", "avg_tat",
           "long_max_slowdown", "avg_slowdown", "jain_slowdown"]
SHOWN = ["short_avg", "short_p99", "avg_tat", "jain_slowdown"]


# Each config: (label, sched_name, sched_kwargs, env_overrides)
SWEEPS = {
    "tau": [
        ("tau=20",  "alps_score", {"tau": 20},    {}),
        ("tau=100", "alps_score", {"tau": 100},   {}),
        ("tau=500", "alps_score", {"tau": 500},   {}),
        ("tau=inf", "alps_score", {"tau": 1e12},  {}),
    ],
    "hyst": [
        ("hyst=0.5", "alps_score", {"hyst": 0.5}, {}),
        ("hyst=0.7", "alps_score", {"hyst": 0.7}, {}),
        ("hyst=1.0", "alps_score", {"hyst": 1.0}, {}),
    ],
    "power": [
        ("power=1", "alps_score", {"power": 1.0}, {}),
        ("power=2", "alps_score", {"power": 2.0}, {}),
    ],
    "predictor": [
        ("mean",  "alps_score", {"predictor": "mean"}, {}),
        ("ewma",  "alps_score", {"predictor": "ewma"}, {}),
        ("p90",   "alps_score", {"predictor": "p90"},  {}),
    ],
    "window": [
        ("win=500",  "alps_score", {"window": 500},  {}),
        ("win=2000", "alps_score", {"window": 2000}, {}),
        ("win=8000", "alps_score", {"window": 8000}, {}),
    ],
    "interval": [
        ("every=100",  "alps_score", {"learn_every": 100},  {}),
        ("every=500",  "alps_score", {"learn_every": 500},  {}),
        ("every=2000", "alps_score", {"learn_every": 2000}, {}),
    ],
    "levels": [
        ("levels=2", "alps_levels", {"levels": 2}, {}),
        ("levels=4", "alps_levels", {"levels": 4}, {}),
        ("levels=6", "alps_levels", {"levels": 6}, {}),
        ("levels=8", "alps_levels", {"levels": 8}, {}),
    ],
    "slice": [
        ("cap=10",  "alps_score", {"cap": 10},  {}),
        ("cap=40",  "alps_score", {"cap": 40},  {}),
        ("cap=100", "alps_score", {"cap": 100}, {}),
    ],
    "cores": [
        ("cores=2", "alps_score", {}, {"cores": 2}),
        ("cores=4", "alps_score", {}, {"cores": 4}),
        ("cores=8", "alps_score", {}, {"cores": 8}),
    ],
    "util": [
        ("util=0.5",  "alps_score", {}, {"util": 0.5}),
        ("util=0.7",  "alps_score", {}, {"util": 0.7}),
        ("util=0.85", "alps_score", {}, {"util": 0.85}),
        ("util=0.95", "alps_score", {}, {"util": 0.95}),
        ("util=1.05", "alps_score", {}, {"util": 1.05}),
    ],
}


def grids():
    out = {}
    out["window_x_interval"] = []
    for w in (500, 2000, 8000):
        for iv in (100, 500, 2000):
            out["window_x_interval"].append(
                (f"win={w},every={iv}", "alps_score",
                 {"window": w, "learn_every": iv}, {})
            )
    out["tau_x_predictor"] = []
    for tau in (20, 100, 500):
        for pred in ("mean", "ewma", "p90"):
            out["tau_x_predictor"].append(
                (f"tau={tau},{pred}", "alps_score",
                 {"tau": tau, "predictor": pred}, {})
            )
    return out


def make_sched(name, kwargs, ncores):
    if name == "alps_score":
        return S.ALPSScore(ncores, **kwargs)
    if name == "alps_levels":
        return S.ALPS(ncores, **kwargs)
    if name == "cfs":
        return S.CFS(ncores)
    if name == "srpt":
        return S.SRPT(ncores)
    raise ValueError(f"unknown sched {name}")


def run_one(args):
    label, sname, skwargs, env, seed = args
    cores = env.get("cores", DEFAULT_CORES)
    util = env.get("util", DEFAULT_UTIL)
    base = S.gen_workload(DURATION, cores, util, seed)
    sched = make_sched(sname, skwargs, cores)
    ts = [S.Task(t.tid, t.func, t.arrival, t.burst) for t in base]
    r = S.summarize(label, S.simulate(ts, sched, cores))
    return (label, seed, cores, util, r)


def run_sweep(name, configs, out_dir="results"):
    jobs = [(label, sname, skwargs, env, seed)
            for (label, sname, skwargs, env) in configs
            for seed in SEEDS]
    with ProcessPoolExecutor() as ex:
        results = list(ex.map(run_one, jobs))

    agg = {}
    for label, seed, cores, util, r in results:
        agg.setdefault(label, []).append(r)

    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, f"sweep_{name}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sweep", "label", "seed", "cores", "util"] + METRICS)
        for label, seed, cores, util, r in results:
            w.writerow([name, label, seed, cores, util] +
                       [r[m] for m in METRICS])

    print(f"\n=== sweep: {name} ({len(configs)} configs x {len(SEEDS)} seeds) ===")
    header = f"{'config':>22}"
    for m in SHOWN:
        header += f"{m:>22}"
    print(header)
    for label, _, _, _ in configs:
        row = f"{label:>22}"
        for m in SHOWN:
            xs = [r[m] for r in agg[label]]
            mean = statistics.mean(xs)
            sd = statistics.stdev(xs) if len(xs) > 1 else 0.0
            row += f"{mean:>13.1f} ±{sd:>7.1f}"
        print(row)
    print(f"saved {csv_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", required=True,
                    choices=list(SWEEPS) + ["grids", "all"])
    a = ap.parse_args()

    if a.sweep == "all":
        for name in SWEEPS:
            run_sweep(name, SWEEPS[name])
        for gname, gconfigs in grids().items():
            run_sweep(gname, gconfigs)
    elif a.sweep == "grids":
        for gname, gconfigs in grids().items():
            run_sweep(gname, gconfigs)
    else:
        run_sweep(a.sweep, SWEEPS[a.sweep])


if __name__ == "__main__":
    main()