"""Phase G: regenerate figures F1-F9 from results/*.csv and results.csv.

One command:  python make_figures.py

Reads:
  results.csv                          (F1, F2, F6)
  results/per_function_095.csv         (F3)
  results/tasks_baseline.csv           (F4, F7)
  results/level_log.csv                (F5)
  results/switch_cost.csv              (F8)
  results/tasks_drift.csv              (F9)

Writes figures/F1..F9.png (200 dpi) and figures/F1..F9.pdf.
"""
import csv
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats as sps

os.makedirs("figures", exist_ok=True)

COLORS = {"cfs": "#1f77b4", "srpt": "#2ca02c",
          "alps_levels": "#d62728", "alps_score": "#9467bd",
          "alps_score_noage": "#8c564b", "alps_score_ewma": "#e377c2"}


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def ci95(xs):
    n = len(xs)
    m = statistics.mean(xs)
    if n < 2:
        return m, 0.0
    s = statistics.stdev(xs)
    return m, sps.t.ppf(0.975, n - 1) * s / (n ** 0.5)


def save(fig, name):
    for ext in ("png", "pdf"):
        path = f"figures/{name}.{ext}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
    print(f"saved figures/{name}.png")


# ---------------- F1 / F2: turnaround vs utilization --------------------
def f1_f2():
    rows = read_csv("results.csv")
    utils = sorted({float(r["util"]) for r in rows})
    scheds = ["cfs", "srpt", "alps_levels", "alps_score"]

    for metric, name, label in [
            ("short_avg", "F1", "Short avg turnaround (ms)"),
            ("short_p99", "F2", "Short p99 turnaround (ms)")]:
        fig, ax = plt.subplots(figsize=(6, 4))
        for s in scheds:
            means, errs = [], []
            for u in utils:
                xs = [float(r[metric]) for r in rows
                      if float(r["util"]) == u and r["sched"] == s]
                m, h = ci95(xs)
                means.append(m)
                errs.append(h)
            ax.errorbar(utils, means, yerr=errs, marker="o", capsize=4,
                        label=s, color=COLORS[s])
        ax.set_xlabel("target utilization")
        ax.set_ylabel(label)
        ax.set_title(f"{name}: {label.split('(')[0].strip()} vs utilization")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        save(fig, name)
        plt.close(fig)


# ---------------- F3: per-function p99 at 0.95 --------------------------
def f3():
    rows = read_csv("results/per_function_095.csv")
    funcs = list(dict.fromkeys(r["func"] for r in rows))
    scheds = ["cfs", "alps_levels", "alps_score"]
    width = 0.8 / len(scheds)

    fig, ax = plt.subplots(figsize=(8, 4))
    for i, s in enumerate(scheds):
        means, errs = [], []
        for f in funcs:
            xs = [float(r["p99"]) for r in rows
                  if r["sched"] == s and r["func"] == f]
            m, h = ci95(xs)
            means.append(m)
            errs.append(h)
        x = [j + (i - (len(scheds) - 1) / 2) * width for j in range(len(funcs))]
        ax.bar(x, means, width=width, yerr=errs, capsize=3,
               label=s, color=COLORS[s], edgecolor="black", linewidth=0.4)
    ax.set_xticks(range(len(funcs)))
    ax.set_xticklabels(funcs)
    ax.set_ylabel("p99 turnaround (ms)")
    ax.set_title("F3: per-function p99 turnaround at util=0.95 (10 seeds)")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend(fontsize=8)
    save(fig, "F3")
    plt.close(fig)


# ---------------- F4: CDF of short turnaround at 0.95 -------------------
def f4():
    rows = read_csv("results/tasks_baseline.csv")
    scheds = ["cfs", "srpt", "alps_levels", "alps_score"]

    fig, ax = plt.subplots(figsize=(6, 4))
    for s in scheds:
        xs = sorted(float(r["turnaround"]) for r in rows
                    if r["sched"] == s and float(r["burst"]) < 100)
        if not xs:
            continue
        ys = [(i + 1) / len(xs) for i in range(len(xs))]
        ax.plot(xs, ys, label=s, color=COLORS[s], linewidth=1.5)
    ax.set_xlim(0, 800)
    ax.set_xlabel("short-task turnaround (ms)")
    ax.set_ylabel("CDF")
    ax.set_title("F4: CDF of short-function turnaround at util=0.95")
    ax.axhline(0.99, linestyle="--", color="gray", linewidth=0.7)
    ax.text(10, 0.995, "p99", color="gray", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    save(fig, "F4")
    plt.close(fig)


# ---------------- F5: learned levels over time --------------------------
def f5():
    rows = read_csv("results/level_log.csv")
    funcs = list(dict.fromkeys(r["func"] for r in rows))
    fig, ax = plt.subplots(figsize=(8, 4))
    for f in funcs:
        ts = [(float(r["time_s"]), int(r["level"])) for r in rows
              if r["func"] == f]
        ax.step([t for t, _ in ts], [l for _, l in ts],
                where="post", label=f, linewidth=1.2)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("learned level (0 = highest priority)")
    ax.set_title("F5: learned levels over time (alps_levels, util=0.95, seed=1)")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="center right")
    save(fig, "F5")
    plt.close(fig)


# ---------------- F6: ablation bars -------------------------------------
def f6():
    rows = read_csv("results.csv")
    scheds = ["alps_levels", "alps_score_noage", "alps_score_ewma",
              "alps_score"]
    metric = "short_p99"
    means, errs = [], []
    for s in scheds:
        xs = [float(r[metric]) for r in rows
              if float(r["util"]) == 0.95 and r["sched"] == s]
        m, h = ci95(xs)
        means.append(m)
        errs.append(h)

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [COLORS.get(s, "#888888") for s in scheds]
    ax.bar(range(len(scheds)), means, yerr=errs, capsize=4,
           color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xticks(range(len(scheds)))
    ax.set_xticklabels(scheds, rotation=15)
    ax.set_ylabel("short p99 (ms)")
    ax.set_title("F6: ablation - short p99 at util=0.95 (10 seeds, 95% CI)")
    ax.grid(True, alpha=0.3, axis="y")
    save(fig, "F6")
    plt.close(fig)


# ---------------- F7: slowdown vs CPU demand scatter --------------------
def f7():
    rows = read_csv("results/tasks_baseline.csv")
    scheds = ["cfs", "srpt", "alps_levels", "alps_score"]
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.5), sharey=True)
    for ax, s in zip(axes, scheds):
        pts = [(float(r["burst"]), float(r["slowdown"])) for r in rows
               if r["sched"] == s]
        if not pts:
            continue
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, s=3, alpha=0.35, color=COLORS[s])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("CPU demand (ms)")
        ax.set_title(s)
        ax.grid(True, alpha=0.3, which="both")
        ax.axhline(1.0, color="gray", linewidth=0.7, linestyle="--")
    axes[0].set_ylabel("slowdown (turnaround / CPU)")
    fig.suptitle("F7: slowdown vs CPU demand at util=0.95", y=1.02)
    save(fig, "F7")
    plt.close(fig)


# ---------------- F8: switch cost sensitivity ---------------------------
def f8():
    rows = read_csv("results/switch_cost.csv")
    costs = sorted({int(r["cost"]) for r in rows})
    scheds = ["cfs", "srpt", "alps_levels", "alps_score"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, metric, ylabel in [
            (axes[0], "short_avg", "short avg turnaround (ms)"),
            (axes[1], "switches", "switches per run")]:
        for s in scheds:
            means, errs = [], []
            for c in costs:
                xs = [float(r[metric]) for r in rows
                      if int(r["cost"]) == c and r["sched"] == s]
                m, h = ci95(xs)
                means.append(m)
                errs.append(h)
            ax.errorbar(costs, means, yerr=errs, marker="o", capsize=4,
                        label=s, color=COLORS[s])
        ax.set_xlabel("switch_cost (ms)")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("F8: context-switch cost sensitivity (util=0.95, 5 seeds)", y=1.02)
    save(fig, "F8")
    plt.close(fig)


# ---------------- F9: drift effect on short tasks -----------------------
def f9():
    rows = read_csv("results/tasks_drift.csv")
    scheds = ["cfs", "srpt", "alps_levels", "alps_score"]
    window = 5000    # 5 s arrival windows

    fig, ax = plt.subplots(figsize=(8, 4))
    for s in scheds:
        pts = [(int(r["arrival"]), float(r["turnaround"]))
               for r in rows if r["sched"] == s
               and r["func"] in ("thumb", "auth")]
        if not pts:
            continue
        max_t = max(t for t, _ in pts)
        xs, ys = [], []
        for lo in range(0, max_t, window):
            hi = lo + window
            vals = [tr for t, tr in pts if lo <= t < hi]
            if len(vals) >= 10:
                xs.append(lo / 1000.0 + window / 2000.0)
                ys.append(statistics.mean(vals))
        ax.plot(xs, ys, marker="o", markersize=4, label=s, color=COLORS[s])
    ax.axvline(30, color="black", linestyle="--", linewidth=1)
    ax.text(30.2, ax.get_ylim()[1] * 0.95, "drift begins",
            fontsize=8, color="black")
    ax.set_xlabel("arrival time (s)")
    ax.set_ylabel("thumb+auth avg turnaround (ms)")
    ax.set_title("F9: drift workload - thumb+auth short tasks (5 s arrival windows)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    save(fig, "F9")
    plt.close(fig)


def main():
    f1_f2()
    f3()
    f4()
    f5()
    f6()
    f7()
    f8()
    f9()
    print("\nAll figures regenerated from CSVs in results/.")


if __name__ == "__main__":
    main()