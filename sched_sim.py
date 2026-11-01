#!/usr/bin/env python3
"""
Mini scheduler simulator for serverless-style workloads.
Inspired by ALPS (USENIX ATC'24). Time unit: 1 tick = 1 ms.

Schedulers:
  cfs        - vruntime-based fair scheduler (simplified CFS)
  srpt       - oracle preemptive shortest-remaining-time-first (best case for short jobs)
  alps       - learned per-function priority levels + CFS vruntime tie-break + slice cap
  alps_aging - alps + our improvement: aging (anti-starvation) and selectable predictor

Run:  python sched_sim.py --cores 4 --util 0.85 --duration 30 --seed 1
"""
import argparse, math, random, statistics
from dataclasses import dataclass


# ----------------------------------------------------------------------------
# Task and workload
# ----------------------------------------------------------------------------
@dataclass
class Task:
    tid: int
    func: str
    arrival: int
    burst: int                 # total CPU demand (ms)
    remaining: int = 0
    vruntime: float = 0.0
    enq_time: int = 0          # when it last entered the run queue
    level: int = 0             # effective priority level chosen at pick time
    finish: int = -1

    def __post_init__(self):
        self.remaining = self.burst


# name: (mean CPU ms, share of invocations)
FUNCS = {
    "thumb":  (10,   0.35),
    "auth":   (25,   0.25),
    "api":    (60,   0.20),
    "resize": (300,  0.10),
    "ml":     (1200, 0.07),
    "etl":    (2500, 0.03),
}


def gen_workload(duration_s, cores, util, seed, burst_period_s=2.0, hi=1.6, lo=0.4):
    """Bursty arrivals: alternating high/low rate segments, lognormal runtimes."""
    rng = random.Random(seed)
    names = list(FUNCS)
    shares = [FUNCS[n][1] for n in names]
    mean_service = sum(FUNCS[n][0] * FUNCS[n][1] for n in names)
    base_rate = util * cores / mean_service            # arrivals per ms
    tasks, tid, t = [], 0, 0.0
    horizon = duration_s * 1000
    while t < horizon:
        seg = int(t // (burst_period_s * 1000))
        rate = base_rate * (hi if seg % 2 == 0 else lo)
        t += rng.expovariate(rate)
        if t >= horizon:
            break
        f = rng.choices(names, shares)[0]
        mean = FUNCS[f][0]
        burst = max(1, int(rng.lognormvariate(math.log(mean) - 0.045, 0.3)))
        tasks.append(Task(tid, f, int(t), burst))
        tid += 1
    return tasks


# ----------------------------------------------------------------------------
# Schedulers
# ----------------------------------------------------------------------------
class Sched:
    name = "base"
    MIN_GRAN, LATENCY = 3, 24      # CFS-like constants (ms)

    def __init__(self, ncores):
        self.ncores = ncores
        self.q = []
        self.min_vr = 0.0

    # --- hooks -------------------------------------------------------------
    def add(self, t, now, new=False):
        if new:                                    # new task starts at min_vruntime
            t.vruntime = self.min_vr
        t.enq_time = now
        self.q.append(t)

    def key(self, t, now):
        return t.vruntime

    def pick(self, now):
        t = min(self.q, key=lambda x: self.key(x, now))
        self.q.remove(t)
        self.min_vr = max(self.min_vr, t.vruntime)
        t.level = 0
        return t

    def cfs_slice(self, nr_running):
        return max(self.MIN_GRAN, math.ceil(self.LATENCY * self.ncores / max(1, nr_running)))

    def slice(self, t, nr_running):
        return self.cfs_slice(nr_running)

    def preempts(self, curr, now):
        return False

    def on_ran(self, t, dt):
        t.vruntime += dt

    def on_finish(self, t, now):
        pass

    def tick(self, now):
        pass


class CFS(Sched):
    name = "cfs"


class SRPT(Sched):
    name = "srpt"

    def key(self, t, now):
        return t.remaining                         # oracle: knows remaining time

    def slice(self, t, nr_running):
        return 10**9

    def preempts(self, curr, now):
        return bool(self.q) and min(x.remaining for x in self.q) < curr.remaining


class ALPS(Sched):
    """
    Frontend (learning): every `learn_every` ms, estimate each function's CPU time
    from a recent window of finished invocations, rank functions by estimate and
    bucket them into `levels` priority levels (quantiles over invocation share).
    Backend (enforcement): pick lowest (level, vruntime); preempt on better level;
    time slice = min(CFS slice, cap).
    """
    name = "alps"

    def __init__(self, ncores, levels=4, cap=40, window=2000, learn_every=500,
                 predictor="mean", aging_ms=None, ewma_alpha=0.3):
        super().__init__(ncores)
        self.levels, self.cap = levels, cap
        self.window, self.learn_every = window, learn_every
        self.predictor, self.aging_ms, self.alpha = predictor, aging_ms, ewma_alpha
        self.hist = {}                # func -> list[(finish_time, cpu_ms)]
        self.ewma = {}                # func -> float
        self.prio = {}                # func -> level
        self.next_learn = learn_every
        self.est_now = {}             # func -> estimated CPU ms (latest learning step)

    # ---- frontend ---------------------------------------------------------
    def on_finish(self, t, now):
        self.hist.setdefault(t.func, []).append((now, t.burst))
        e = self.ewma.get(t.func)
        self.ewma[t.func] = t.burst if e is None else self.alpha * t.burst + (1 - self.alpha) * e

    def _estimate(self, f, samples):
        if self.predictor == "ewma":
            return self.ewma[f]
        xs = sorted(c for _, c in samples)
        if self.predictor == "p90":
            return xs[min(len(xs) - 1, int(0.9 * len(xs)))]
        return sum(xs) / len(xs)

    def tick(self, now):
        if now < self.next_learn:
            return
        self.next_learn = now + self.learn_every
        est, cnt = {}, {}
        for f, h in self.hist.items():
            recent = [(ts, c) for ts, c in h if ts >= now - self.window]
            self.hist[f] = recent
            if recent:
                est[f], cnt[f] = self._estimate(f, recent), len(recent)
        if not est:
            return
        self.est_now = dict(est)                    # continuous per-function estimates
        total, cum, new = sum(cnt.values()), 0, {}
        for f in sorted(est, key=est.get):          # shortest functions first
            new[f] = min(self.levels - 1, int(cum / total * self.levels))
            cum += cnt[f]
        self.prio = new

    # ---- backend ----------------------------------------------------------
    def _level(self, t, now):
        lvl = self.prio.get(t.func, self.levels // 2)
        if self.aging_ms:
            lvl = max(0, lvl - int((now - t.enq_time) / self.aging_ms))
        return lvl

    def key(self, t, now):
        return (self._level(t, now), t.vruntime)

    def pick(self, now):
        t = super().pick(now)
        t.level = self._level(t, now)
        return t

    def slice(self, t, nr_running):
        return min(self.cfs_slice(nr_running), self.cap)

    def preempts(self, curr, now):
        return bool(self.q) and min(self._level(x, now) for x in self.q) < curr.level


class ALPSAging(ALPS):
    name = "alps_aging"

    def __init__(self, ncores, aging_ms=150, predictor="ewma", **kw):
        super().__init__(ncores, aging_ms=aging_ms, predictor=predictor, **kw)


class ALPSScore(ALPS):
    """
    Improvement: replace strict priority levels with a continuous score.
        score = predicted_runtime / (1 + wait/tau) ** power      (lower runs first)
    Short functions still win, but any task's score shrinks the longer it waits,
    so mid-length functions (e.g. 60 ms) are no longer starved by bursts of
    shorter ones. `hyst` (<1) requires a clearly better score before preempting,
    to avoid thrashing.
    """
    name = "alps_score"

    def __init__(self, ncores, tau=100, power=1.0, hyst=0.7, **kw):
        super().__init__(ncores, **kw)
        self.tau, self.power, self.hyst = tau, power, hyst

    def _est(self, f):
        if f in self.est_now:
            return self.est_now[f]
        if self.est_now:
            v = sorted(self.est_now.values())
            return v[len(v) // 2]
        return 100.0

    def _score(self, t, now):
        return self._est(t.func) / (1 + (now - t.enq_time) / self.tau) ** self.power

    def key(self, t, now):
        return (self._score(t, now), t.vruntime)

    def pick(self, now):
        t = Sched.pick(self, now)
        t.level = self._est(t.func)        # running task's score (it is not waiting)
        return t

    def preempts(self, curr, now):
        return bool(self.q) and min(self._score(x, now) for x in self.q) < self.hyst * curr.level



# ----------------------------------------------------------------------------
# Simulator core (tick-based, global run queue, N cores)
# ----------------------------------------------------------------------------
def simulate(tasks, sched, ncores):
    tasks = sorted(tasks, key=lambda t: t.arrival)
    i, now, done = 0, 0, 0
    cores = [None] * ncores
    left = [0] * ncores
    while done < len(tasks):
        while i < len(tasks) and tasks[i].arrival <= now:
            sched.add(tasks[i], now, new=True)
            i += 1
        sched.tick(now)

        # fast-forward when fully idle
        if not sched.q and all(c is None for c in cores):
            if i < len(tasks):
                now = tasks[i].arrival
                continue

        for c in range(ncores):
            cur = cores[c]
            if cur is not None and (left[c] <= 0 or sched.preempts(cur, now)):
                sched.add(cur, now)                     # preempted: back to queue
                cores[c] = cur = None
            if cur is None and sched.q:
                busy = sum(1 for x in cores if x is not None)
                cur = sched.pick(now)
                cores[c] = cur
                left[c] = sched.slice(cur, len(sched.q) + busy + 1)

        for c in range(ncores):                          # run 1 ms
            t = cores[c]
            if t is None:
                continue
            t.remaining -= 1
            left[c] -= 1
            sched.on_ran(t, 1)
            if t.remaining == 0:
                t.finish = now + 1
                sched.on_finish(t, now + 1)
                cores[c] = None
                done += 1
        now += 1
    return tasks


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))] if xs else float("nan")


def jain(xs):
    return (sum(xs) ** 2) / (len(xs) * sum(x * x for x in xs)) if xs else float("nan")


def summarize(name, tasks):
    tat = [t.finish - t.arrival for t in tasks]
    sd = [(t.finish - t.arrival) / t.burst for t in tasks]
    short = [t for t in tasks if t.burst < 100]
    long_ = [t for t in tasks if t.burst >= 1000]
    s_tat = [t.finish - t.arrival for t in short]
    l_tat = [t.finish - t.arrival for t in long_]
    l_sd = [(t.finish - t.arrival) / t.burst for t in long_]
    return {
        "sched": name,
        "avg_tat": statistics.mean(tat),
        "p99_tat": pct(tat, 99),
        "short_avg": statistics.mean(s_tat) if s_tat else float("nan"),
        "short_p99": pct(s_tat, 99),
        "long_avg": statistics.mean(l_tat) if l_tat else float("nan"),
        "long_max_slowdown": max(l_sd) if l_sd else float("nan"),
        "avg_slowdown": statistics.mean(sd),
        "jain_slowdown": jain(sd),
    }


def print_table(rows):
    cols = ["sched", "avg_tat", "p99_tat", "short_avg", "short_p99",
            "long_avg", "long_max_slowdown", "avg_slowdown", "jain_slowdown"]
    print("  ".join(f"{c:>17}" for c in cols))
    for r in rows:
        print("  ".join(f"{r[c]:>17.2f}" if c != "sched" else f"{r[c]:>17}" for c in cols))


def maybe_plot(rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return
    metrics = ["short_avg", "short_p99", "long_avg", "avg_tat"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 3.5))
    for ax, m in zip(axes, metrics):
        ax.bar([r["sched"] for r in rows], [r[m] for r in rows])
        ax.set_title(m + " (ms)")
        ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print("saved", path)


# ----------------------------------------------------------------------------
def make_scheds(n):
    return [CFS(n), SRPT(n), ALPS(n), ALPSAging(n), ALPSScore(n)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cores", type=int, default=4)
    ap.add_argument("--util", type=float, default=0.85)
    ap.add_argument("--duration", type=int, default=30, help="seconds of arrivals")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--plot", default=None, help="save bar chart to this png path")
    a = ap.parse_args()

    base = gen_workload(a.duration, a.cores, a.util, a.seed)
    print(f"{len(base)} invocations, {a.cores} cores, target util {a.util}\n")
    rows = []
    for s in make_scheds(a.cores):
        # fresh copies so every scheduler sees an identical workload
        ts = [Task(t.tid, t.func, t.arrival, t.burst) for t in base]
        rows.append(summarize(s.name, simulate(ts, s, a.cores)))
    print_table(rows)
    if a.plot:
        maybe_plot(rows, a.plot)
