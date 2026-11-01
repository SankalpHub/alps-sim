"""Phase A unit tests for sched_sim.py. Run from project root with: pytest -q"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sched_sim import (Task, CFS, SRPT, ALPS, ALPSAging, ALPSScore,
                       gen_workload, simulate)


def test_srpt_preempts_on_arrival():
    ts = [Task(0, "A", 0, 100), Task(1, "B", 10, 20)]
    by = {t.tid: t for t in simulate(ts, SRPT(1), 1)}
    assert by[1].finish == 30 and by[0].finish == 120


def test_srpt_orders_shortest_first():
    ts = [Task(0, "A", 0, 30), Task(1, "B", 0, 10), Task(2, "C", 0, 20)]
    by = {t.tid: t.finish for t in simulate(ts, SRPT(1), 1)}
    assert by == {1: 10, 2: 30, 0: 60}


def test_cfs_shares_fairly():
    ts = [Task(0, "A", 0, 100), Task(1, "B", 0, 100)]
    out = simulate(ts, CFS(1), 1)
    f = sorted(t.finish for t in out)
    assert f[1] == 200 and f[1] - f[0] <= 24


def test_invariants_all_schedulers():
    base = gen_workload(10, 2, 0.8, seed=3)
    for make in (CFS, SRPT, ALPS, ALPSAging, ALPSScore):
        ts = [Task(t.tid, t.func, t.arrival, t.burst) for t in base]
        out = simulate(ts, make(2), 2)
        assert all(t.remaining == 0 for t in out)
        assert all(t.finish >= t.arrival + t.burst for t in out)
        assert sum(t.burst for t in out) <= 2 * max(t.finish for t in out)


def test_deterministic():
    a = gen_workload(5, 2, 0.8, seed=7)
    b = gen_workload(5, 2, 0.8, seed=7)
    assert [(t.arrival, t.burst, t.func) for t in a] == \
           [(t.arrival, t.burst, t.func) for t in b]


def test_work_conservation_single_core():
    ts = [Task(i, "A", 0, 20) for i in range(5)]
    out = simulate(ts, CFS(1), 1)
    assert max(t.finish for t in out) == 100


def test_work_conservation_multi_core():
    ts = [Task(i, "A", 0, 25) for i in range(8)]
    out = simulate(ts, CFS(4), 4)
    assert max(t.finish for t in out) == 50


def test_offered_load_close_to_target():
    cores, duration_s, util, seed = 4, 30, 0.85, 1
    ts = gen_workload(duration_s, cores, util, seed)
    offered = sum(t.burst for t in ts) / (cores * duration_s * 1000)
    print(f"\n  offered load = {offered:.3f}  (target {util:.3f}, "
          f"n = {len(ts)})")
    assert abs(offered - util) < 0.15


def test_alps_learner_orders_by_runtime():
    runtimes = {"short": 10, "medium": 50, "long": 200}
    ts = []
    for i in range(60):
        f = ["short", "medium", "long"][i % 3]
        ts.append(Task(i, f, i * 100, runtimes[f]))

    s = ALPS(1)
    simulate(ts, s, 1)

    assert "short" in s.est_now and "medium" in s.est_now and "long" in s.est_now
    assert s.est_now["short"] < s.est_now["medium"] < s.est_now["long"]
    assert s.prio["short"] <= s.prio["medium"] <= s.prio["long"]
    assert s.prio["short"] < s.prio["long"]
    
    
def test_switch_cost_zero_matches_default():
    """switch_cost=0 must produce byte-identical results to no kwarg."""
    base = gen_workload(5, 2, 0.8, seed=3)

    def run(extra):
        out = []
        for make in (CFS, SRPT, ALPS, ALPSScore):
            ts = [Task(t.tid, t.func, t.arrival, t.burst) for t in base]
            r = simulate(ts, make(2), 2, **extra)
            out.append([(t.tid, t.finish) for t in r])
        return out

    assert run({}) == run({"switch_cost": 0})


def test_switch_cost_produces_switches():
    """switch_cost>0 must record switches on a busy workload."""
    base = gen_workload(5, 2, 0.8, seed=3)
    s = CFS(2)
    ts = [Task(t.tid, t.func, t.arrival, t.burst) for t in base]
    simulate(ts, s, 2, switch_cost=1)
    assert s.n_switches > 0