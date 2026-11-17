"""IDSSE check of v16's mechanism: replay evidence with a SIMULATED broadcast replay flag.

    python ours/validate_replay.py        # needs the fold predictions cached by validate_gru_seeds.py

IDSSE has no broadcast, so the replay flag is simulated on each held-out match with the quality
measured on BAR-ATH without labels (broadcast_evidence.py + EM): replays cover ~69 % of dead frames
(stretches starting 1-3 s into a stoppage and sometimes running 0-2 s past the restart) and
close-ups cover ~5.6 % of live frames (random stretches, mean 4 s). The flag is only an input: the
v16 term is still estimated label-free by EM on the held-out match, exactly as on BAR-ATH.

Protocol, fixed BEFORE running: v15 fold models (trees + 3 GRUs, trained on the other 6 matches in
bench 2 + atd_stop), held-out match in atd_stop, control truth. v16 must beat v15 on mean spellF1
AND on at least 5/7 matches with the measured flag quality, AND lose at most 0.005 with a flag twice
as bad (55 % of dead frames, 11 % of live frames). Log -> ours/bench_results_replay.csv.
"""
import csv
import datetime
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import idsse_bench as bench  # noqa: E402
import v5_train as T  # noqa: E402
import v8  # noqa: E402
import gap_evidence as G  # noqa: E402
import broadcast_evidence as BE  # noqa: E402
from validate_gap import prepare_world, decode  # noqa: E402

FPS = 25
CACHE = os.path.join(bench.EXT, "seedcache")
QUALITY = {"measured": (0.69, 0.056), "twice_as_bad": (0.55, 0.11)}


def _runs(mask):
    d = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def simulate_flag(gt, graded, p_dead, p_alive, seed):
    """Replay/close-up flag with P(flag | dead) ~ p_dead and P(flag | alive) ~ p_alive."""
    n = len(gt)
    dead = (gt == "D") & graded
    alive = np.isin(gt, ["H", "A"]) & graded

    def make(q, rate):
        rng = np.random.default_rng(seed)
        f = np.zeros(n, bool)
        for a, b in _runs(dead):
            if b - a >= 2 * FPS and rng.random() < q:
                s = a + int(rng.integers(FPS, 3 * FPS + 1))
                e = min(n, b + int(rng.integers(0, 2 * FPS + 1)))
                f[s:e] = True
        starts = rng.random(n) < rate
        for t in np.flatnonzero(starts & alive):
            f[t:t + int(rng.exponential(4 * FPS)) + FPS] = True
        return f & graded

    lo, hi = 0.0, 1.0                                   # coverage of dead frames
    for _ in range(12):
        q = (lo + hi) / 2
        cov = make(q, 0.0)[dead].mean()
        lo, hi = (q, hi) if cov < p_dead else (lo, q)
    q = (lo + hi) / 2
    lo, hi = 0.0, 0.01                                  # close-up rate in live play
    for _ in range(14):
        r = (lo + hi) / 2
        fa = make(q, r)[alive].mean()
        lo, hi = (r, hi) if fa < p_alive else (lo, r)
    f = make(q, (lo + hi) / 2)
    return f, float(f[dead].mean()), float(f[alive].mean())


def main():
    rows = []
    for k, fold in enumerate(T.MATCHES):
        t = np.load(os.path.join(CACHE, "trees_%s.npz" % fold))
        gru = np.mean([np.load(os.path.join(CACHE, "gru_%s_s%d.npz" % (fold, s)))["P_atd_stop"] for s in (0, 1, 2)], 0)
        P = 0.5 * t["P_atd_stop"] + 0.5 * gru
        z = prepare_world(fold, k, "atd_stop")
        d = {key: z[key] for key in ("graded", "gt", "blocks", "no_players", "cost_in", "cost_out")}
        g = d["graded"]
        tr = [prepare_world(m, i, "legacy") for i, m in enumerate(T.MATCHES) if m != fold]
        d_tr = np.mean(np.concatenate([np.asarray(x["y"])[x["graded"] & (np.asarray(x["y"]) >= 0)] == 2 for x in tr]))
        prior = np.array([(1 - d_tr) / 2, (1 - d_tr) / 2, d_tr])
        alive = v8.em_alive(P, g, prior)
        truth = bench.align_to_control(d["gt"], 2.0)
        base = decode(P, d, t["sw_atd_stop"], alive)
        s0 = T.score(truth, base)[0]
        for qname, (pd_, pa) in QUALITY.items():
            flag, cov_d, cov_a = simulate_flag(d["gt"], g, pd_, pa, seed=100 + k)
            cat = np.where(flag, 2, 1).astype(np.int8)
            lr, info = BE.category_log_ratio(P, cat, g, prior)
            pred = decode(G.apply_log_ratio(P, lr), d, t["sw_atd_stop"], alive)
            s1 = T.score(truth, pred)[0]
            rec0 = float(np.mean(base[truth == "D"] == "D"))
            rec1 = float(np.mean(pred[truth == "D"] == "D"))
            rows.append(dict(match=fold, quality=qname, flag_dead=cov_d, flag_alive=cov_a, em_replay=info[2][1],
                             v15=s0, v16=s1, diff=s1 - s0, recallD_v15=rec0, recallD_v16=rec1))
            print("  %s %-12s flag covers %.0f %% dead / %.1f %% live | EM dead rate in flag %.2f | spellF1 v15 %.4f v16 %.4f (%+.4f) | recall D %.3f -> %.3f"
                  % (fold, qname, 100 * cov_d, 100 * cov_a, info[2][1], s0, s1, s1 - s0, rec0, rec1), flush=True)
    for qname in QUALITY:
        r = [x for x in rows if x["quality"] == qname]
        dd = [x["diff"] for x in r]
        print("%-12s v15 %.4f  v16 %.4f  diff %+.4f  wins %d/7" % (qname, np.mean([x["v15"] for x in r]),
                                                           np.mean([x["v16"] for x in r]), np.mean(dd), sum(x > 0 for x in dd)))
    log = os.path.join(HERE, "bench_results_replay.csv")
    with open(log, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date"] + list(rows[0].keys()))
        w.writeheader()
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        for r in rows:
            w.writerow(dict(date=stamp, **r))
    print("log -> %s" % os.path.relpath(log, ROOT))


if __name__ == "__main__" and len(sys.argv) == 1:
    main()


def add_only(P, cat, graded, prior, weight=0.5):
    """v19 variant: replay frames ADD dead-ball evidence; nothing is taken from other frames, and
    the alive rate is re-estimated (EM) after the evidence instead of being held fixed."""
    from broadcast_evidence import category_log_ratio
    lr, info = category_log_ratio(P, cat, graded, prior, weight)
    lr = np.where(cat == 2, np.maximum(lr, 0.0), 0.0)
    Q = G.apply_log_ratio(P, lr)
    return Q, v8.em_alive(Q, graded, prior), info


def main_add_only():
    out = []
    for k, fold in enumerate(T.MATCHES):
        t = np.load(os.path.join(CACHE, "trees_%s.npz" % fold))
        gru = np.mean([np.load(os.path.join(CACHE, "gru_%s_s%d.npz" % (fold, s)))["P_atd_stop"] for s in (0, 1, 2)], 0)
        P = 0.5 * t["P_atd_stop"] + 0.5 * gru
        z = prepare_world(fold, k, "atd_stop")
        d = {key: z[key] for key in ("graded", "gt", "blocks", "no_players", "cost_in", "cost_out")}
        g = d["graded"]
        tr = [prepare_world(m, i, "legacy") for i, m in enumerate(T.MATCHES) if m != fold]
        d_tr = np.mean(np.concatenate([np.asarray(x["y"])[x["graded"] & (np.asarray(x["y"]) >= 0)] == 2 for x in tr]))
        prior = np.array([(1 - d_tr) / 2, (1 - d_tr) / 2, d_tr])
        truth = bench.align_to_control(d["gt"], 2.0)
        base = decode(P, d, t["sw_atd_stop"], v8.em_alive(P, g, prior))
        s0 = T.score(truth, base)[0]
        for qname, (pd_, pa) in QUALITY.items():
            flag, cov_d, cov_a = simulate_flag(d["gt"], g, pd_, pa, seed=100 + k)
            Q, alive, _ = add_only(P, np.where(flag, 2, 1).astype(np.int8), g, prior)
            pred = decode(Q, d, t["sw_atd_stop"], alive)
            s1 = T.score(truth, pred)[0]
            out.append((qname, s1 - s0, float(np.mean(pred[truth == "D"] == "D")) - float(np.mean(base[truth == "D"] == "D"))))
            print("  %s %-12s spellF1 %+.4f | stoppage recall %+.3f" % (fold, qname, out[-1][1], out[-1][2]), flush=True)
    for qname in QUALITY:
        r = [x for x in out if x[0] == qname]
        print("ADD-ONLY %-12s spellF1 diff %+.4f wins %d/7 | stoppage recall %+.3f" % (qname, np.mean([x[1] for x in r]), sum(x[1] > 0 for x in r), np.mean([x[2] for x in r])))


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "add_only":
    main_add_only()
