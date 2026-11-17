"""Recalibrate the decoder's costs for the v15 probabilities, with nested leave-one-match-out.

    python ours/validate_decoder.py --fold J03WMX     # all configurations for one held-out match
    python ours/validate_decoder.py --summary         # nested selection + verdict

Why: the switch costs (H<->A 30 -> 8 at turnover peaks) and the stoppage entry/exit costs were
tuned for v8, a boosted-tree model. v15 averages the trees with 3 GRUs, whose probabilities are
much smoother, so the decoder is probably too reluctant to change state (v15 predicts 4.2-5.1
possessions per live minute on IDSSE where the control truth has 5-7). Recalibrating the decoder for
a new emission model is standard (False Five tuned the same costs on held-out spellF1).

Uses the out-of-fold v15 probabilities cached by validate_gru_seeds.py (models never saw the match);
decoding only. Protocol, fixed BEFORE running: grid of 30 configurations (C_HIGH in 15/20/25/30/40,
C_LOW in 4/8/12, stoppage entry/exit costs x0.6 or x1). NESTED: for each held-out match the
configuration is chosen on the other 6 (mean spellF1, control truth, atd_stop world) and scored on
the held-out one. Adoption: nested result beats v15 (30/8/x1) on mean spellF1 AND on >= 5/7 matches
in atd_stop; no loss in bench 2 (mean); and the final configuration keeps BAR-ATH's predicted
possessions per live minute within +15 % of v15's.
"""
import argparse
import csv
import glob
import itertools
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
from validate_gap import prepare_world, decode  # noqa: E402

CACHE = os.path.join(bench.EXT, "seedcache")
OUT = os.path.join(bench.EXT, "deccache")
GRID = [(ch, cl, s) for ch, cl, s in itertools.product((15, 20, 25, 30, 40), (4, 8, 12), (0.6, 1.0)) if cl < ch]
BASE = (30, 8, 1.0)
WORLDS = ("atd_stop", "legacy")


def run_fold(fold):
    k = T.MATCHES.index(fold)
    t = np.load(os.path.join(CACHE, "trees_%s.npz" % fold))
    tr = [prepare_world(m, i, "legacy") for i, m in enumerate(T.MATCHES) if m != fold]
    d_tr = np.mean(np.concatenate([np.asarray(x["y"])[x["graded"] & (np.asarray(x["y"]) >= 0)] == 2 for x in tr]))
    prior = np.array([(1 - d_tr) / 2, (1 - d_tr) / 2, d_tr])
    rows = []
    for w in WORLDS:
        gru = np.mean([np.load(os.path.join(CACHE, "gru_%s_s%d.npz" % (fold, s)))["P_" + w] for s in (0, 1, 2)], 0)
        P = 0.5 * t["P_" + w] + 0.5 * gru
        q = (v8.C_HIGH - t["sw_" + w]) / (v8.C_HIGH - v8.C_LOW)       # local max of P(turnover)
        z = prepare_world(fold, k, w)
        d0 = {key: z[key] for key in ("graded", "gt", "blocks", "no_players", "cost_in", "cost_out")}
        g = d0["graded"]
        alive = v8.em_alive(P, g, prior)
        ctrl, raw = bench.align_to_control(d0["gt"], 2.0), d0["gt"]
        for ch, cl, s in GRID:
            d = dict(d0, cost_in=d0["cost_in"] * s, cost_out=d0["cost_out"] * s)
            pred = decode(P, d, ch - (ch - cl) * q, alive)
            st = np.flatnonzero(np.r_[True, pred[g][1:] != pred[g][:-1]])
            lab = pred[g]
            ppm = sum(1 for a in st if lab[a] in "HA") / (np.isin(lab, ["H", "A"]).sum() / 1500)
            rows.append(dict(match=fold, world=w, ch=ch, cl=cl, s=s, ctrl=T.score(ctrl, pred)[0],
                             raw=T.score(raw, pred)[0], recall_D=float(np.mean(pred[ctrl == "D"] == "D")), ppm=ppm))
        print("fold %s world %s done" % (fold, w), flush=True)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "dec_%s.csv" % fold), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


def summary():
    rows = [r for f in sorted(glob.glob(os.path.join(OUT, "dec_*.csv"))) for r in csv.DictReader(open(f))]
    S = {}
    for r in rows:
        S[r["match"], r["world"], (int(r["ch"]), int(r["cl"]), float(r["s"])), "ctrl"] = float(r["ctrl"])
        S[r["match"], r["world"], (int(r["ch"]), int(r["cl"]), float(r["s"])), "raw"] = float(r["raw"])
        S[r["match"], r["world"], (int(r["ch"]), int(r["cl"]), float(r["s"])), "ppm"] = float(r["ppm"])
    M = T.MATCHES

    def mean(cfg, w, key, ms):
        return np.mean([S[m, w, cfg, key] for m in ms])
    print("top configurations, atd_stop, control truth (mean over 7):")
    ranked = sorted(GRID, key=lambda c: -mean(c, "atd_stop", "ctrl", M))
    for c in ranked[:6]:
        print("  C_HIGH %2d C_LOW %2d stop-costs x%.1f  atd_stop %.4f  legacy %.4f  raw(atd) %.4f  poss/min %.2f"
              % (c + (mean(c, "atd_stop", "ctrl", M), mean(c, "legacy", "ctrl", M), mean(c, "atd_stop", "raw", M), mean(c, "atd_stop", "ppm", M))))
    print("  base   30/8/x1.0              atd_stop %.4f  legacy %.4f  raw(atd) %.4f  poss/min %.2f"
          % (mean(BASE, "atd_stop", "ctrl", M), mean(BASE, "legacy", "ctrl", M), mean(BASE, "atd_stop", "raw", M), mean(BASE, "atd_stop", "ppm", M)))
    nested = {w: [] for w in WORLDS}
    picks = []
    for m in M:
        others = [x for x in M if x != m]
        best = max(GRID, key=lambda c: mean(c, "atd_stop", "ctrl", others))
        picks.append(best)
        for w in WORLDS:
            nested[w].append((S[m, w, best, "ctrl"], S[m, w, BASE, "ctrl"]))
    print("nested picks per held-out match:", picks)
    for w in WORLDS:
        a = np.array(nested[w])
        d = a[:, 0] - a[:, 1]
        print("NESTED %-8s v15 %.4f -> recalibrated %.4f  diff %+.4f  wins %d/7  per match %s"
              % (w, a[:, 1].mean(), a[:, 0].mean(), d.mean(), int((d > 0).sum()), np.round(d, 4).tolist()))
    final = ranked[0]
    print("final configuration (chosen on all 7): C_HIGH %d C_LOW %d stop-costs x%.1f" % final)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold")
    ap.add_argument("--summary", action="store_true")
    a = ap.parse_args()
    summary() if a.summary else run_fold(a.fold)


if __name__ == "__main__":
    main()
