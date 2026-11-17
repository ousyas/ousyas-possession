"""IDSSE validation: train v11 on labels aligned to the challenge's CONTROL definition.

    python ours/validate_relabel.py                  # tau 2 s (fixed a priori) + 3 s sensitivity

Why: DFL's ball_owning_team flips on every touch (8.4 possession changes per live minute, mean
possession 7.3 s), while the challenge only counts a change when the new team gains CONTROL (4.2
per live minute, 14.2 s on the public window). v11's models and switch costs learned the touch-level
rhythm. idsse_bench.align_to_control gives short touch-spells back to the previous team.

Protocol, fixed BEFORE running: leave-one-match-out on the 7 IDSSE matches, legacy degradation
(the bench every v1-v11 decision used), identical features, hyper-parameters and decoder. The
held-out match is scored against ITS ALIGNED labels (= the challenge definition) with the official
scorer. Compared: v11 (current, raw labels) vs v11-ctrl2 (trained on labels aligned with tau = 2 s).
Adoption rule (the team's own): v11-ctrl2 beats v11 on mean spellF1 AND on at least 5/7 matches.
Scores against tau-3 s truth are reported as a robustness check of the comparison.
Log -> ours/bench_results_relabel.csv.
"""
import csv
import datetime
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import idsse_bench as bench  # noqa: E402
import v5_train as T  # noqa: E402
import v8  # noqa: E402
from validate_gap import prepare_world, decode  # noqa: E402
from validate_v8 import train_turnover  # noqa: E402
from validate_v11 import train_main  # noqa: E402

LOG = os.path.join(HERE, "bench_results_relabel.csv")
COLS = ["spellF1", "macroF1", "alive_dead_f1", "team_acc", "team_far"]
KEYS = ("X", "E", "XT", "XTs", "graded", "gt", "blocks", "no_players", "cost_in", "cost_out", "names")


def with_labels(d, gt):
    y = np.full(len(gt), -1, np.int8)
    for k, c in enumerate("HAD"):
        y[gt == c] = k
    return dict(d, gt=gt, y=y, turn=v8.turnover_labels(gt).astype(np.int8))


def rhythm(lab, g):
    lab = lab[g]
    st = np.flatnonzero(np.r_[True, lab[1:] != lab[:-1]])
    n = sum(1 for a in st if lab[a] in "HA")
    return n / (np.isin(lab, ["H", "A"]).sum() / 25 / 60)


def main():
    t0 = time.time()
    raw, ctrl = [], {2.0: [], 3.0: []}
    for i, mid in enumerate(T.MATCHES):
        d = prepare_world(mid, i, "legacy")
        d = {k: d[k] for k in KEYS}
        raw.append(with_labels(d, d["gt"]))
        for tau in ctrl:
            ctrl[tau].append(with_labels(d, bench.align_to_control(d["gt"], tau)))
    print("features + labels (%.0f s)" % (time.time() - t0), flush=True)

    rows = []
    for k, mid in enumerate(T.MATCHES):
        idx = [j for j in range(len(raw)) if j != k]
        t1 = time.time()
        target = ctrl[2.0][k]                                   # challenge-definition truth
        g = target["graded"]
        d_tr = np.mean(np.concatenate([raw[j]["y"][raw[j]["graded"] & (raw[j]["y"] >= 0)] == 2 for j in idx]))
        prior = np.array([(1 - d_tr) / 2, (1 - d_tr) / 2, d_tr])  # D frames are unchanged by alignment
        for name, datas in (("v11", raw), ("v11-ctrl2", ctrl[2.0])):
            main_m, turn_m = train_main(datas, idx, True), train_turnover(datas, idx)
            P = main_m.predict_proba(np.hstack([target["X"], target["E"]]))
            sw = v8.switch_cost(turn_m.predict_proba(target["XT"])[:, 1])
            pred = decode(P, target, sw, v8.em_alive(P, g, prior))
            for truth_name, truth in (("ctrl2", target["gt"]), ("ctrl3", ctrl[3.0][k]["gt"]), ("raw", raw[k]["gt"])):
                s = T.score(truth, pred)
                rows.append(dict(match=mid, version=name, truth=truth_name, poss_per_min=rhythm(pred, g),
                                 true_poss_per_min=rhythm(truth, g), **dict(zip(COLS, s))))
            r = rows[-3]
            print("  %-6s %-9s vs ctrl2 truth: spellF1 %.4f macroF1 %.4f | %.2f poss/min (truth %.2f)"
                  % (mid, name, r["spellF1"], r["macroF1"], r["poss_per_min"], r["true_poss_per_min"]), flush=True)
        print("  fold %s (%.0f s)" % (mid, time.time() - t1), flush=True)

    print("\nSUMMARY  leave-one-match-out, legacy bench")
    for truth in ("ctrl2", "ctrl3", "raw"):
        base = np.array([r["spellF1"] for r in rows if r["truth"] == truth and r["version"] == "v11"])
        for name in ("v11", "v11-ctrl2"):
            a = np.array([[r[c] for c in COLS + ["poss_per_min"]] for r in rows
                          if r["truth"] == truth and r["version"] == name])
            extra = "" if name == "v11" else "  | vs v11 %+.4f, wins %d/7" % (
                (a[:, 0] - base).mean(), int((a[:, 0] > base).sum()))
            print("  truth %-5s %-9s spellF1 %.4f (+/- %.4f) macroF1 %.4f  %.2f poss/min%s"
                  % (truth, name, a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean(), a[:, 5].mean(), extra))

    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date"] + list(rows[0].keys()))
        if new:
            w.writeheader()
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        for r in rows:
            w.writerow(dict(date=stamp, **r))
    print("\nlog -> %s  (%.0f min)" % (os.path.relpath(LOG, ROOT), (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
