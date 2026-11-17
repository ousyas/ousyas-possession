"""IDSSE validation of v13 = v11 + temporal GRU head (ours/seq_model.py), ensembled 50/50.

    python ours/validate_seq.py --labels raw      # train on DFL labels, as v11
    python ours/validate_seq.py --labels ctrl2    # train on control-aligned labels, as v12

Protocol, fixed BEFORE running: leave-one-match-out on the 7 IDSSE matches, legacy bench. Per
fold, v11's gradient-boosted classifier and turnover detector and the GRU are trained on the 6
other matches (same labels); the held-out match is decoded with v11's unchanged decoder
(EM alive rate + turnover-driven Viterbi). Scored with the official scorer against the
control-aligned truth (the challenge's definition; validate_relabel.py), raw truth also logged.
Variants: v11 (boosted trees only) | gru (GRU only, diagnostic) | v13 (0.5 trees + 0.5 GRU).
Adoption rule (the team's own): v13 beats v11 on mean spellF1 AND on at least 5/7 matches.
Log -> ours/bench_results_seq.csv.
"""
import argparse
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
import v5  # noqa: E402
import v5_train as T  # noqa: E402
import v8  # noqa: E402
import seq_model as S  # noqa: E402
from validate_gap import prepare_world, decode  # noqa: E402
from validate_relabel import with_labels, rhythm  # noqa: E402
from validate_v8 import train_turnover  # noqa: E402
from validate_v11 import train_main  # noqa: E402

LOG = os.path.join(HERE, "bench_results_seq.csv")
COLS = ["spellF1", "macroF1", "alive_dead_f1", "team_acc", "team_far"]
KEYS = ("X", "E", "XT", "XTs", "graded", "gt", "blocks", "no_players", "cost_in", "cost_out", "names")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="raw", choices=["raw", "ctrl2"])
    ap.add_argument("--folds", nargs="+", default=T.MATCHES,
                    help="held-out matches to run (folds are independent: run them in parallel processes)")
    args = ap.parse_args()
    t0 = time.time()
    datas, truth_ctrl, truth_raw = [], [], []
    for i, mid in enumerate(T.MATCHES):
        d = prepare_world(mid, i, "legacy")
        d = {k: d[k] for k in KEYS}
        ctrl = bench.align_to_control(d["gt"], 2.0)
        truth_ctrl.append(ctrl)
        truth_raw.append(d["gt"])
        d = with_labels(d, ctrl if args.labels == "ctrl2" else d["gt"])
        d["F"] = np.hstack([d["X"], d["E"]])
        d["Fs"] = np.hstack([v5.swap_features(d["X"], d["names"]), d["E"]])
        datas.append(d)
    print("features (%.0f s), training labels = %s" % (time.time() - t0, args.labels), flush=True)

    rows = []
    for k, mid in enumerate(T.MATCHES):
        if mid not in args.folds:
            continue
        idx = [j for j in range(len(datas)) if j != k]
        t1 = time.time()
        d = datas[k]
        g = d["graded"]
        d_tr = np.mean(np.concatenate([datas[j]["y"][datas[j]["graded"] & (datas[j]["y"] >= 0)] == 2 for j in idx]))
        prior = np.array([(1 - d_tr) / 2, (1 - d_tr) / 2, d_tr])
        P_t = train_main(datas, idx, True).predict_proba(d["F"])
        sw = v8.switch_cost(train_turnover(datas, idx).predict_proba(d["XT"])[:, 1])
        P_g = S.predict(S.train(datas, idx), d["F"], d["blocks"])
        for name, P in (("v11", P_t), ("gru", P_g), ("v13", 0.5 * P_t + 0.5 * P_g)):
            pred = decode(P, d, sw, v8.em_alive(P, g, prior))
            for tname, truth in (("ctrl2", truth_ctrl[k]), ("raw", truth_raw[k])):
                s = T.score(truth, pred)
                rows.append(dict(match=mid, train_labels=args.labels, version=name, truth=tname,
                                 poss_per_min=rhythm(pred, g), **dict(zip(COLS, s))))
            r = rows[-2]
            print("  %-6s %-4s vs ctrl2 truth: spellF1 %.4f macroF1 %.4f | %.2f poss/min"
                  % (mid, name, r["spellF1"], r["macroF1"], r["poss_per_min"]), flush=True)
        print("  fold %s (%.0f s)" % (mid, time.time() - t1), flush=True)

    if len(args.folds) < len(T.MATCHES):          # partial run: log rows only, summarise later
        args_summary = False
    else:
        args_summary = True
    print("\nSUMMARY  training labels = %s, leave-one-match-out, legacy bench" % args.labels) if args_summary else None
    for tname in (("ctrl2", "raw") if args_summary else ()):
        base = np.array([r["spellF1"] for r in rows if r["truth"] == tname and r["version"] == "v11"])
        for name in ("v11", "gru", "v13"):
            a = np.array([[r[c] for c in COLS + ["poss_per_min"]] for r in rows
                          if r["truth"] == tname and r["version"] == name])
            extra = "" if name == "v11" else "  | vs v11 %+.4f, wins %d/7" % (
                (a[:, 0] - base).mean(), int((a[:, 0] > base).sum()))
            print("  truth %-5s %-4s spellF1 %.4f (+/- %.4f) macroF1 %.4f  %.2f poss/min%s"
                  % (tname, name, a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean(), a[:, 5].mean(), extra))

    log = LOG if args_summary else LOG.replace(".csv", "_%s_%s.csv" % (args.labels, "_".join(args.folds)))
    new = not os.path.exists(log)
    with open(log, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date"] + list(rows[0].keys()))
        if new:
            w.writeheader()
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        for r in rows:
            w.writerow(dict(date=stamp, **r))
    print("\nlog -> %s  (%.0f min)" % (os.path.relpath(log, ROOT), (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
