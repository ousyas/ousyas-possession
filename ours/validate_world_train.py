"""IDSSE validation: train v13r on BAR-ATH-like ball gaps as well (v14 candidate).

    python ours/validate_world_train.py --folds J03WMX      # one held-out match (run folds in parallel; merged log: bench_results_world.csv)

Why: the BAR-ATH ATD loses the ball in long blocks (85 % of missing frames in gaps > 10 s, 44 % in
gaps > 30 s, players at 58 % speed inside them: atd_gap_stats.py), but every model so far was
trained on the legacy bench, where gaps are short (median 2 s). "Degrade the training data with the
target's own noise" is the central idea of the strongest public entries.

Protocol, fixed BEFORE running: leave-one-match-out on the 7 IDSSE matches. Per fold:
  v13r      trees + turnover + GRU trained on the 6 other matches in the legacy world (= submission)
  v14       the same, trained on the 6 other matches in BOTH the legacy and the atd_stop worlds
Both are scored on the held-out match degraded in atd_stop (the world whose label-free statistics
match BAR-ATH: gap shape and the speed ratio in long gaps) and in legacy, against control-aligned
truth (and raw truth, logged). Adoption rule: v14 beats v13r in atd_stop on mean spellF1 AND on at
least 5/7 matches. Log -> ours/bench_results_world_<folds>.csv.
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

COLS = ["spellF1", "macroF1", "alive_dead_f1", "team_acc", "team_far"]
KEYS = ("X", "E", "XT", "XTs", "graded", "gt", "blocks", "no_players", "cost_in", "cost_out", "names")
TEST_WORLDS = ("atd_stop", "legacy")


def load(mid, i, world):
    d = prepare_world(mid, i, world)
    d = with_labels({k: d[k] for k in KEYS}, d["gt"])          # raw DFL labels, as v13r
    d["F"] = np.hstack([d["X"], d["E"]])
    d["Fs"] = np.hstack([v5.swap_features(d["X"], d["names"]), d["E"]])
    return d


def fit(datas, idx):
    return train_main(datas, idx, True), train_turnover(datas, idx), S.train(datas, idx)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", default=T.MATCHES)
    args = ap.parse_args()
    t0 = time.time()
    data = {(w, mid): load(mid, i, w) for i, mid in enumerate(T.MATCHES) for w in TEST_WORLDS}
    print("features (%.0f s)" % (time.time() - t0), flush=True)
    rows = []
    for k, mid in enumerate(T.MATCHES):
        if mid not in args.folds:
            continue
        others = [m for m in T.MATCHES if m != mid]
        legacy = [data["legacy", m] for m in others]
        mixed = legacy + [data["atd_stop", m] for m in others]
        d_tr = np.mean(np.concatenate([d["y"][d["graded"] & (d["y"] >= 0)] == 2 for d in legacy]))
        prior = np.array([(1 - d_tr) / 2, (1 - d_tr) / 2, d_tr])  # dead frames identical in both worlds
        models = {"v13r": fit(legacy, list(range(len(legacy)))), "v14": fit(mixed, list(range(len(mixed))))}
        print("  fold %s trained (%.0f s)" % (mid, time.time() - t0), flush=True)
        ctrl_truth = bench.align_to_control(data["legacy", mid]["gt"], 2.0)
        raw_truth = data["legacy", mid]["gt"]
        for w in TEST_WORLDS:
            d = data[w, mid]
            g = d["graded"]
            for name, (mm, tm, gm) in models.items():
                P = 0.5 * mm.predict_proba(d["F"]) + 0.5 * S.predict(gm, d["F"], d["blocks"])
                sw = v8.switch_cost(tm.predict_proba(d["XT"])[:, 1])
                pred = decode(P, d, sw, v8.em_alive(P, g, prior))
                for tname, truth in (("ctrl2", ctrl_truth), ("raw", raw_truth)):
                    s = T.score(truth, pred)
                    rows.append(dict(match=mid, world=w, version=name, truth=tname,
                                     poss_per_min=rhythm(pred, g), recall_D=float(np.mean(pred[truth == "D"] == "D")),
                                     **dict(zip(COLS, s))))
                r = rows[-2]
                print("  %-6s %-8s %-4s vs ctrl2 truth: spellF1 %.4f macroF1 %.4f recall D %.3f | %.2f poss/min"
                      % (mid, w, name, r["spellF1"], r["macroF1"], r["recall_D"], r["poss_per_min"]), flush=True)
    log = os.path.join(HERE, "bench_results_world_%s.csv" % "_".join(args.folds))
    with open(log, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["date"] + list(rows[0].keys()))
        wr.writeheader()
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        for r in rows:
            wr.writerow(dict(date=stamp, **r))
    print("log -> %s (%.0f min)" % (os.path.relpath(log, ROOT), (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
