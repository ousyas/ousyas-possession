"""Validation IDSSE : v8 contre v9 (= v8 + features de mobilite longue, ours/mobility.py, ajoutees
au modele principal H/A/D). Le detecteur de changements de v8 est le meme dans les deux cas.

    python ours/validate_v9.py

Meme protocole (un match laisse de cote, EM, penalite H<->A variable 30 -> 8).
Journal -> ours/bench_results_v9.csv.
"""
import csv
import datetime
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import idsse_bench as bench  # noqa: E402
import v5_train as T  # noqa: E402
import shape  # noqa: E402
import mobility  # noqa: E402
import v8  # noqa: E402
from validate_v8 import train_turnover  # noqa: E402

LOG = os.path.join(HERE, "bench_results_v9%s.csv" % ("_banc2" if bench.DEAD_SIM else ""))
COLS = ["spellF1", "macroF1", "alive_dead_f1", "team_acc", "team_far"]


def train_main(datas, idx, use_mob):
    from sklearn.ensemble import HistGradientBoostingClassifier
    import v5
    Xs, ys = [], []
    for j in idx:
        d = datas[j]
        keep = np.flatnonzero(d["graded"] & (d["y"] >= 0))[::T.SUBSAMPLE]
        X, y = d["X"][keep], d["y"][keep]
        Xw = v5.swap_features(X, d["names"])
        if use_mob:                                    # mobilite : symetrique H/A
            X, Xw = np.hstack([X, d["M"][keep]]), np.hstack([Xw, d["M"][keep]])
        Xs += [X, Xw]
        ys += [y, np.where(y == 0, 1, np.where(y == 1, 0, y))]
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                           l2_regularization=1.0, random_state=0)
    model.fit(np.concatenate(Xs), np.concatenate(ys))
    return model


def main():
    import v1, v2, v3, v4, v42, v5  # noqa: F401
    datas = []
    for i, mid in enumerate(T.MATCHES):
        t0 = time.time()
        d = T.prepare(mid, i)
        teams, px, py, ps, bx, by, bs = d["tracking"]
        S, sn = shape.shape_features(px, py, teams, bx, by, d["blocks"])
        d["XT"] = v8.turnover_features(d["X"], d["names"], S, sn)
        d["XTs"] = v8.turnover_features(v5.swap_features(d["X"], d["names"]), d["names"],
                                        shape.swap_shape(S, sn), sn)
        d["turn"] = v8.turnover_labels(d["gt"]).astype(int)
        d["M"], d["mnames"] = mobility.mobility_features(ps)
        datas.append(d)
        print("features %s (%.0f s)" % (mid, time.time() - t0), flush=True)

    rows = []
    for k, mid in enumerate(T.MATCHES):
        idx = [j for j in range(len(datas)) if j != k]
        d = datas[k]
        g = d["graded"]
        tmodel = train_turnover(datas, idx)
        d_tr = np.mean(np.concatenate([datas[j]["y"][datas[j]["graded"] & (datas[j]["y"] >= 0)] == 2
                                       for j in idx]))
        prior = np.array([(1 - d_tr) / 2, (1 - d_tr) / 2, d_tr])
        bench.patch([v1, v2, v3, v4, v42, v5, v8], d["tracking"], d["blocks"])
        sw = v8.switch_cost(tmodel.predict_proba(d["XT"])[:, 1])
        for name, use_mob in (("v8", False), ("v9", True)):
            model = train_main(datas, idx, use_mob)
            P = model.predict_proba(np.hstack([d["X"], d["M"]]) if use_mob else d["X"])
            alive = v8.em_alive(P, g, prior)
            pred = np.where(g, v8.decode_probs(P, d["extras"], sw, alive), "X")
            s = T.score(d["gt"], pred)
            rows.append(dict(match=mid, version=name, alive_em=alive, **dict(zip(COLS, s))))
            print("  %-8s %-3s spellF1 %.4f macroF1 %.4f viv/mort %.3f equipe %.3f hors-trans %.3f (EM %.1f %%)"
                  % ((mid, name) + tuple(s) + (100 * alive,)), flush=True)

    print("\nMOYENNE (%d matchs, leave-one-match-out)" % len(T.MATCHES))
    for name in ("v8", "v9"):
        a = np.array([[r[c] for c in COLS] for r in rows if r["version"] == name])
        print("  %-3s spellF1 %.4f (+/- %.4f) macroF1 %.4f viv/mort %.3f equipe %.3f hors-trans %.3f"
              % (name, a[:, 0].mean(), a[:, 0].std(), *a[:, 1:].mean(0)))
    for c in COLS:
        a = np.array([r[c] for r in rows if r["version"] == "v8"])
        b = np.array([r[c] for r in rows if r["version"] == "v9"])
        print("  %-14s v9 - v8 : %+.4f   v9 gagne %d/%d matchs" % (c, (b - a).mean(), int((b > a).sum()), len(a)))

    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date"] + list(rows[0].keys()))
        if new:
            w.writeheader()
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        for r in rows:
            w.writerow(dict(date=stamp, **r))
    print("\njournal -> %s" % os.path.relpath(LOG, os.path.dirname(HERE)))


if __name__ == "__main__":
    main()
