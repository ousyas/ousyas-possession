"""Validation IDSSE : v6 (features v5 + EM) contre v7 (= v6 + features de forme, ours/shape.py).

    python ours/validate_v7.py

Meme protocole que validate_v5_v6.py : tracking IDSSE degrade, un match laisse de cote, decodage
v5 avec taux de jeu vivant estime par EM, score officiel. Journal -> ours/bench_results_v7.csv.
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
from v5_em import em_prior  # noqa: E402

LOG = os.path.join(HERE, "bench_results_v7.csv")
COLS = ["spellF1", "macroF1", "alive_dead_f1", "team_acc", "team_far"]


def train(datas, idx, use_shape):
    from sklearn.ensemble import HistGradientBoostingClassifier
    import v5
    Xs, ys = [], []
    for j in idx:
        d = datas[j]
        keep = np.flatnonzero(d["graded"] & (d["y"] >= 0))[::T.SUBSAMPLE]
        X, y = d["X"][keep], d["y"][keep]
        Xw = v5.swap_features(X, d["names"])
        if use_shape:
            S = d["S"][keep]
            X = np.hstack([X, S])
            Xw = np.hstack([Xw, shape.swap_shape(S, d["snames"])])
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
        d["S"], d["snames"] = shape.shape_features(px, py, teams, bx, by, d["blocks"])
        datas.append(d)
        print("features %s (%.0f s)" % (mid, time.time() - t0), flush=True)

    # pouvoir discriminant de chaque feature de forme seule (tracking degrade, jeu vivant)
    from sklearn.metrics import roc_auc_score
    print("\nAUC H vs A de chaque difference de forme seule (tracking degrade) :")
    for k in [s for s in datas[0]["snames"] if "_diff" in s]:
        j = datas[0]["snames"].index(k)
        a = [roc_auc_score(d["y"][(d["y"] < 2) & (d["y"] >= 0)] == 0,
                           d["S"][(d["y"] < 2) & (d["y"] >= 0), j]) for d in datas]
        print("  %-22s %.3f (min %.3f)" % (k, np.mean(a), np.min(a)))

    rows = []
    for k, mid in enumerate(T.MATCHES):
        idx = [j for j in range(len(datas)) if j != k]
        d_tr = np.mean(np.concatenate([datas[j]["y"][datas[j]["graded"] & (datas[j]["y"] >= 0)] == 2
                                       for j in idx]))
        prior = np.array([(1 - d_tr) / 2, (1 - d_tr) / 2, d_tr])
        d = datas[k]
        bench.patch([v1, v2, v3, v4, v42, v5], d["tracking"], d["blocks"])
        g = d["graded"]
        for name, use_shape in (("v6", False), ("v7", True)):
            model = train(datas, idx, use_shape)
            X = np.hstack([d["X"], d["S"]]) if use_shape else d["X"]
            P = model.predict_proba(X)
            alive_em = float(np.clip(1 - em_prior(P[g], prior)[2], 0.50, 0.65))
            pred = np.where(g, v5.decode_probs(P, d["extras"], alive_em), "X")
            s = T.score(d["gt"], pred)
            rows.append(dict(match=mid, version=name, alive_em=alive_em, **dict(zip(COLS, s))))
            print("  %-8s %-4s spellF1 %.4f macroF1 %.4f viv/mort %.3f equipe %.3f hors-trans %.3f"
                  % ((mid, name) + tuple(s)), flush=True)

    print("\nMOYENNE (%d matchs, leave-one-match-out)" % len(T.MATCHES))
    for name in ("v6", "v7"):
        a = np.array([[r[c] for c in COLS] for r in rows if r["version"] == name])
        print("  %-4s spellF1 %.4f (+/- %.4f) macroF1 %.4f viv/mort %.3f equipe %.3f hors-trans %.3f"
              % (name, a[:, 0].mean(), a[:, 0].std(), *a[:, 1:].mean(0)))
    for c in COLS:
        a = np.array([r[c] for r in rows if r["version"] == "v6"])
        b = np.array([r[c] for r in rows if r["version"] == "v7"])
        print("  %-14s v7 - v6 : %+.4f   v7 gagne %d/%d matchs" % (c, (b - a).mean(), int((b > a).sum()), len(a)))

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
