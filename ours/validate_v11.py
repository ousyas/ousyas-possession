"""Validation IDSSE (banc 2) : v8 contre v11 = v8 + features "evenements" (ours/events.py).

    python ours/validate_v11.py

  v8    : soumission actuelle
  v11   : events ajoutees au modele principal P(H/A/D)
  v11t  : events ajoutees au modele principal ET au detecteur de changements
Critere fixe A L'AVANCE : battre v8 en spellF1 moyen ET sur au moins 5 matchs sur 7.
Journal -> ours/bench_results_v11_banc2.csv.
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
import events  # noqa: E402
import v8  # noqa: E402
from validate_v8 import train_turnover  # noqa: E402

LOG = os.path.join(HERE, "bench_results_v11%s.csv" % ("_banc2" if bench.DEAD_SIM else ""))
COLS = ["spellF1", "macroF1", "alive_dead_f1", "team_acc", "team_far"]


def train_main(datas, idx, use_ev):
    from sklearn.ensemble import HistGradientBoostingClassifier
    import v5
    Xs, ys = [], []
    for j in idx:
        d = datas[j]
        keep = np.flatnonzero(d["graded"] & (d["y"] >= 0))[::T.SUBSAMPLE]
        X, y = d["X"][keep], d["y"][keep]
        Xw = v5.swap_features(X, d["names"])
        if use_ev:                                     # events : symetriques H/A
            X, Xw = np.hstack([X, d["E"][keep]]), np.hstack([Xw, d["E"][keep]])
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
        d["E"], _ = events.event_features(px, py, ps, bx, by)
        datas.append(d)
        print("features %s (%.0f s)" % (mid, time.time() - t0), flush=True)
    # vues "detecteur + events"
    datas_t = [dict(d, XT=np.hstack([d["XT"], d["E"]]), XTs=np.hstack([d["XTs"], d["E"]])) for d in datas]

    rows = []
    for k, mid in enumerate(T.MATCHES):
        idx = [j for j in range(len(datas)) if j != k]
        d = datas[k]
        g = d["graded"]
        t0 = time.time()
        sw = v8.switch_cost(train_turnover(datas, idx).predict_proba(d["XT"])[:, 1])
        sw_e = v8.switch_cost(train_turnover(datas_t, idx).predict_proba(datas_t[k]["XT"])[:, 1])
        d_tr = np.mean(np.concatenate([datas[j]["y"][datas[j]["graded"] & (datas[j]["y"] >= 0)] == 2
                                       for j in idx]))
        prior = np.array([(1 - d_tr) / 2, (1 - d_tr) / 2, d_tr])
        P8 = train_main(datas, idx, False).predict_proba(d["X"])
        P11 = train_main(datas, idx, True).predict_proba(np.hstack([d["X"], d["E"]]))
        bench.patch([v1, v2, v3, v4, v42, v5, v8], d["tracking"], d["blocks"])
        for name, P, s_ in (("v8", P8, sw), ("v11", P11, sw), ("v11t", P11, sw_e)):
            alive = v8.em_alive(P, g, prior)
            pred = np.where(g, v8.decode_probs(P, d["extras"], s_, alive), "X")
            s = T.score(d["gt"], pred)
            rec_d = float(np.mean(pred[d["gt"] == "D"] == "D"))
            rows.append(dict(match=mid, version=name, alive_em=alive, recall_D=rec_d, **dict(zip(COLS, s))))
            print("  %-8s %-5s spellF1 %.4f macroF1 %.4f viv/mort %.3f equipe %.3f hors-trans %.3f rappel D %.3f (EM %.1f %%)"
                  % ((mid, name) + tuple(s) + (rec_d, 100 * alive)), flush=True)
        print("  (%.0f s)" % (time.time() - t0), flush=True)

    print("\nMOYENNE (banc %d, 7 matchs, leave-one-match-out)" % (2 if bench.DEAD_SIM else 1))
    base = np.array([r["spellF1"] for r in rows if r["version"] == "v8"])
    cols = COLS + ["recall_D"]
    for name in ("v8", "v11", "v11t"):
        a = np.array([[r[c] for c in cols] for r in rows if r["version"] == name])
        extra = "" if name == "v8" else "  | vs v8 %+.4f, gagne %d/7" % ((a[:, 0] - base).mean(), int((a[:, 0] > base).sum()))
        print("  %-5s spellF1 %.4f (+/- %.4f) macroF1 %.4f viv/mort %.3f equipe %.3f hors-trans %.3f rappel D %.3f%s"
              % ((name, a[:, 0].mean(), a[:, 0].std()) + tuple(a[:, 1:].mean(0)) + (extra,)))

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
