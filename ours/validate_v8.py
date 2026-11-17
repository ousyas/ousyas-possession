"""Validation IDSSE : v6 (penalite H<->A fixe) contre v8 (penalite variable, ours/v8.py).

    python ours/validate_v8.py

Protocole identique a validate_v5_v6.py / validate_v7.py (un match laisse de cote) ; le modele
principal ET le detecteur de changements sont entraines sur les 6 autres matchs.
Plusieurs reglages (C_HIGH, C_LOW) sont rapportes ; journal -> ours/bench_results_v8.csv.
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
import v8  # noqa: E402

LOG = os.path.join(HERE, "bench_results_v8.csv")
COLS = ["spellF1", "macroF1", "alive_dead_f1", "team_acc", "team_far"]
CONFIGS = {"v6": None, "v8_30_15": (30.0, 15.0), "v8_30_8": (30.0, 8.0), "v8_40_8": (40.0, 8.0)}


def train_turnover(datas, idx):
    from sklearn.ensemble import HistGradientBoostingClassifier
    import v5
    Xs, ys = [], []
    for j in idx:
        d = datas[j]
        keep = np.flatnonzero(d["graded"] & (d["y"] >= 0))[::T.SUBSAMPLE]
        Xs += [d["XT"][keep], d["XTs"][keep]]
        ys += [d["turn"][keep]] * 2
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                           l2_regularization=1.0, random_state=0)
    model.fit(np.concatenate(Xs), np.concatenate(ys))
    return model


def main():
    import v1, v2, v3, v4, v42, v5  # noqa: F401
    from sklearn.metrics import roc_auc_score, average_precision_score
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
        datas.append(d)
        print("features %s (%.0f s)" % (mid, time.time() - t0), flush=True)

    rows = []
    for k, mid in enumerate(T.MATCHES):
        idx = [j for j in range(len(datas)) if j != k]
        d = datas[k]
        g = d["graded"]
        model = T.train(datas, idx)
        tmodel = train_turnover(datas, idx)
        d_tr = np.mean(np.concatenate([datas[j]["y"][datas[j]["graded"] & (datas[j]["y"] >= 0)] == 2
                                       for j in idx]))
        prior = np.array([(1 - d_tr) / 2, (1 - d_tr) / 2, d_tr])
        bench.patch([v1, v2, v3, v4, v42, v5, v8], d["tracking"], d["blocks"])
        P = model.predict_proba(d["X"])
        p_turn = tmodel.predict_proba(d["XT"])[:, 1]
        print("  %s detecteur de changements : AUC %.3f  AP %.3f (taux positifs %.3f)"
              % (mid, roc_auc_score(d["turn"][g], p_turn[g]), average_precision_score(d["turn"][g], p_turn[g]),
                 d["turn"][g].mean()), flush=True)
        alive = v8.em_alive(P, g, prior)
        for name, cfg in CONFIGS.items():
            sw = np.full(len(P), 30.0) if cfg is None else v8.switch_cost(p_turn, *cfg)
            pred = np.where(g, v8.decode_probs(P, d["extras"], sw, alive), "X")
            s = T.score(d["gt"], pred)
            rows.append(dict(match=mid, version=name, **dict(zip(COLS, s))))
            print("  %-8s %-9s spellF1 %.4f macroF1 %.4f viv/mort %.3f equipe %.3f hors-trans %.3f"
                  % ((mid, name) + tuple(s)), flush=True)

    print("\nMOYENNE (%d matchs, leave-one-match-out)" % len(T.MATCHES))
    base = np.array([r["spellF1"] for r in rows if r["version"] == "v6"])
    for name in CONFIGS:
        a = np.array([[r[c] for c in COLS] for r in rows if r["version"] == name])
        print("  %-9s spellF1 %.4f (+/- %.4f) macroF1 %.4f viv/mort %.3f equipe %.3f hors-trans %.3f"
              % (name, a[:, 0].mean(), a[:, 0].std(), *a[:, 1:].mean(0))
              + ("" if name == "v6" else "  | vs v6 %+.4f, gagne %d/%d"
                 % ((a[:, 0] - base).mean(), int((a[:, 0] > base).sum()), len(base))))

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
