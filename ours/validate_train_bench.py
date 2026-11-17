"""Faut-il entrainer v8 sur le banc corrige (ballon realiste pendant les arrets) ?

    python ours/validate_train_bench.py

Test TOUJOURS sur le banc corrige (le plus proche de l'ATD du challenge). Pour chaque match laisse
de cote, le modele principal + le detecteur de changements sont entraines sur les 6 autres :
  "v8_banc1" : avec l'ancien banc (ballon fige pendant les arrets) = la soumission actuelle
  "v8_banc2" : avec le banc corrige
Journal -> ours/bench_results_train_bench.csv.
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
from validate_v8 import train_turnover  # noqa: E402

LOG = os.path.join(HERE, "bench_results_train_bench.csv")
COLS = ["spellF1", "macroF1", "alive_dead_f1", "team_acc", "team_far"]


def prepare_all(dead_sim):
    import v5
    bench.DEAD_SIM = dead_sim
    out = []
    for i, mid in enumerate(T.MATCHES):
        t0 = time.time()
        d = T.prepare(mid, i)
        teams, px, py, ps, bx, by, bs = d["tracking"]
        S, sn = shape.shape_features(px, py, teams, bx, by, d["blocks"])
        d["XT"] = v8.turnover_features(d["X"], d["names"], S, sn)
        d["XTs"] = v8.turnover_features(v5.swap_features(d["X"], d["names"]), d["names"],
                                        shape.swap_shape(S, sn), sn)
        d["turn"] = v8.turnover_labels(d["gt"]).astype(int)
        out.append(d)
        print("features %s banc%d (%.0f s)" % (mid, 2 if dead_sim else 1, time.time() - t0), flush=True)
    return out


def main():
    import v1, v2, v3, v4, v42, v5  # noqa: F401
    banks = {"v8_banc1": prepare_all(False), "v8_banc2": prepare_all(True)}
    bench.DEAD_SIM = True
    test = banks["v8_banc2"]
    rows = []
    for k, mid in enumerate(T.MATCHES):
        idx = [j for j in range(len(T.MATCHES)) if j != k]
        d = test[k]
        g = d["graded"]
        for name, train_d in banks.items():
            model = T.train(train_d, idx)
            tmodel = train_turnover(train_d, idx)
            d_tr = np.mean(np.concatenate([train_d[j]["y"][train_d[j]["graded"] & (train_d[j]["y"] >= 0)] == 2
                                           for j in idx]))
            prior = np.array([(1 - d_tr) / 2, (1 - d_tr) / 2, d_tr])
            bench.patch([v1, v2, v3, v4, v42, v5, v8], d["tracking"], d["blocks"])
            P = model.predict_proba(d["X"])
            sw = v8.switch_cost(tmodel.predict_proba(d["XT"])[:, 1])
            alive = v8.em_alive(P, g, prior)
            pred = np.where(g, v8.decode_probs(P, d["extras"], sw, alive), "X")
            s = T.score(d["gt"], pred)
            # rappel des arrets : part des frames D vraies predites D
            gd = d["gt"] == "D"
            rec_d = float(np.mean(pred[gd] == "D"))
            rows.append(dict(match=mid, version=name, alive_em=alive, recall_D=rec_d, **dict(zip(COLS, s))))
            print("  %-8s %-9s spellF1 %.4f macroF1 %.4f viv/mort %.3f rappel D %.3f equipe %.3f (EM %.1f %%)"
                  % (mid, name, s[0], s[1], s[2], rec_d, s[3], 100 * alive), flush=True)

    print("\nMOYENNE (teste sur le banc corrige, 7 matchs, leave-one-match-out)")
    cols = COLS + ["recall_D"]
    for name in banks:
        a = np.array([[r[c] for c in cols] for r in rows if r["version"] == name])
        print("  %-9s spellF1 %.4f (+/- %.4f) macroF1 %.4f viv/mort %.3f equipe %.3f hors-trans %.3f rappel D %.3f"
              % ((name, a[:, 0].mean(), a[:, 0].std()) + tuple(a[:, 1:].mean(0))))
    for c in cols:
        a = np.array([r[c] for r in rows if r["version"] == "v8_banc1"])
        b = np.array([r[c] for r in rows if r["version"] == "v8_banc2"])
        print("  %-14s banc2 - banc1 : %+.4f   banc2 gagne %d/%d" % (c, (b - a).mean(), int((b > a).sum()), len(a)))

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
