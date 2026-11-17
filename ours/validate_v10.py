"""Validation IDSSE : v8 (gradient boosting) contre v10 (Transformer, ours/tf_model.py) et leur
moyenne (ensemble), dans le MEME pipeline (detecteur de changements v8, EM, Viterbi).

    python ours/validate_v10.py                 # 7 matchs, un laisse de cote a chaque fois
    python ours/validate_v10.py --folds 1       # un seul match teste (mesure du temps)

Critere fixe A L'AVANCE : on garde v10 (ou l'ensemble) seulement s'il bat v8 en spellF1 moyen
ET sur au moins 5 matchs sur 7. Journal -> ours/bench_results_v10.csv.
"""
import argparse
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
import tf_model as TF  # noqa: E402
from validate_v8 import train_turnover  # noqa: E402
from validate_v9 import train_main  # noqa: E402

LOG = os.path.join(HERE, "bench_results_v10%s.csv" % ("_banc2" if bench.DEAD_SIM else ""))
COLS = ["spellF1", "macroF1", "alive_dead_f1", "team_acc", "team_far"]


def sequences(d, mu, sd, swapped):
    import v5
    X = v5.swap_features(d["X"], d["names"]) if swapped else d["X"]
    F = (np.hstack([X, d["M"]]) - mu) / sd
    idx = np.arange(0, len(F), TF.STEP)
    P, B = (TF.swap_tokens(d["tokP"], d["tokB"]) if swapped else (d["tokP"], d["tokB"]))
    y = np.where(d["graded"], d["y"], -1)
    if swapped:
        y = np.where(y == 0, 1, np.where(y == 1, 0, y))
    return TF.Data(P, d["tokV"], B, F[idx].astype(np.float32), y[idx].astype(np.int64))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=7)
    ap.add_argument("--epochs", type=int, default=6)
    args = ap.parse_args()
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
        d["M"], _ = mobility.mobility_features(ps)
        d["tokP"], d["tokV"], d["tokB"] = TF.build_tokens(d["tracking"], d["blocks"])
        datas.append(d)
        print("features %s (%.0f s)" % (mid, time.time() - t0), flush=True)

    rows = []
    for k, mid in enumerate(T.MATCHES[:args.folds]):
        idx = [j for j in range(len(datas)) if j != k]
        d = datas[k]
        g = d["graded"]
        t0 = time.time()
        tmodel = train_turnover(datas, idx)
        sw = v8.switch_cost(tmodel.predict_proba(d["XT"])[:, 1])
        d_tr = np.mean(np.concatenate([datas[j]["y"][datas[j]["graded"] & (datas[j]["y"] >= 0)] == 2
                                       for j in idx]))
        prior = np.array([(1 - d_tr) / 2, (1 - d_tr) / 2, d_tr])
        gbm = train_main(datas, idx, False)          # v8 : sans mobilite
        P9 = gbm.predict_proba(d["X"])
        print("  %s : gradient boosting entraine (%.0f s)" % (mid, time.time() - t0), flush=True)

        t0 = time.time()
        allF = np.concatenate([np.hstack([datas[j]["X"], datas[j]["M"]]) for j in idx])
        mu, sd = allF.mean(0), allF.std(0) + 1e-6
        seqs = [sequences(datas[j], mu, sd, s) for j in idx for s in (False, True)]
        net = TF.train(seqs, epochs=args.epochs, seed=0)
        P10 = TF.predict(net, sequences(d, mu, sd, False), len(g))
        print("  %s : Transformer entraine (%.0f s)" % (mid, time.time() - t0), flush=True)

        bench.patch([v1, v2, v3, v4, v42, v5, v8], d["tracking"], d["blocks"])
        for name, P in (("v8", P9), ("v10", P10), ("v8+v10", (P9 + P10) / 2)):
            alive = v8.em_alive(P, g, prior)
            pred = np.where(g, v8.decode_probs(P, d["extras"], sw, alive), "X")
            s = T.score(d["gt"], pred)
            rows.append(dict(match=mid, version=name, alive_em=alive, **dict(zip(COLS, s))))
            print("  %-8s %-7s spellF1 %.4f macroF1 %.4f viv/mort %.3f equipe %.3f hors-trans %.3f (EM %.1f %%)"
                  % ((mid, name) + tuple(s) + (100 * alive,)), flush=True)

    nf = min(args.folds, len(T.MATCHES))
    print("\nMOYENNE (%d match(s) teste(s), leave-one-match-out)" % nf)
    base = np.array([r["spellF1"] for r in rows if r["version"] == "v8"])
    for name in ("v8", "v10", "v8+v10"):
        a = np.array([[r[c] for c in COLS] for r in rows if r["version"] == name])
        extra = "" if name == "v8" else "  | vs v8 %+.4f, gagne %d/%d" % (
            (a[:, 0] - base).mean(), int((a[:, 0] > base).sum()), len(base))
        print("  %-7s spellF1 %.4f (+/- %.4f) macroF1 %.4f viv/mort %.3f equipe %.3f hors-trans %.3f%s"
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
