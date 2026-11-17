"""Validation IDSSE de v4.2 / v5 (taux vivant fixe 58 %) / v6 (= v5 --em), un match laisse de cote.

    python ours/validate_v5_v6.py

Pour chaque match IDSSE : entraine le modele v5 sur les 6 autres, predit celui-ci, decode avec
(a) 58 % fixe = v5, (b) taux estime par EM = v6, (c) taux vrai = oracle (borne haute, info seulement).
v4.2 (regles seules) sert de reference. Score officiel. Journal -> ours/bench_results_v5.csv.
Aucun label du match du challenge n'est utilise.
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
from v5_em import em_prior  # noqa: E402

LOG = os.path.join(HERE, "bench_results_v5.csv")
COLS = ["spellF1", "macroF1", "alive_dead_f1", "team_acc", "team_far"]


def main():
    import v1, v2, v3, v4, v42, v5  # noqa: F401
    datas = []
    for i, mid in enumerate(T.MATCHES):
        t0 = time.time()
        datas.append(T.prepare(mid, i))
        print("features %s (%.0f s)" % (mid, time.time() - t0), flush=True)

    rows = []
    for k, mid in enumerate(T.MATCHES):
        idx = [j for j in range(len(datas)) if j != k]
        model = T.train(datas, idx)
        d_tr = np.mean(np.concatenate([datas[j]["y"][datas[j]["graded"] & (datas[j]["y"] >= 0)] == 2
                                       for j in idx]))
        prior = np.array([(1 - d_tr) / 2, (1 - d_tr) / 2, d_tr])
        d = datas[k]
        bench.patch([v1, v2, v3, v4, v42, v5], d["tracking"], d["blocks"])
        P = model.predict_proba(d["X"])
        g = d["graded"]
        alive_em = float(1 - em_prior(P[g], prior)[2])
        alive_true = float(np.mean(d["gt"][g] != "D"))
        preds = {"v4_2": d["ref"]}
        for name, target in (("v5", 0.58), ("v6", float(np.clip(alive_em, 0.50, 0.65))),
                             ("oracle", alive_true)):
            preds[name] = np.where(g, v5.decode_probs(P, d["extras"], target), "X")
        for name, p in preds.items():
            s = T.score(d["gt"], p)
            rows.append(dict(match=mid, version=name, alive_true=alive_true, alive_em=alive_em,
                             **dict(zip(COLS, s))))
            print("  %-8s %-7s spellF1 %.4f macroF1 %.4f viv/mort %.3f equipe %.3f hors-trans %.3f"
                  % ((mid, name) + tuple(s)), flush=True)
        print("  %s : vivant vrai %.1f %% | EM %.1f %%" % (mid, 100 * alive_true, 100 * alive_em), flush=True)

    print("\nMOYENNE (%d matchs, leave-one-match-out)" % len(T.MATCHES))
    for name in ("v4_2", "v5", "v6", "oracle"):
        a = np.array([[r[c] for c in COLS] for r in rows if r["version"] == name])
        print("  %-7s spellF1 %.4f (+/- %.4f) macroF1 %.4f viv/mort %.3f equipe %.3f hors-trans %.3f"
              % (name, a[:, 0].mean(), a[:, 0].std(), *a[:, 1:].mean(0)))
    sp = {n: np.array([r["spellF1"] for r in rows if r["version"] == n]) for n in ("v4_2", "v5", "v6")}
    for a, b in (("v4_2", "v5"), ("v5", "v6"), ("v4_2", "v6")):
        diff = sp[b] - sp[a]
        print("  %s vs %s : %+.4f en moyenne, %s gagne %d/%d matchs"
              % (b, a, diff.mean(), b, int((diff > 0).sum()), len(diff)))

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
