"""Test : taux de jeu vivant fixe (58 %) vs estime par match (EM de Saerens et al. 2002).
    python ours/v5_em.py
"""
import os
import sys
import time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import idsse_bench as bench  # noqa: E402
import v5_train as T  # noqa: E402


def em_prior(P, prior_train, iters=200, tol=1e-6):
    P = np.clip(P, 1e-6, 1)
    pi = prior_train.copy()
    for _ in range(iters):
        Q = P * (pi / prior_train)
        Q /= Q.sum(1, keepdims=True)
        new = Q.mean(0)
        if np.abs(new - pi).max() < tol:
            pi = new
            break
        pi = new
    return pi


def main():
    import v1, v2, v3, v4, v42, v5
    datas = []
    for i, mid in enumerate(T.MATCHES):
        t0 = time.time()
        datas.append(T.prepare(mid, i))
        print("features %s (%.0f s)" % (mid, time.time() - t0))
    res = {"fixe": [], "em": [], "oracle": []}
    print("\n  %-8s %7s %7s | %8s %8s %8s" % ("match", "vrai", "EM", "fixe", "em", "oracle"))
    for k, mid in enumerate(T.MATCHES):
        idx = [j for j in range(len(datas)) if j != k]
        model = T.train(datas, idx)
        d_train = np.mean(np.concatenate([datas[j]["y"][datas[j]["graded"] & (datas[j]["y"] >= 0)] == 2
                                          for j in idx]))
        prior_train = np.array([(1 - d_train) / 2, (1 - d_train) / 2, d_train])
        d = datas[k]
        bench.patch([v1, v2, v3, v4, v42, v5], d["tracking"], d["blocks"])
        P = model.predict_proba(d["X"])
        g = d["graded"]
        alive_em = 1 - em_prior(P[g], prior_train)[2]
        alive_true = np.mean(d["gt"][g] != "D")
        scores = []
        for name, target in (("fixe", 0.58), ("em", alive_em), ("oracle", alive_true)):
            pred = np.where(g, v5.decode_probs(P, d["extras"], target), "X")
            s = T.score(d["gt"], pred)[0]
            res[name].append(s)
            scores.append(s)
        print("  %-8s %6.1f%% %6.1f%% | %8.4f %8.4f %8.4f" % ((mid, 100 * alive_true, 100 * alive_em) + tuple(scores)))
    print("\nMOYENNE spellF1")
    for name in res:
        a = np.array(res[name])
        print("  %-6s %.4f (+/- %.4f)" % (name, a.mean(), a.std()))
    wins = sum(b > a for a, b in zip(res["fixe"], res["em"]))
    print("  em bat fixe sur %d/%d matchs" % (wins, len(T.MATCHES)))


if __name__ == "__main__":
    main()
