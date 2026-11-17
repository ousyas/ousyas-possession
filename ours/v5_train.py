"""Entrainement + validation de v5 sur IDSSE (7 matchs, un match laisse de cote a chaque fois).

    pip install scikit-learn
    python ours/v5_train.py
    python ours/v5_train.py --matches J03WMX J03WN1 J03WOH J03WOY J03WPY J03WQQ J03WR9

Pour chaque match : tracking degrade (comme le banc), features v5, labels H/A/D du fournisseur.
Validation : entraine sur 6 matchs, predit le 7e, decode (Viterbi v3), score officiel ; x7.
Compare a v4.2 sur les memes donnees. Puis entraine sur les 7 et sauve ours/models/v5.joblib.
"""

import argparse
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import idsse_bench as bench  # noqa: E402
import possession_metrics as pm  # noqa: E402

MATCHES = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]
SUBSAMPLE = 3          # 1 frame sur 3 pour l'entrainement (frames voisines tres correlees)
LAB = np.array(["H", "A", "D"])


def prepare(mid, i, synthetic=False, seed=0):
    import v1, v2, v3, v4, v42, v5
    m = bench.synthetic_match(i) if synthetic else bench.load_match(mid)
    blocks = bench.blocks_from_period(m["period"])
    n = len(m["labels"])
    tracking = bench.degrade(m, seed + i)
    bench.patch([v1, v2, v3, v4, v42, v5], tracking, blocks)
    X, names, extras = v5.build_features()
    graded = np.zeros(n, bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True
    y = np.full(n, -1)
    for k, c in enumerate("HAD"):
        y[m["labels"] == c] = k
    gt = np.full(n, "X", "<U1")
    gt[graded] = m["labels"][graded]
    # reference v4.2 sur les memes donnees (meme pipeline que le banc)
    p_h, z, nop, ko, ci, co, _ = v42.evidences()
    ref = bench.calibrate_and_decode(None, lambda b: v3.decode(p_h, z, nop, ko, ci, co, b, 30.0), 0.58)
    ref = np.where(graded, ref, "X")
    return dict(X=X, names=names, extras=extras, y=y, graded=graded, gt=gt, blocks=blocks, ref=ref,
                tracking=tracking)


def train(datas, idx):
    from sklearn.ensemble import HistGradientBoostingClassifier
    import v5
    Xs, ys = [], []
    for j in idx:
        d = datas[j]
        keep = np.flatnonzero(d["graded"] & (d["y"] >= 0))[::SUBSAMPLE]
        X, y = d["X"][keep], d["y"][keep]
        Xs += [X, v5.swap_features(X, d["names"])]          # augmentation H <-> A
        ys += [y, np.where(y == 0, 1, np.where(y == 1, 0, y))]
    model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                           l2_regularization=1.0, random_state=0)
    model.fit(np.concatenate(Xs), np.concatenate(ys))
    return model


def score(gt, pred):
    r = pm.evaluate(pm.label_codes(gt), pm.label_codes(pred), bootstrap_n=20)
    return r["spell_f1_mean"], r["macro_f1"], r["alive_dead_f1"], r["team_acc_both_alive"], \
        bench.team_acc_far(gt, pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", nargs="+", default=MATCHES)
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--final-only", action="store_true",
                    help="saute la validation, entraine seulement le modele final (reproduction)")
    args = ap.parse_args()
    import v1, v2, v3, v4, v42, v5
    import joblib

    datas = []
    for i, mid in enumerate(args.matches):
        t0 = time.time()
        datas.append(prepare(mid, i, args.synthetic))
        print("features %s : %d frames, %d features (%.0f s)"
              % (mid, len(datas[-1]["y"]), datas[-1]["X"].shape[1], time.time() - t0))

    if not args.final_only:
        print("\nVALIDATION : un match laisse de cote a chaque fois (entraine sur les autres)")
        print("  %-8s %-6s %8s %8s %8s %8s %10s" % ("match", "modele", "spellF1", "macroF1", "viv/mort", "equipe", "hors-trans"))
        res = {"v4_2": [], "v5": []}
        for k, mid in enumerate(args.matches):
            model = train(datas, [j for j in range(len(datas)) if j != k])
            d = datas[k]
            bench.patch([v1, v2, v3, v4, v42, v5], d["tracking"], d["blocks"])
            P = model.predict_proba(d["X"])
            pred = np.where(d["graded"], v5.decode_probs(P, d["extras"]), "X")
            for name, p in (("v4_2", d["ref"]), ("v5", pred)):
                s = score(d["gt"], p)
                res[name].append(s)
                print("  %-8s %-6s %8.4f %8.4f %8.3f %8.3f %10.3f" % ((mid, name) + s))

        print("\nMOYENNE (validation, %d matchs)" % len(args.matches))
        for name in ("v4_2", "v5"):
            a = np.array(res[name])
            print("  %-6s spellF1 %.4f (+/- %.4f)  macroF1 %.4f  vivant/mort %.3f  equipe %.3f  hors-trans %.3f"
                  % (name, a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean(), a[:, 2].mean(), a[:, 3].mean(), a[:, 4].mean()))
        wins = sum(1 for a, b in zip(res["v4_2"], res["v5"]) if b[0] > a[0])
        print("  v5 bat v4_2 sur %d/%d matchs (spellF1)" % (wins, len(args.matches)))

    if not args.synthetic:
        model = train(datas, list(range(len(datas))))
        os.makedirs(os.path.dirname(v5.MODEL_PATH), exist_ok=True)
        d_tr = np.mean(np.concatenate([d["y"][d["graded"] & (d["y"] >= 0)] == 2 for d in datas]))
        joblib.dump({"model": model, "names": datas[0]["names"],
                     "prior": np.array([(1 - d_tr) / 2, (1 - d_tr) / 2, d_tr])}, v5.MODEL_PATH)
        print("\nmodele final (7 matchs) -> %s" % os.path.relpath(v5.MODEL_PATH, ROOT))


if __name__ == "__main__":
    main()
