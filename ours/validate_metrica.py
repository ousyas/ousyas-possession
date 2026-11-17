"""Does adding Metrica's sample games 1-2 (same tracker family as BAR-ATH) to the training set help?

    python ours/validate_metrica.py --job trees --fold J03WMX
    python ours/validate_metrica.py --job gru --fold MET1
    python ours/validate_metrica.py --job score

Protocol, fixed BEFORE running: 9 held-out matches (7 IDSSE + MET1 + MET2), one GRU seed (v14
structure, so the baseline is v14's own fold models). For each held-out match:
  base  v14: trees + GRU trained on the 7 IDSSE matches other than it (cached by
        validate_gru_seeds.py for IDSSE folds; the final v14 models for the Metrica folds),
  new   the same pipeline trained on the same IDSSE matches PLUS the Metrica game(s) not held out.
Every training match appears in bench 2 and atd_stop (Metrica keeps its real tracking in both, so it
counts twice). Scored in atd_stop against control-aligned truth. Adoption: the new training beats v14
on mean spellF1 over the 9 matches AND wins at least 6 of 9. Log -> ours/bench_results_metrica.csv.
"""
import argparse
import csv
import datetime
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
import idsse_bench as bench  # noqa: E402
import metrica_bench as MB  # noqa: E402,F401  (registers MET1 / MET2 in idsse_bench)
import v5  # noqa: E402
import v5_train as T  # noqa: E402
import v8  # noqa: E402
import seq_model as S  # noqa: E402
from validate_gap import prepare_world, decode  # noqa: E402
from validate_relabel import with_labels  # noqa: E402

ALL = T.MATCHES + MB.MET_MATCHES
SEED_OF = {m: i for i, m in enumerate(ALL)}
WORLDS = ("legacy", "atd_stop")
CACHE = os.path.join(bench.EXT, "metcache")
SEEDCACHE = os.path.join(bench.EXT, "seedcache")


def load(mid, world, heavy):
    z = prepare_world(mid, SEED_OF[mid], world)
    keys = ["X", "E", "graded", "gt", "blocks", "names"] + (["XT", "XTs"] if heavy else [])
    d = with_labels({k: z[k] for k in keys}, z["gt"])
    d["F"] = np.hstack([d["X"], d["E"]])
    d["Fs"] = np.hstack([v5.swap_features(d["X"], d["names"]), d["E"]])
    return d


def train_set(fold, heavy):
    idsse = [m for m in T.MATCHES if m != fold]
    met = [m for m in MB.MET_MATCHES if m != fold]
    return [load(m, w, heavy) for w in WORLDS for m in idsse + met]


def prior_of(datas):
    d_tr = np.mean(np.concatenate([d["y"][d["graded"] & (d["y"] >= 0)] == 2 for d in datas]))
    return np.array([(1 - d_tr) / 2, (1 - d_tr) / 2, d_tr])


def job_trees(fold):
    from validate_v8 import train_turnover
    from validate_v11 import train_main
    datas = train_set(fold, True)
    idx = list(range(len(datas)))
    mm, tm = train_main(datas, idx, True), train_turnover(datas, idx)
    d = load(fold, "atd_stop", True)
    np.savez(os.path.join(CACHE, "trees_%s.npz" % fold), P=mm.predict_proba(d["F"]),
             sw=v8.switch_cost(tm.predict_proba(d["XT"])[:, 1]),
             prior=prior_of([x for x in datas if True]))


def job_gru(fold):
    datas = train_set(fold, False)
    S.SEED = 0
    b = S.train(datas, list(range(len(datas))))
    d = load(fold, "atd_stop", False)
    np.savez(os.path.join(CACHE, "gru_%s.npz" % fold), P=S.predict(b, d["F"], d["blocks"]))


def base_probs(fold):
    """v14 (seed 0) fold models: cached for IDSSE folds, final v14 models for Metrica folds."""
    if fold in T.MATCHES:
        t = np.load(os.path.join(SEEDCACHE, "trees_%s.npz" % fold))
        g = np.load(os.path.join(SEEDCACHE, "gru_%s_s0.npz" % fold))
        tr = [prepare_world(m, SEED_OF[m], "legacy") for m in T.MATCHES if m != fold]
        pri = prior_of([with_labels({"graded": x["graded"]}, x["gt"]) for x in tr])
        return 0.5 * t["P_atd_stop"] + 0.5 * g["P_atd_stop"], t["sw_atd_stop"], pri
    import joblib
    mb, tb = joblib.load(os.path.join(HERE, "models", "v14.joblib")), joblib.load(os.path.join(HERE, "models", "v14_turnover.joblib"))
    gb = joblib.load(os.path.join(HERE, "models", "v14_gru.joblib"))
    d = load(fold, "atd_stop", True)
    P = 0.5 * mb["model"].predict_proba(d["F"]) + 0.5 * S.predict(gb, d["F"], d["blocks"])
    return P, v8.switch_cost(tb["model"].predict_proba(d["XT"])[:, 1]), mb["prior"]


def job_score():
    rows = []
    for fold in ALL:
        z = prepare_world(fold, SEED_OF[fold], "atd_stop")
        d = {k: z[k] for k in ("graded", "gt", "blocks", "no_players", "cost_in", "cost_out")}
        g = d["graded"]
        truth = bench.align_to_control(d["gt"], 2.0)
        Pb, swb, prb = base_probs(fold)
        t, gr = np.load(os.path.join(CACHE, "trees_%s.npz" % fold)), np.load(os.path.join(CACHE, "gru_%s.npz" % fold))
        Pn = 0.5 * t["P"] + 0.5 * gr["P"]
        res = {}
        for name, P, sw, pr in (("v14", Pb, swb, prb), ("v14+metrica", Pn, t["sw"], t["prior"])):
            pred = decode(P, d, sw, v8.em_alive(P, g, pr))
            s = T.score(truth, pred)
            res[name] = (s[0], s[1], float(np.mean(pred[truth == "D"] == "D")))
        rows.append(dict(match=fold, v14=res["v14"][0], new=res["v14+metrica"][0], diff=res["v14+metrica"][0] - res["v14"][0],
                         macro_v14=res["v14"][1], macro_new=res["v14+metrica"][1],
                         recallD_v14=res["v14"][2], recallD_new=res["v14+metrica"][2]))
        r = rows[-1]
        print("  %-7s spellF1 v14 %.4f -> +Metrica %.4f (%+.4f) | macroF1 %.4f -> %.4f | stoppage recall %.3f -> %.3f"
              % (fold, r["v14"], r["new"], r["diff"], r["macro_v14"], r["macro_new"], r["recallD_v14"], r["recallD_new"]), flush=True)
    d = np.array([r["diff"] for r in rows])
    print("ALL 9:   v14 %.4f -> +Metrica %.4f  diff %+.4f  wins %d/9" % (np.mean([r["v14"] for r in rows]), np.mean([r["new"] for r in rows]), d.mean(), int((d > 0).sum())))
    di = d[:7]
    print("IDSSE 7: diff %+.4f wins %d/7 | Metrica 2: diff %+.4f wins %d/2" % (di.mean(), int((di > 0).sum()), d[7:].mean(), int((d[7:] > 0).sum())))
    log = os.path.join(HERE, "bench_results_metrica.csv")
    with open(log, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date"] + list(rows[0].keys()))
        w.writeheader()
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        for r in rows:
            w.writerow(dict(date=stamp, **r))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, choices=["trees", "gru", "score", "prep"])
    ap.add_argument("--fold")
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    if a.job == "prep":
        for m in MB.MET_MATCHES:
            for w in WORLDS:
                prepare_world(m, SEED_OF[m], w)
        print("metrica features cached")
    elif a.job == "trees":
        job_trees(a.fold)
    elif a.job == "gru":
        job_gru(a.fold)
    else:
        job_score()


if __name__ == "__main__" and not (len(sys.argv) > 1 and sys.argv[1] == "gru_only"):
    main()


def job_score_gru_only():
    """Time-bound variant (decided 20:21 when the tree retraining could not finish before the
    deadline): only the GRU learns from Metrica, the trees and turnover detector stay v14's.
    Same adoption rule. This can only under-estimate what Metrica adds."""
    rows = []
    for fold in ALL:
        z = prepare_world(fold, SEED_OF[fold], "atd_stop")
        d = {k: z[k] for k in ("graded", "gt", "blocks", "no_players", "cost_in", "cost_out")}
        g = d["graded"]
        truth = bench.align_to_control(d["gt"], 2.0)
        Pb, swb, prb = base_probs(fold)
        if fold in T.MATCHES:
            Pt = np.load(os.path.join(SEEDCACHE, "trees_%s.npz" % fold))["P_atd_stop"]
        else:
            import joblib
            Pt = joblib.load(os.path.join(HERE, "models", "v14.joblib"))["model"].predict_proba(load(fold, "atd_stop", False)["F"])
        Pn = 0.5 * Pt + 0.5 * np.load(os.path.join(CACHE, "gru_%s.npz" % fold))["P"]
        res = {}
        for name, P in (("v14", Pb), ("new", Pn)):
            pred = decode(P, d, swb, v8.em_alive(P, g, prb))
            s = T.score(truth, pred)
            res[name] = (s[0], s[1], float(np.mean(pred[truth == "D"] == "D")))
        rows.append(dict(match=fold, v14=res["v14"][0], new=res["new"][0], diff=res["new"][0] - res["v14"][0],
                         macro_v14=res["v14"][1], macro_new=res["new"][1], recallD_v14=res["v14"][2], recallD_new=res["new"][2]))
        r = rows[-1]
        print("  %-7s spellF1 v14 %.4f -> GRU+Metrica %.4f (%+.4f) | macroF1 %.4f -> %.4f | stoppage recall %.3f -> %.3f"
              % (fold, r["v14"], r["new"], r["diff"], r["macro_v14"], r["macro_new"], r["recallD_v14"], r["recallD_new"]), flush=True)
    dd = np.array([r["diff"] for r in rows])
    print("ALL 9:   v14 %.4f -> GRU+Metrica %.4f  diff %+.4f  wins %d/9" % (np.mean([r["v14"] for r in rows]), np.mean([r["new"] for r in rows]), dd.mean(), int((dd > 0).sum())))
    print("IDSSE 7: diff %+.4f wins %d/7 | Metrica 2: diff %+.4f wins %d/2" % (dd[:7].mean(), int((dd[:7] > 0).sum()), dd[7:].mean(), int((dd[7:] > 0).sum())))
    with open(os.path.join(HERE, "bench_results_metrica.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "gru_only":
    job_score_gru_only()
