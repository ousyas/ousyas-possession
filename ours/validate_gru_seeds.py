"""IDSSE validation of v15 = v14 with the GRU head averaged over 3 training seeds (0, 1, 2).

    python ours/validate_gru_seeds.py --job trees --fold J03WMX          # boosted trees + detector
    python ours/validate_gru_seeds.py --job gru --fold J03WMX --seed 1   # one GRU seed
    python ours/validate_gru_seeds.py --job score                        # decode + score all folds

Jobs are independent and cached (data/ext/seedcache/), so they run in parallel processes.
Protocol, fixed BEFORE running: exactly v14's training (each of the 6 other IDSSE matches degraded
in bench 2 and in atd_stop); v14 = 0.5 trees + 0.5 GRU(seed 0); v15 = 0.5 trees + 0.5 mean of the
GRUs of seeds 0, 1, 2. Same decoder. Scored on the held-out match degraded in atd_stop (BAR-ATH-like)
and in bench 2, against control-aligned and raw truth. Adoption rule: v15 beats v14 in atd_stop on
mean spellF1 (control truth) AND on at least 5/7 matches. Log -> ours/bench_results_gru_seeds.csv.
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
import v5  # noqa: E402
import v5_train as T  # noqa: E402
import v8  # noqa: E402
import seq_model as S  # noqa: E402
from validate_gap import prepare_world, decode  # noqa: E402
from validate_relabel import with_labels, rhythm  # noqa: E402

CACHE = os.path.join(bench.EXT, "seedcache")
WORLDS = ("legacy", "atd_stop")
TEST_WORLDS = ("atd_stop", "legacy")
SEEDS = (0, 1, 2)
COLS = ["spellF1", "macroF1", "alive_dead_f1", "team_acc", "team_far"]


def load(mid, world, heavy):
    i = T.MATCHES.index(mid)
    z = prepare_world(mid, i, world)
    keys = ["X", "E", "graded", "gt", "blocks", "names"] + (["XT", "XTs"] if heavy else [])
    d = with_labels({k: z[k] for k in keys}, z["gt"])
    d["F"] = np.hstack([d["X"], d["E"]])
    d["Fs"] = np.hstack([v5.swap_features(d["X"], d["names"]), d["E"]])
    return d


def train_set(fold, heavy):
    return [load(m, w, heavy) for w in WORLDS for m in T.MATCHES if m != fold]


def job_trees(fold):
    from validate_v8 import train_turnover
    from validate_v11 import train_main
    datas = train_set(fold, True)
    idx = list(range(len(datas)))
    mm, tm = train_main(datas, idx, True), train_turnover(datas, idx)
    out = {}
    for w in TEST_WORLDS:
        d = load(fold, w, True)
        out["P_" + w] = mm.predict_proba(d["F"])
        out["sw_" + w] = v8.switch_cost(tm.predict_proba(d["XT"])[:, 1])
    np.savez(os.path.join(CACHE, "trees_%s.npz" % fold), **out)


def job_gru(fold, seed):
    datas = train_set(fold, False)
    S.SEED = seed
    b = S.train(datas, list(range(len(datas))))
    out = {"P_" + w: S.predict(b, load(fold, w, False)["F"], load(fold, w, False)["blocks"]) for w in TEST_WORLDS}
    np.savez(os.path.join(CACHE, "gru_%s_s%d.npz" % (fold, seed)), **out)


def job_score():
    import validate_gap as V  # noqa: F401
    rows = []
    for fold in T.MATCHES:
        t = np.load(os.path.join(CACHE, "trees_%s.npz" % fold))
        g = {s: np.load(os.path.join(CACHE, "gru_%s_s%d.npz" % (fold, s))) for s in SEEDS}
        legacy_train = [load(m, "legacy", False) for m in T.MATCHES if m != fold]
        d_tr = np.mean(np.concatenate([d["y"][d["graded"] & (d["y"] >= 0)] == 2 for d in legacy_train]))
        prior = np.array([(1 - d_tr) / 2, (1 - d_tr) / 2, d_tr])
        for w in TEST_WORLDS:
            z = prepare_world(fold, T.MATCHES.index(fold), w)
            d = {k: z[k] for k in ("graded", "gt", "blocks", "no_players", "cost_in", "cost_out")}
            gm = d["graded"]
            ctrl, raw = bench.align_to_control(d["gt"], 2.0), d["gt"]
            Pg = {"v14": g[0]["P_" + w], "v15": np.mean([g[s]["P_" + w] for s in SEEDS], 0)}
            for name in ("v14", "v15"):
                P = 0.5 * t["P_" + w] + 0.5 * Pg[name]
                pred = decode(P, d, t["sw_" + w], v8.em_alive(P, gm, prior))
                for tname, truth in (("ctrl2", ctrl), ("raw", raw)):
                    s = T.score(truth, pred)
                    rows.append(dict(match=fold, world=w, version=name, truth=tname, poss_per_min=rhythm(pred, gm),
                                     recall_D=float(np.mean(pred[truth == "D"] == "D")), **dict(zip(COLS, s))))
    for w in TEST_WORLDS:
        for tname in ("ctrl2", "raw"):
            r = {(x["match"], x["version"]): x["spellF1"] for x in rows if x["world"] == w and x["truth"] == tname}
            d = [r[m, "v15"] - r[m, "v14"] for m in T.MATCHES]
            print("%-8s truth %-5s v14 %.4f  v15 %.4f  diff %+.4f  wins %d/7  per match %s"
                  % (w, tname, np.mean([r[m, "v14"] for m in T.MATCHES]), np.mean([r[m, "v15"] for m in T.MATCHES]),
                     np.mean(d), sum(x > 0 for x in d), [round(x, 4) for x in d]))
    log = os.path.join(HERE, "bench_results_gru_seeds.csv")
    with open(log, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["date"] + list(rows[0].keys()))
        wr.writeheader()
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        for r in rows:
            wr.writerow(dict(date=stamp, **r))
    print("log -> %s" % os.path.relpath(log, ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, choices=["trees", "gru", "score"])
    ap.add_argument("--fold")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(CACHE, exist_ok=True)
    if a.job == "trees":
        job_trees(a.fold)
    elif a.job == "gru":
        job_gru(a.fold, a.seed)
    else:
        job_score()


if __name__ == "__main__":
    main()
