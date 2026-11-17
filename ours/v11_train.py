"""Entraine les 2 modeles de v11 (= v8 + features "evenements", ours/events.py) sur les 7 matchs
IDSSE (banc 2) et les sauve.

    python ours/v11_train.py         # -> ours/models/v11.joblib + ours/models/v11_turnover.joblib
    python ours/v11.py               # -> evaluation/results/v11.csv

Retenu sur IDSSE (validate_v11.py, variante "v11t") : spellF1 0.4861 contre 0.4765 pour v8,
gagne 5/7 matchs (critere fixe a l'avance).
"""
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import v5_train as T  # noqa: E402
import shape  # noqa: E402
import events  # noqa: E402
import v8  # noqa: E402
from validate_v8 import train_turnover  # noqa: E402
from validate_v11 import train_main  # noqa: E402

MODEL_PATH = os.path.join(HERE, "models", "v11.joblib")
TURN_MODEL_PATH = os.path.join(HERE, "models", "v11_turnover.joblib")
# v12 = the v11 pipeline trained on IDSSE labels aligned to the challenge's CONTROL definition
# (idsse_bench.align_to_control, tau 2 s; validate_relabel.py). Same features, models, decoder.
# v13 / v13r = v12 / v11 + a temporal GRU head over the same features (seq_model.py, validate_seq.py),
# its probabilities averaged 50/50 with the boosted trees'. Same decoder.
# v14 = v13r trained on each IDSSE match degraded twice: the legacy bench AND the atd_stop world,
# whose ball gaps copy BAR-ATH's (long blocks, lost through stoppages; validate_world_train.py).
# v15 = v14 with the GRU head averaged over 3 training seeds (validate_gru_seeds.py).
# v18 = v15 trained on Metrica's sample games 1-2 as well (metrica_bench.py, validate_metrica.py).
CONTROL_TAU = {"v11": 0.0, "v12": 2.0, "v13": 2.0, "v13r": 0.0, "v14": 0.0, "v15": 0.0, "v18": 0.0}
GRU_VERSIONS = {"v13", "v13r", "v14", "v15", "v18"}
GRU_SEEDS = {"v15": (0, 1, 2), "v18": (0, 1, 2)}              # default: (0,)
TRAIN_WORLDS = {"v14": ("legacy", "atd_stop"), "v15": ("legacy", "atd_stop"),
                "v18": ("legacy", "atd_stop")}                 # default: ("legacy",)
EXTRA_MATCHES = {"v18": ("MET1", "MET2")}                      # Metrica sample games
TREES_FROM = {"v18": "v15"}     # v18 keeps v15's boosted trees + detector; only its GRUs see Metrica
# v16 = v15's models + broadcast replays as label-free evidence on D at prediction time
# (broadcast_evidence.py); no new model, so v16 reads and trains v15's files.
ALIAS = {"v16": "v15", "v17": "v15", "v19": "v15"}       # v17 = v15's models + a decoder recalibrated for them
WORLD_CFG = {"legacy": dict(world="legacy"), "atd_stop": dict(world="blocks", affinity=1.0, align=0.8)}


def model_paths(version="v11"):
    version = ALIAS.get(version, version)
    version = TREES_FROM.get(version, version)
    return (os.path.join(HERE, "models", version + ".joblib"),
            os.path.join(HERE, "models", version + "_turnover.joblib"))


def gru_path(version, seed=0):
    version = ALIAS.get(version, version)
    return os.path.join(HERE, "models", version + ("_gru.joblib" if seed == 0 else "_gru_s%d.joblib" % seed))


def main():
    import argparse
    import joblib
    import v5
    ap = argparse.ArgumentParser()
    ap.add_argument("--version", default="v11", choices=sorted(set(CONTROL_TAU) | set(ALIAS)))
    ap.add_argument("--only-gru-seed", type=int, default=None,
                    help="train only this GRU seed (lets the seeds of v15 train in parallel processes)")
    args = ap.parse_args()
    version = ALIAS.get(args.version, args.version)
    tau = CONTROL_TAU[version]
    MODEL_PATH, TURN_MODEL_PATH = model_paths(version)
    import idsse_bench as bench
    datas = []
    if version in EXTRA_MATCHES:
        import metrica_bench  # noqa: F401  (registers MET1 / MET2 in idsse_bench)
    matches = list(T.MATCHES) + list(EXTRA_MATCHES.get(version, ()))
    for world, i, mid in [(w, i, m) for w in TRAIN_WORLDS.get(version, ("legacy",))
                          for i, m in enumerate(matches)]:
        t0 = time.time()
        bench.set_world(**WORLD_CFG[world])
        try:
            d = T.prepare(mid, i)
        finally:
            bench.set_world("legacy")
        if tau:                                   # v12: touch-spells without control -> previous team
            import idsse_bench as bench
            d["gt"] = bench.align_to_control(d["gt"], tau)
            d["y"] = np.full(len(d["gt"]), -1)
            for k, c in enumerate("HAD"):
                d["y"][d["gt"] == c] = k
        teams, px, py, ps, bx, by, bs = d["tracking"]
        S, sn = shape.shape_features(px, py, teams, bx, by, d["blocks"])
        d["E"], en = events.event_features(px, py, ps, bx, by)
        d["XT"] = np.hstack([v8.turnover_features(d["X"], d["names"], S, sn), d["E"]])
        d["XTs"] = np.hstack([v8.turnover_features(v5.swap_features(d["X"], d["names"]), d["names"],
                                                   shape.swap_shape(S, sn), sn), d["E"]])
        d["turn"] = v8.turnover_labels(d["gt"]).astype(int)
        d.pop("tracking", None)                   # not needed any more: keeps 14 datasets in memory
        datas.append(d)
        print("features %s %s (%.0f s)" % (world, mid, time.time() - t0), flush=True)
    idx = list(range(len(datas)))
    d_tr = np.mean(np.concatenate([d["y"][d["graded"] & (d["y"] >= 0)] == 2 for d in datas]))
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if args.only_gru_seed is None and version not in TREES_FROM:
        joblib.dump({"model": train_main(datas, idx, True), "names": datas[0]["names"], "enames": en,
                     "prior": np.array([(1 - d_tr) / 2, (1 - d_tr) / 2, d_tr])}, MODEL_PATH)
        joblib.dump({"model": train_turnover(datas, idx), "names": datas[0]["names"], "snames": sn,
                     "enames": en}, TURN_MODEL_PATH)
    if version in GRU_VERSIONS:
        import seq_model
        for d in datas:
            d["F"] = np.hstack([d["X"], d["E"]])
            d["Fs"] = np.hstack([v5.swap_features(d["X"], d["names"]), d["E"]])
        seeds = GRU_SEEDS.get(version, (0,)) if args.only_gru_seed is None else (args.only_gru_seed,)
        for seed in seeds:
            seq_model.SEED = seed
            joblib.dump(seq_model.train(datas, idx), gru_path(version, seed))
            print("GRU -> %s" % os.path.relpath(gru_path(version, seed), os.path.dirname(HERE)))
    print("modeles -> %s, %s" % (os.path.relpath(MODEL_PATH, os.path.dirname(HERE)),
                                 os.path.relpath(TURN_MODEL_PATH, os.path.dirname(HERE))))


if __name__ == "__main__":
    main()
