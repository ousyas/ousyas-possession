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


def main():
    import joblib
    import v5
    datas = []
    for i, mid in enumerate(T.MATCHES):
        t0 = time.time()
        d = T.prepare(mid, i)
        teams, px, py, ps, bx, by, bs = d["tracking"]
        S, sn = shape.shape_features(px, py, teams, bx, by, d["blocks"])
        d["E"], en = events.event_features(px, py, ps, bx, by)
        d["XT"] = np.hstack([v8.turnover_features(d["X"], d["names"], S, sn), d["E"]])
        d["XTs"] = np.hstack([v8.turnover_features(v5.swap_features(d["X"], d["names"]), d["names"],
                                                   shape.swap_shape(S, sn), sn), d["E"]])
        d["turn"] = v8.turnover_labels(d["gt"]).astype(int)
        datas.append(d)
        print("features %s (%.0f s)" % (mid, time.time() - t0), flush=True)
    idx = list(range(len(datas)))
    d_tr = np.mean(np.concatenate([d["y"][d["graded"] & (d["y"] >= 0)] == 2 for d in datas]))
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump({"model": train_main(datas, idx, True), "names": datas[0]["names"], "enames": en,
                 "prior": np.array([(1 - d_tr) / 2, (1 - d_tr) / 2, d_tr])}, MODEL_PATH)
    joblib.dump({"model": train_turnover(datas, idx), "names": datas[0]["names"], "snames": sn,
                 "enames": en}, TURN_MODEL_PATH)
    print("modeles -> %s, %s" % (os.path.relpath(MODEL_PATH, os.path.dirname(HERE)),
                                 os.path.relpath(TURN_MODEL_PATH, os.path.dirname(HERE))))


if __name__ == "__main__":
    main()
