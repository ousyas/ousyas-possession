"""Entraine le detecteur de changements de v8 sur les 7 matchs IDSSE et le sauve.

    python ours/v8_train.py          # -> ours/models/v8_turnover.joblib
    python ours/v8.py                # -> evaluation/results/v8.csv

(Le modele principal reste ours/models/v5.joblib, produit par ours/v5_train.py.)
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
import v8  # noqa: E402
from validate_v8 import train_turnover  # noqa: E402


def main():
    import joblib
    import v5
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
    model = train_turnover(datas, list(range(len(datas))))
    os.makedirs(os.path.dirname(v8.TURN_MODEL_PATH), exist_ok=True)
    joblib.dump({"model": model, "names": datas[0]["names"], "snames": sn}, v8.TURN_MODEL_PATH)
    print("detecteur -> %s" % os.path.relpath(v8.TURN_MODEL_PATH, os.path.dirname(HERE)))


if __name__ == "__main__":
    main()
