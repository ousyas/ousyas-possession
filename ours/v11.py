"""v11 -- la soumission : v8 + features "evenements" (ours/events.py) dans le modele principal
P(H/A/D) ET dans le detecteur de changements. Meme decodage que v8 (EM + Viterbi, cout 30 -> 8).

    python ours/v11_train.py         # modeles (IDSSE, 7 matchs)
    python ours/v11.py --name Ousyas # -> evaluation/results/Ousyas.csv
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import v1  # noqa: E402
import v3  # noqa: E402
import v5  # noqa: E402
import v8  # noqa: E402
import shape  # noqa: E402
import events  # noqa: E402
from v11_train import MODEL_PATH, TURN_MODEL_PATH  # noqa: E402


def main():
    import joblib
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v11")
    args = ap.parse_args()
    for p in (MODEL_PATH, TURN_MODEL_PATH):
        if not os.path.exists(p):
            sys.exit("Modele absent : lance d'abord  python ours/v11_train.py")
    main_b, turn_b = joblib.load(MODEL_PATH), joblib.load(TURN_MODEL_PATH)
    X, names, extras = v5.build_features()
    assert names == main_b["names"] == turn_b["names"], "features differentes de l'entrainement"
    teams, px, py, ps, bx, by, bs = v3.load_tracking()
    blocks = v3.half_video_blocks(v3.GAME)
    S, sn = shape.shape_features(px, py, teams, bx, by, blocks)
    E, en = events.event_features(px, py, ps, bx, by)
    assert sn == turn_b["snames"] and en == main_b["enames"] == turn_b["enames"]
    P = main_b["model"].predict_proba(np.hstack([X, E]))
    p_turn = turn_b["model"].predict_proba(np.hstack([v8.turnover_features(X, names, S, sn), E]))[:, 1]
    graded = np.zeros(len(P), bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True
    alive = v8.em_alive(P, graded, main_b["prior"])
    labels = v8.decode_probs(P, extras, v8.switch_cost(p_turn), alive)
    out = args.name if os.path.isabs(args.name) else os.path.join(v1.RESULTS, args.name + ".csv")
    v1.write(out, labels)
    print("jeu vivant %.1f %% (EM %.1f %%) | segments D : %d | ecrit -> %s"
          % (100 * v3.alive_fraction(labels), 100 * alive, v3.count_dead_segments(labels), out))


if __name__ == "__main__":
    main()
