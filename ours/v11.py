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
from v11_train import model_paths, gru_path, GRU_VERSIONS, GRU_SEEDS, ALIAS  # noqa: E402

REPLAY_VERSIONS = {"v16", "v19"}    # broadcast replays as D evidence (needs data/broadcast/)
# v19 = add-only: replays only ADD dead-ball evidence (nothing taken elsewhere) and the alive rate is
# re-estimated after it (validate_replay.py add_only); v16 moved dead frames at a fixed alive rate.
REPLAY_ADD_ONLY = {"v19"}
# v17: decoder costs recalibrated for v15's smoother probabilities (validate_decoder.py, nested
# leave-one-match-out): (C_HIGH, C_LOW of the H<->A switch, factor on the stoppage entry/exit costs)
DECODER = {"v17": None}


def main():
    import joblib
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v11")
    ap.add_argument("--version", default="v11", help="v11 (submitted) or v12 (control-aligned labels)")
    ap.add_argument("--replay-evidence", action="store_true",
                    help="broadcast replays / close-ups as label-free D evidence (ours/broadcast_evidence.py)")
    ap.add_argument("--gap-evidence", action="store_true",
                    help="label-free ball-gap evidence on D (ours/gap_evidence.py). Off = the submitted file")
    args = ap.parse_args()
    model_path, turn_path = model_paths(args.version)
    for p in (model_path, turn_path):
        if not os.path.exists(p):
            sys.exit("Modele absent : lance d'abord  python ours/v11_train.py --version %s" % args.version)
    main_b, turn_b = joblib.load(model_path), joblib.load(turn_path)
    X, names, extras = v5.build_features()
    assert names == main_b["names"] == turn_b["names"], "features differentes de l'entrainement"
    teams, px, py, ps, bx, by, bs = v3.load_tracking()
    blocks = v3.half_video_blocks(v3.GAME)
    S, sn = shape.shape_features(px, py, teams, bx, by, blocks)
    E, en = events.event_features(px, py, ps, bx, by)
    assert sn == turn_b["snames"] and en == main_b["enames"] == turn_b["enames"]
    P = main_b["model"].predict_proba(np.hstack([X, E]))
    mv = ALIAS.get(args.version, args.version)  # model version (v16 uses v15's models)
    if mv in GRU_VERSIONS:                     # v13: + temporal GRU head, 50/50 (validate_seq.py)
        import seq_model
        P_g = np.mean([seq_model.predict(joblib.load(gru_path(mv, s)), np.hstack([X, E]), blocks)
                       for s in GRU_SEEDS.get(mv, (0,))], 0)
        P = 0.5 * P + 0.5 * P_g
    p_turn = turn_b["model"].predict_proba(np.hstack([v8.turnover_features(X, names, S, sn), E]))[:, 1]
    graded = np.zeros(len(P), bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True
    alive = v8.em_alive(P, graded, main_b["prior"])
    if args.replay_evidence or args.version in REPLAY_VERSIONS:   # global alive rate unchanged
        import broadcast_evidence as BE
        import gap_evidence as G
        from load_homography import load as load_h
        _, _, b_valid = load_h(source="broadcast")
        b_bx, _ = BE.broadcast_ball()
        offs = BE.sync(bx, b_bx, blocks, b_valid)
        cat = BE.replay_category(len(P), blocks, offs, b_valid)
        lr, info = BE.category_log_ratio(P, cat, graded, main_b["prior"])
        if args.version in REPLAY_ADD_ONLY:
            lr = np.where(cat == 2, np.maximum(lr, 0.0), 0.0)
        P = G.apply_log_ratio(P, lr)
        if args.version in REPLAY_ADD_ONLY:
            alive = v8.em_alive(P, graded, main_b["prior"])
        print("replay evidence: offsets %s | EM dead rate per category (0 none, 1 pitch, 2 replay): %s"
              % ({h: v[0] for h, v in offs.items()}, {c: round(v[1], 3) for c, v in info.items() if c != "global"}))
    if args.gap_evidence:                      # the global alive rate above stays the target
        import gap_evidence as G
        lr, info = G.gap_log_ratio(P, bx, graded, main_b["prior"])
        P = G.apply_log_ratio(P, lr)
        rates = {c: round(v[1], 3) for c, v in info.items() if c != "global"}
        print("gap evidence (EM dead rate per gap class, global %.3f): %s" % (info["global"], rates))
    switch = v8.switch_cost(p_turn)
    if DECODER.get(args.version):
        ch, cl, f = DECODER[args.version]
        switch = v8.switch_cost(p_turn, ch, cl)
        extras = dict(extras, cost_in=extras["cost_in"] * f, cost_out=extras["cost_out"] * f)
    labels = v8.decode_probs(P, extras, switch, alive)
    out = args.name if os.path.isabs(args.name) else os.path.join(v1.RESULTS, args.name + ".csv")
    v1.write(out, labels)
    print("jeu vivant %.1f %% (EM %.1f %%) | segments D : %d | ecrit -> %s"
          % (100 * v3.alive_fraction(labels), 100 * alive, v3.count_dead_segments(labels), out))


if __name__ == "__main__":
    main()
