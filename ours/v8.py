"""v8 -- v6 + penalite de changement d'equipe VARIABLE, pilotee par un detecteur de changements
de possession (piste 4 : "bascules de forme").

Constat (validate_v7.py) : la forme des equipes dit QUI a le ballon, mais v6 le sait deja ; ce qui
coute en spellF1, c'est le MOMENT des changements. Or dans le Viterbi, passer H <-> A coute la
meme chose partout (30 nats).

Idee : quand la possession change, les deux equipes basculent de forme EN MEME TEMPS (celle qui
recupere s'ecarte, l'autre se resserre), avec 2-4 s de retard. On apprend sur IDSSE un detecteur
P(changement d'equipe a +/- 0.5 s de t) a partir :
  - des features v5 (ballon, touches, etat d'equipe v4.2, ...),
  - des features de forme (ours/shape.py),
  - de leurs variations futur - passe a 2 s et 4 s (le retard de la forme est ainsi absorbe).
Puis : penalite H<->A a la frame t = C_HIGH - (C_HIGH - C_LOW) * q(t), q = max local du detecteur.
Changer d'equipe devient bon marche la ou un changement est probable, et cher ailleurs.
Entraine uniquement sur IDSSE ; aucun label du match du challenge.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import v1  # noqa: E402
import v3  # noqa: E402
from v5_em import em_prior  # noqa: E402

FPS = 25
TURN_TOL = FPS // 2                  # label positif : changement a +/- 0.5 s
HORIZONS = (2 * FPS, 4 * FPS)
TREND_KEYS = ("p_state_m05", "d_diff_m05")
C_HIGH, C_LOW, Q_WIN = 30.0, 8.0, FPS // 2


def turnover_features(X, names, S, snames):
    """Features du detecteur. Toutes les colonnes ajoutees changent de signe sous l'echange H<->A."""
    cols = [X[:, names.index(k)] for k in TREND_KEYS] + \
           [S[:, i] for i, k in enumerate(snames) if k.endswith("_diff_m2")]
    extra = []
    for c in cols:
        c = c.astype(np.float64)
        for h in HORIZONS:
            extra.append(np.nan_to_num(v3.shift(c, -h) - v3.shift(c, h)))
        extra.append(np.nan_to_num(v3.shift(c, -HORIZONS[0]) - c))
    return np.hstack([X, S, np.stack(extra, 1).astype(np.float32)])


def turnover_labels(gt):
    """1 si un changement direct H<->A a lieu a +/- TURN_TOL frames."""
    ch = np.zeros(len(gt), bool)
    ch[1:] = np.isin(gt[1:], ["H", "A"]) & np.isin(gt[:-1], ["H", "A"]) & (gt[1:] != gt[:-1])
    return v3.spread(ch, TURN_TOL)


def local_max(p, w):
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.pad(p, w, mode="edge")
    return sliding_window_view(pad, 2 * w + 1).max(axis=1)


def switch_cost(p_turn, c_high=C_HIGH, c_low=C_LOW, win=Q_WIN):
    return c_high - (c_high - c_low) * local_max(p_turn, win)


def viterbi_switch(logp, switch, cost_in, cost_out):
    """Viterbi de v3 avec une penalite H<->A propre a chaque frame."""
    n = len(logp)
    logp = np.asarray(logp, dtype=np.float64).tolist()
    sw, ci_, co_ = (list(map(float, a)) for a in (switch, cost_in, cost_out))
    s0, s1, s2 = logp[0]
    back = [None] * n
    for t in range(1, n):
        ci, co, st = ci_[t], co_[t], sw[t]
        c00, c10, c20 = s0, s1 - st, s2 - co
        b0 = 0 if c00 >= c10 and c00 >= c20 else (1 if c10 >= c20 else 2)
        c01, c11, c21 = s0 - st, s1, s2 - co
        b1 = 1 if c11 >= c01 and c11 >= c21 else (0 if c01 >= c21 else 2)
        c02, c12, c22 = s0 - ci, s1 - ci, s2
        b2 = 2 if c22 >= c02 and c22 >= c12 else (0 if c02 >= c12 else 1)
        lp = logp[t]
        s0 = (c00, c10, c20)[b0] + lp[0]
        s1 = (c01, c11, c21)[b1] + lp[1]
        s2 = (c02, c12, c22)[b2] + lp[2]
        back[t] = (b0, b1, b2)
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax((s0, s1, s2)))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t][path[t]]
    return path


def decode(p_h, z, no_players, kickoffs, cost_in, cost_out, bias, switch):
    """Comme v3.decode, avec switch = tableau de penalites H<->A par frame."""
    p_dead = np.clip(np.where(no_players, 0.95, v1.sigmoid(z + bias)), v1.EPS, 1 - v1.EPS)
    logp = np.stack([np.log(1 - p_dead) + np.log(p_h),
                     np.log(1 - p_dead) + np.log(1 - p_h),
                     np.log(p_dead)], axis=1)
    labels = np.full(len(p_h), "X", dtype="<U1")
    for half, first, last in v3.half_video_blocks(v3.GAME):
        sl = slice(first, last + 1)
        lp = logp[sl].copy()
        ko = [k for k in kickoffs if first - 30 * FPS <= k <= last]
        if ko and ko[0] > first:
            lp[: ko[0] - first, :2] = -1e6
        path = viterbi_switch(lp, switch[sl], cost_in[sl], cost_out[sl])
        labels[sl] = v3.enforce_min_durations(v1.LETTERS[path])
    return labels


def decode_probs(P, extras, switch, alive_target):
    """Comme v5.decode_probs (calibrage du biais de D), avec penalite H<->A variable."""
    eps = 1e-4
    p_d = np.clip(P[:, 2], eps, 1 - eps)
    z = np.log(p_d / (1 - p_d))
    p_h = np.clip(P[:, 0] / np.maximum(P[:, 0] + P[:, 1], eps), eps, 1 - eps)
    args = (p_h, z, extras["no_players"], extras["kickoffs"], extras["cost_in"], extras["cost_out"])
    lo, hi = -6.0, 6.0
    for _ in range(12):
        mid = (lo + hi) / 2
        frac = v3.alive_fraction(decode(*args, mid, switch))
        lo, hi = (mid, hi) if frac > alive_target else (lo, mid)
    return decode(*args, (lo + hi) / 2, switch)


def em_alive(P, graded, prior):
    return float(np.clip(1 - em_prior(P[graded], prior)[2], 0.50, 0.65))


TURN_MODEL_PATH = os.path.join(HERE, "models", "v8_turnover.joblib")


if __name__ == "__main__":
    # Soumission : modele v5 (ours/v5_train.py) + detecteur (ours/v8_train.py), C_HIGH 30 / C_LOW 8
    # (reglage retenu sur IDSSE, validate_v8.py), taux de jeu vivant par EM comme v6.
    import argparse
    import joblib
    import shape
    import v5
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="v8")
    args = ap.parse_args()
    for p, cmd in ((v5.MODEL_PATH, "v5_train.py"), (TURN_MODEL_PATH, "v8_train.py")):
        if not os.path.exists(p):
            sys.exit("Modele absent : lance d'abord  python ours/%s" % cmd)
    main_b, turn_b = joblib.load(v5.MODEL_PATH), joblib.load(TURN_MODEL_PATH)
    X, names, extras = v5.build_features()
    assert names == main_b["names"] == turn_b["names"], "features differentes de l'entrainement"
    teams, px, py, ps, bx, by, bs = v3.load_tracking()
    blocks = v3.half_video_blocks(v3.GAME)
    S, sn = shape.shape_features(px, py, teams, bx, by, blocks)
    assert sn == turn_b["snames"]
    P = main_b["model"].predict_proba(X)
    p_turn = turn_b["model"].predict_proba(turnover_features(X, names, S, sn))[:, 1]
    graded = np.zeros(len(P), bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True
    alive = em_alive(P, graded, main_b["prior"])
    labels = decode_probs(P, extras, switch_cost(p_turn), alive)
    out = os.path.join(v1.RESULTS, args.name + ".csv")
    v1.write(out, labels)
    print("jeu vivant %.1f %% (EM) | segments D : %d | ecrit -> %s"
          % (100 * v3.alive_fraction(labels), v3.count_dead_segments(labels), out))
