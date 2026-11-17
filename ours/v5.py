"""v5 -- modele APPRIS (gradient boosting) sur nos indices + Viterbi dynamique (v3).

    python ours/v5_train.py          # 1) entraine sur IDSSE (validation 1 match laisse de cote)
    python ours/v5.py                # 2) applique au match du challenge -> evaluation/results/v5.csv

Pourquoi (litterature) :
  - Borghesi et al. 2023 : un modele appris bat les regles de ~7 points sur la possession
    d'equipe ; les joueurs proches du ballon portent l'essentiel de l'information.
  - Lang et al. 2022 : le statut du jeu se predit bien a partir du mouvement des joueurs,
    avec du contexte temporel.
Le modele ne remplace pas nos regles : il les COMBINE (etat d'equipe v4.2, score d'arret v3)
avec des features geometriques, et apprend leurs poids sur 7 vrais matchs (IDSSE, legal).
Entraine uniquement sur des matchs externes ; aucun label du match du challenge.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from features import FPS, LENGTH, WIDTH, rolling_mean, speed_signals, ball_out  # noqa: E402
import v1  # noqa: E402
import v2  # noqa: E402
import v3  # noqa: E402
import v42  # noqa: E402

MODEL_PATH = os.path.join(HERE, "models", "v5.joblib")


def _clip(x, hi=50.0):
    return np.where(np.isfinite(x), np.minimum(x, hi), hi)


def build_features():
    """Retourne (X, noms, extras) pour le tracking courant (v3.load_tracking)."""
    p_h_old, z, no_players, kickoffs, cost_in, cost_out, _ = v3.evidences()
    teams, px, py, ps, bx, by, bs = v3.load_tracking()
    n = len(bx)
    events = v3.load_smart_tagging.load()
    p_tag = v1.team_from_tags(events, n) if events else None
    p_state, _ = v42.team_probability(px, py, bx, by, teams, p_tag)

    bx_i, by_i = v1.interpolate_ball(bx, by)
    d = np.hypot((px - bx_i[:, None]) * LENGTH, (py - by_i[:, None]) * WIDTH)
    d = np.where(np.isnan(d), np.inf, d)
    dh_all = np.where(teams[None, :] == 0, d, np.inf)
    da_all = np.where(teams[None, :] == 1, d, np.inf)
    d_h, d_a = _clip(dh_all.min(1)), _clip(da_all.min(1))
    n_h5, n_a5 = (dh_all < 5).sum(1), (da_all < 5).sum(1)
    n_h10, n_a10 = (dh_all < 10).sum(1), (da_all < 10).sum(1)

    def spread(code):
        cols = teams == code
        with np.errstate(all="ignore"):
            return np.nan_to_num(np.nanstd(px[:, cols] * LENGTH, 1) + np.nanstd(py[:, cols] * WIDTH, 1))
    sp_h, sp_a = spread(0), spread(1)

    med, frac_slow, n_seen = speed_signals(ps)
    med = np.nan_to_num(med, nan=0.0)
    frac_slow = np.nan_to_num(frac_slow, nan=1.0)
    bv = np.nan_to_num(v2.ball_speed(bx, by), nan=-1.0)
    vis = (~np.isnan(bx)).astype(float)
    gap = np.minimum(v3.gap_length(bx), 10 * FPS) / FPS
    out = np.nan_to_num(ball_out(bx, by))
    bxf = np.nan_to_num(bx_i, nan=-1.0)
    byf = np.nan_to_num(by_i, nan=-1.0)
    edge = np.where(bxf < 0, -1.0, np.minimum.reduce([bxf, 1 - bxf, byf, 1 - byf]))

    f = {
        "p_state": p_state, "z_dead": z, "cin_low": (cost_in < 10).astype(float),
        "cout_low": (cost_out < 10).astype(float), "ball_vis": vis, "gap_s": gap,
        "ball_speed": bv, "ball_edge": edge, "out": out,
        "d_h": d_h, "d_a": d_a, "d_diff": d_a - d_h,
        "n_h5": n_h5, "n_a5": n_a5, "n_h10": n_h10, "n_a10": n_a10,
        "sp_h": sp_h, "sp_a": sp_a, "sp_diff": sp_h - sp_a,
        "med_speed": med, "frac_slow": frac_slow, "n_seen": n_seen.astype(float),
    }
    for key in ("p_state", "d_diff", "ball_speed", "med_speed", "sp_diff", "ball_vis", "z_dead"):
        x = np.asarray(f[key], dtype=float)
        f[key + "_m05"] = np.nan_to_num(rolling_mean(x, FPS // 2))
        f[key + "_m2"] = np.nan_to_num(rolling_mean(x, 2 * FPS))
        f[key + "_trend"] = np.nan_to_num(v3.shift(x, -FPS) - v3.shift(x, FPS))
    names = sorted(f)
    X = np.stack([np.asarray(f[k], dtype=np.float32) for k in names], axis=1)
    extras = dict(no_players=no_players, kickoffs=kickoffs, cost_in=cost_in, cost_out=cost_out)
    return X, names, extras


# Echange H <-> A sur les features (augmentation par symetrie)
def swap_features(X, names):
    Xs = X.copy()
    idx = {k: i for i, k in enumerate(names)}
    for i, k in enumerate(names):
        if k.startswith("p_state"):
            Xs[:, i] = (1 - X[:, i]) if not k.endswith("_trend") else -X[:, i]
        elif k.startswith("d_diff") or k.startswith("sp_diff"):
            Xs[:, i] = -X[:, i]
        elif k in ("d_h", "n_h5", "n_h10", "sp_h"):
            Xs[:, i] = X[:, idx[k.replace("_h", "_a", 1)]]
        elif k in ("d_a", "n_a5", "n_a10", "sp_a"):
            Xs[:, i] = X[:, idx[k.replace("_a", "_h", 1)]]
    return Xs


def decode_probs(P, extras, alive_target=0.58, switch=30.0):
    """P (n,3) = P(H), P(A), P(D) -> labels, avec calibrage du biais de D et Viterbi v3."""
    eps = 1e-4
    p_d = np.clip(P[:, 2], eps, 1 - eps)
    z = np.log(p_d / (1 - p_d))
    p_h = np.clip(P[:, 0] / np.maximum(P[:, 0] + P[:, 1], eps), eps, 1 - eps)
    args = (p_h, z, extras["no_players"], extras["kickoffs"], extras["cost_in"], extras["cost_out"])
    lo, hi = -6.0, 6.0
    for _ in range(12):
        mid = (lo + hi) / 2
        frac = v3.alive_fraction(v3.decode(*args, mid, switch))
        lo, hi = (mid, hi) if frac > alive_target else (lo, mid)
    return v3.decode(*args, (lo + hi) / 2, switch)


if __name__ == "__main__":
    import joblib
    ap = argparse.ArgumentParser()
    ap.add_argument("--alive-target", type=float, default=0.58)
    ap.add_argument("--name", default="v5")
    ap.add_argument("--em", action="store_true")
    args = ap.parse_args()
    if not os.path.exists(MODEL_PATH):
        sys.exit("Modele absent : lance d'abord  python ours/v5_train.py")
    bundle = joblib.load(MODEL_PATH)
    X, names, extras = build_features()
    assert names == bundle["names"], "features differentes de l'entrainement"
    P = bundle["model"].predict_proba(X)
    target = args.alive_target
    if args.em:
        from v5_em import em_prior
        graded = np.zeros(len(P), bool)
        for _, a, b in v1.half_video_blocks(v1.GAME):
            graded[a:b + 1] = True
        est = 1 - em_prior(P[graded], bundle["prior"])[2]
        target = float(np.clip(est, 0.50, 0.65))
        print("taux vivant estime par EM : %.1f %% -> utilise %.1f %%" % (100 * est, 100 * target))
    labels = decode_probs(P, extras, target)
    out = os.path.join(v1.RESULTS, args.name + ".csv")
    v1.write(out, labels)
    print("jeu vivant %.1f %% | segments D : %d | ecrit -> %s"
          % (100 * v3.alive_fraction(labels), v3.count_dead_segments(labels), os.path.relpath(out, ROOT)))
