"""Features "evenements" : temps depuis / jusqu'aux evenements qui bornent un arret de jeu.

Idee (vue dans un autre repo public du challenge, reimplementee ici de zero) : un arret commence
quand le ballon sort, s'immobilise ou n'est plus controle, et finit quand il repart. Le modele
v5 ne voit que des moyennes/tendances a +/-1-2 s ; ici on lui donne la DISTANCE TEMPORELLE (passe
et futur, en secondes, plafonnee) a ces evenements, et la vitesse du ballon avant / apres.
Toutes les features sont symetriques H/A (aucune n'a de cote) -> rien a echanger en augmentation.
Systeme hors ligne : utilise le futur, comme le decodeur Viterbi.
"""
import numpy as np

from features import FPS, LENGTH, WIDTH, rolling_mean, ball_out
import v1
import v2
import v3

CAP_S = 30.0
STILL_MS = 0.6          # ballon "immobile" (m/s)
CTRL_R = 1.5            # controle : joueur a moins de 1.5 m...
CTRL_VMAX = 12.0        # ...d'un ballon plus lent que 12 m/s


def since(mask):
    """Secondes depuis la derniere frame ou mask est vrai (CAP_S si jamais)."""
    n = len(mask)
    idx = np.where(mask, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    out = np.where(last < 0, CAP_S * FPS, np.arange(n) - last) / FPS
    return np.minimum(out, CAP_S).astype(np.float32)


def until(mask):
    return since(mask[::-1])[::-1]


def event_features(px, py, ps, bx, by):
    bx_i, by_i = v1.interpolate_ball(bx, by)
    seen = ~np.isnan(bx_i)
    bv = v2.ball_speed(bx_i, by_i)
    out = np.nan_to_num(ball_out(bx_i, by_i)) > 0
    still = seen & (np.nan_to_num(bv, nan=99.0) < STILL_MS)
    d = np.hypot((px - bx_i[:, None]) * LENGTH, (py - by_i[:, None]) * WIDTH)
    dmin = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=1)
    ctrl = seen & (dmin < CTRL_R) & (np.nan_to_num(bv, nan=99.0) < CTRL_VMAX)
    s = np.where(ps > 12.0, np.nan, ps)
    with np.errstate(all="ignore"):
        mean_sp = np.nan_to_num(np.nanmean(s, axis=1))
        fast = np.nansum(s > 4.0, axis=1)
    burst = fast >= 3                      # au moins 3 joueurs qui sprintent

    bvf = np.nan_to_num(bv, nan=-1.0)
    f = {
        "ev_since_out": since(out), "ev_until_out": until(out),
        "ev_since_still": since(still), "ev_until_still": until(still),
        "ev_since_ctrl": since(ctrl), "ev_until_ctrl": until(ctrl),
        "ev_since_seen": since(seen), "ev_until_seen": until(seen),
        "ev_since_burst": since(burst), "ev_until_burst": until(burst),
        "ev_still_m1": np.nan_to_num(rolling_mean(still.astype(float), FPS)),
        "ev_ctrl_m1": np.nan_to_num(rolling_mean(ctrl.astype(float), FPS)),
        "ev_bv_before1": np.nan_to_num(v3.shift(bvf, FPS), nan=-1.0),
        "ev_bv_before2": np.nan_to_num(v3.shift(bvf, 2 * FPS), nan=-1.0),
        "ev_bv_after1": np.nan_to_num(v3.shift(bvf, -FPS), nan=-1.0),
        "ev_bv_after2": np.nan_to_num(v3.shift(bvf, -2 * FPS), nan=-1.0),
        "ev_sp_m6": np.nan_to_num(rolling_mean(mean_sp, 3 * FPS)),
        "ev_sp_delta2": np.nan_to_num(v3.shift(rolling_mean(mean_sp, FPS), -2 * FPS)
                                      - v3.shift(rolling_mean(mean_sp, FPS), 2 * FPS)),
    }
    names = sorted(f)
    return np.stack([np.asarray(f[k], dtype=np.float32) for k in names], axis=1), names
