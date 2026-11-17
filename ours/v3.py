"""v3.1 -- arrets de jeu par EVENEMENTS (debut / reprise) + transitions dynamiques.

    python ours/v3.py                        # ecrit evaluation/results/v3.csv
    python ours/v3.py --alive-target 0.58

Probleme de v2 (analyse + video) : pendant un arret, le ballon BOUGE aussi (on le renvoie, le
gardien le replace apres un tir rate) -> v2 croyait que le jeu avait repris.

Idees et sources :
  1. Transitions dynamiques (PathCRF, Kim et al. 2026) : entrer en D n'est "bon marche" qu'autour
     d'un evenement de DEBUT d'arret ; sortir de D qu'autour d'un evenement de REPRISE.
     Ailleurs, changer d'etat coute cher -> un ballon qui roule pendant l'arret ne casse plus D.
  2. Debut d'arret : ballon sorti, tir rate (ballon rapide vers le but puis perdu/sorti),
     deceleration collective brutale (Lang et al. 2022 : le mouvement des joueurs depend du statut).
  3. Reprise : acceleration collective + ballon frappe (Lang et al. 2022).
  4. Signatures de reprise imposees par les Lois du jeu (cf. Vidal-Codina et al. 2022) :
     six metres, corner, touche, coup d'envoi, coup franc (mur a 9,15 m) -> forte preuve de D.
  5. Calibrage non supervise du biais de D sur la litterature (comme v2).
Aucune verite terrain, aucun label fait main.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks  # noqa: E402
from features import (FPS, LENGTH, WIDTH, load_tracking, rolling_mean,  # noqa: E402
                      speed_signals, ball_out)
from v1 import (interpolate_ball, team_from_ball, team_from_tags, write,  # noqa: E402
                before_kickoff_mask, sigmoid, LETTERS, RESULTS, EPS, TAG_WEIGHT_VISIBLE, H, A)
from v2 import ball_speed, gap_length, set_piece_phases, alive_fraction, count_dead_segments  # noqa: E402
import load_smart_tagging  # noqa: E402

# --- Geometrie du terrain (metres), Lois du jeu ---
GOAL_AREA_DEPTH, GOAL_AREA_HALF = 5.5, 9.16
PEN_DEPTH, PEN_HALF = 16.5, 20.16
WALL_DIST = 9.15

# --- Transitions dynamiques ---
EVENT_WINDOW = int(1.5 * FPS)     # un evenement "ouvre" la porte +/- 1.5 s
COST_HIGH = 35.0                  # changer vivant<->D sans evenement
COST_LOW = 6.0                    # changer vivant<->D pres d'un evenement


def spread(mask, w):
    """Dilate un masque booleen de +/- w frames."""
    c = np.concatenate(([0], np.cumsum(mask.astype(np.int64))))
    i = np.arange(len(mask))
    return (c[np.minimum(len(mask), i + w + 1)] - c[np.maximum(0, i - w)]) > 0


def shift(x, k):
    """x decale de k frames (k>0 : valeur k frames plus tot)."""
    out = np.full_like(x, np.nan, dtype=np.float64)
    if k > 0:
        out[k:] = x[:-k]
    elif k < 0:
        out[:k] = x[-k:]
    else:
        out[:] = x
    return out


# ----------------------------------------------------------------------------------------------
# Signatures de reprise (Lois du jeu)
# ----------------------------------------------------------------------------------------------
def restart_signatures(px, py, teams, bx, by, bv, slow):
    """Score [0,1] par frame : la configuration ressemble a une reprise de jeu arretee."""
    n = len(bx)
    X, Y = px * LENGTH, py * WIDTH                       # joueurs en metres
    BX, BY = bx * LENGTH, by * WIDTH
    vis = ~np.isnan(BX)
    static = vis & (np.nan_to_num(bv, nan=99) < 1.5)
    cy = WIDTH / 2
    sig = np.zeros(n)

    # distances joueurs-ballon, equipe la plus proche (= equipe qui va jouer la reprise)
    d = np.hypot(X - BX[:, None], Y - BY[:, None])
    d_h = np.nanmin(np.where(teams[None, :] == H, d, np.nan), axis=1)
    d_a = np.nanmin(np.where(teams[None, :] == A, d, np.nan), axis=1)
    kicker = np.where(np.nan_to_num(d_h, nan=99) <= np.nan_to_num(d_a, nan=99), H, A)
    d_opp = np.where(kicker == H, d_a, d_h)

    # 1) six metres : ballon immobile dans une surface de but, aucun adversaire dans la surface
    for goal_x in (0.0, LENGTH):
        in_ga = static & (np.abs(BX - goal_x) <= GOAL_AREA_DEPTH + 1) & (np.abs(BY - cy) <= GOAL_AREA_HALF + 1)
        in_box = (np.abs(X - goal_x) <= PEN_DEPTH) & (np.abs(Y - cy) <= PEN_HALF)
        opp = teams[None, :] != kicker[:, None]
        opp_in_box = np.sum(in_box & opp & ~np.isnan(X), axis=1)
        sig = np.maximum(sig, (in_ga & (opp_in_box == 0)).astype(float))

    # 2) corner : ballon (vu) a moins de 2 m d'un coin
    for cx in (0.0, LENGTH):
        for cyy in (0.0, WIDTH):
            sig = np.maximum(sig, (vis & (np.hypot(BX - cx, BY - cyy) < 2.0)).astype(float) * 0.9)

    # 3) touche : un joueur hors du terrain par la ligne de cote, ballon pres de lui ou invisible
    outside = (Y < -0.3) | (Y > WIDTH + 0.3)
    near_ball = (d < 2.0) | ~vis[:, None]
    sig = np.maximum(sig, (np.any(outside & near_ball & ~np.isnan(X), axis=1) & (slow > 0.5)) * 0.8)

    # 4) coup d'envoi : ballon au centre, chaque equipe entierement dans sa moitie
    centre = vis & (np.hypot(BX - LENGTH / 2, BY - cy) < 1.5)
    with np.errstate(all="ignore"):
        left_h = np.nanmean(np.where(teams[None, :] == H, X < LENGTH / 2 + 1, np.nan), axis=1)
        left_a = np.nanmean(np.where(teams[None, :] == A, X < LENGTH / 2 + 1, np.nan), axis=1)
    halves = ((left_h > 0.95) & (left_a < 0.05)) | ((left_h < 0.05) & (left_a > 0.95))
    sig = np.maximum(sig, (centre & halves).astype(float))

    # 5) coup franc : ballon immobile, un joueur dessus, adversaires a >= ~8 m, tout le monde lent
    fk = static & (np.minimum(np.nan_to_num(d_h, nan=99), np.nan_to_num(d_a, nan=99)) < 2.0) \
        & (np.nan_to_num(d_opp, nan=0) >= WALL_DIST - 1.5) & (slow > 0.6)
    sig = np.maximum(sig, fk * 0.7)

    return rolling_mean(sig, FPS // 2)


# ----------------------------------------------------------------------------------------------
# Evenements de debut d'arret et de reprise
# ----------------------------------------------------------------------------------------------
def onset_and_restart(bx, by, bv, med_s, out_now):
    BX = bx * LENGTH
    n = len(bx)
    # tir rate : ballon rapide (> 15 m/s) a < 30 m d'une ligne de but et qui s'en approche,
    # puis perdu ou sorti dans les 2 s
    fast = np.nan_to_num(bv, nan=0) > 15
    dx = np.diff(BX, prepend=np.nan) * FPS
    toward = ((BX < 30) & (dx < -5)) | ((BX > LENGTH - 30) & (dx > 5))
    shot = fast & np.nan_to_num(toward, nan=False).astype(bool)
    lost_or_out = np.isnan(bx) | out_now
    after = np.zeros(n, bool)                            # perdu/sorti dans les 2 s qui suivent
    c = np.concatenate(([0], np.cumsum(lost_or_out)))
    i = np.arange(n)
    after = (c[np.minimum(n, i + 2 * FPS + 1)] - c[np.minimum(n, i + 1)]) > FPS // 2
    missed_shot = shot & after

    # deceleration / acceleration collectives (variation de la vitesse mediane sur 2 s)
    before, later = shift(med_s, 2 * FPS), shift(med_s, -2 * FPS)
    decel = (np.nan_to_num(before - med_s) > 1.2) & (np.nan_to_num(med_s, nan=9) < 1.6)
    accel = (np.nan_to_num(later - med_s) > 1.2) & (np.nan_to_num(med_s, nan=9) < 1.6)

    # ballon frappe : passe de immobile/invisible a rapide
    prev = shift(bv, FPS // 2)
    was_still = ~np.isnan(prev) & (np.nan_to_num(prev, nan=99) < 2)   # VU et immobile (v3.1)
    kicked = was_still & (np.nan_to_num(bv, nan=0) > 6)

    onset = out_now | missed_shot | decel
    restart = accel | kicked
    return onset, restart, missed_shot


# ----------------------------------------------------------------------------------------------
# Viterbi a transitions dynamiques
# ----------------------------------------------------------------------------------------------
def viterbi_dynamic(logp, switch_team, cost_in, cost_out):
    """cost_in[t] : penalite pour passer vivant -> D a la frame t ; cost_out[t] : D -> vivant."""
    n = len(logp)
    logp = np.asarray(logp, dtype=np.float64).tolist()      # listes Python : boucle ~10x plus rapide
    cost_in, cost_out = list(map(float, cost_in)), list(map(float, cost_out))
    s0, s1, s2 = logp[0]
    back = [None] * n
    for t in range(1, n):
        ci, co = cost_in[t], cost_out[t]
        # vers H (0)
        c00, c10, c20 = s0, s1 - switch_team, s2 - co
        b0 = 0 if c00 >= c10 and c00 >= c20 else (1 if c10 >= c20 else 2)
        # vers A (1)
        c01, c11, c21 = s0 - switch_team, s1, s2 - co
        b1 = 1 if c11 >= c01 and c11 >= c21 else (0 if c01 >= c21 else 2)
        # vers D (2)
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


# ----------------------------------------------------------------------------------------------
def evidences():
    teams, px, py, ps, bx, by, bs = load_tracking()
    n = len(bx)
    events = load_smart_tagging.load()

    # equipe : comme v2 (sera retravaille en v4 : possession par controle)
    bx_i, by_i = interpolate_ball(bx, by)
    p_ball = team_from_ball(px, py, bx_i, by_i, teams)
    p_tag = team_from_tags(events, n)
    p_h = np.full(n, 0.5)
    vis, tag = ~np.isnan(p_ball), ~np.isnan(p_tag)
    p_h[vis] = p_ball[vis]
    both = vis & tag
    p_h[both] = (1 - TAG_WEIGHT_VISIBLE) * p_ball[both] + TAG_WEIGHT_VISIBLE * p_tag[both]
    p_h[~vis & tag] = p_tag[~vis & tag]
    p_h = np.clip(p_h, EPS, 1 - EPS)

    med, frac_slow, n_seen = speed_signals(ps)
    med_s = rolling_mean(med, FPS)
    slow = sigmoid((1.6 - np.nan_to_num(med_s, nan=1.0)) / 0.25)
    bv = ball_speed(bx, by)
    ball_moving = np.where(np.isnan(bv), 0.0, sigmoid((bv - 4.0) / 1.0))
    ball_static = rolling_mean(np.where(np.isnan(bv), 0.0, sigmoid((1.0 - bv) / 0.3)), FPS // 2)
    missing_long = (gap_length(bx) > 2 * FPS).astype(float)
    out_now = np.nan_to_num(ball_out(bx, by)) > 0
    out_recent = (rolling_mean(out_now.astype(float), FPS) > 0.15).astype(float)
    sp_pre, sp_post = set_piece_phases(events, med_s, ball_moving, n)
    sig = restart_signatures(px, py, teams, bx, by, bv, slow)

    onset, restart, missed_shot = onset_and_restart(bx, by, bv, med_s, out_now)
    # apres un tir rate, l'arret s'installe : on prolonge l'indice ~3 s
    shot_after = spread(missed_shot, 0) | (rolling_mean(missed_shot.astype(float), 3 * FPS) > 0)

    # Emission : comme v2, MAIS le ballon qui bouge pese moins quand les joueurs sont lents
    # (pendant un arret, on renvoie / replace le ballon) ; + signatures + tir rate.
    z = (3.5 * out_recent
         + 2.0 * sp_pre
         - 1.0 * sp_post
         + 3.0 * sig
         + 1.5 * shot_after * slow
         + slow * (1.8 * ball_static + 1.2 * missing_long)
         + 0.4 * slow
         - 3.0 * ball_moving * (1 - 0.6 * slow))

    cost_in = np.where(spread(onset, EVENT_WINDOW) | (sig > 0.5), COST_LOW, COST_HIGH)
    cost_out = np.where(spread(restart, EVENT_WINDOW), COST_LOW, COST_HIGH)

    stats = {"tirs rates": int(np.sum(np.diff(missed_shot.astype(int), prepend=0) == 1)),
             "evts debut": int(np.sum(np.diff(onset.astype(int), prepend=0) == 1)),
             "evts reprise": int(np.sum(np.diff(restart.astype(int), prepend=0) == 1)),
             "frames signature": int(np.sum(sig > 0.5))}
    return p_h, z, n_seen < 8, before_kickoff_mask(events, n), cost_in, cost_out, stats


MIN_DEAD = 2 * FPS        # un arret de moins de 2 s n'existe pas (touche ~9 s en moyenne)
MIN_POSS = FPS            # une possession de moins de 1 s = clignotement


def enforce_min_durations(lab):
    """Supprime les segments trop courts en les rattachant a leur voisin (v3.1)."""
    lab = lab.copy()
    for _ in range(3):
        changed = False
        n = len(lab)
        starts = np.flatnonzero(np.concatenate(([True], lab[1:] != lab[:-1])))
        ends = np.concatenate((starts[1:], [n]))
        for a, b in zip(starts, ends):
            short = (lab[a] == "D" and b - a < MIN_DEAD) or (lab[a] in "HA" and b - a < MIN_POSS)
            if not short:
                continue
            left = lab[a - 1] if a > 0 else None
            right = lab[b] if b < n else None
            fill = left if left is not None else right
            if left is not None and right is not None and left != right and lab[a] in "HA":
                fill = right if right != "D" else left
            if fill is not None and fill != lab[a]:
                lab[a:b] = fill
                changed = True
        if not changed:
            break
    return lab


def decode(p_h, z, no_players, kickoffs, cost_in, cost_out, bias, switch_team):
    p_dead = np.clip(np.where(no_players, 0.95, sigmoid(z + bias)), EPS, 1 - EPS)
    logp = np.stack([np.log(1 - p_dead) + np.log(p_h),
                     np.log(1 - p_dead) + np.log(1 - p_h),
                     np.log(p_dead)], axis=1)
    labels = np.full(len(p_h), "X", dtype="<U1")
    for half, first, last in half_video_blocks(GAME):
        lp = logp[first:last + 1].copy()
        ko = [k for k in kickoffs if first - 30 * FPS <= k <= last]
        if ko and ko[0] > first:
            lp[: ko[0] - first, :2] = -1e6
        path = viterbi_dynamic(lp, switch_team, cost_in[first:last + 1], cost_out[first:last + 1])
        labels[first:last + 1] = enforce_min_durations(LETTERS[path])
    return labels


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--switch", type=float, default=30.0, help="penalite H<->A (nats)")
    ap.add_argument("--alive-target", type=float, default=0.58)
    ap.add_argument("--name", default="v3_1")
    args = ap.parse_args()

    p_h, z, no_players, kickoffs, cost_in, cost_out, stats = evidences()
    print("indices :", stats)

    lo, hi = -10.0, 6.0
    for it in range(12):
        mid = (lo + hi) / 2
        frac = alive_fraction(decode(p_h, z, no_players, kickoffs, cost_in, cost_out, mid, args.switch))
        print("  calibrage %2d : biais %+6.2f -> jeu vivant %.1f %%" % (it + 1, mid, 100 * frac))
        lo, hi = (mid, hi) if frac > args.alive_target else (lo, mid)
    bias = (lo + hi) / 2
    labels = decode(p_h, z, no_players, kickoffs, cost_in, cost_out, bias, args.switch)

    out = os.path.join(RESULTS, args.name + ".csv")
    write(out, labels)
    print("\nbiais %+.2f | jeu vivant %.1f %% | segments D : %d"
          % (bias, 100 * alive_fraction(labels), count_dead_segments(labels)))
    print("ecrit ->", os.path.relpath(out, ROOT))
    print("\nEnsuite :\n  python ours/analyze.py --sub", os.path.relpath(out, ROOT),
          "\n  python evaluation/run_evaluation.py | head -16")
