"""v2 -- corrige le defaut principal de v1 : "joueurs lents" n'est PAS "jeu arrete".

    python ours/v2.py                       # ecrit evaluation/results/v2.csv
    python ours/v2.py --alive-target 0.58   # cible de temps de jeu (fraction des frames notees)

Changements par rapport a v1 (issus de ours/analyze.py + verification video) :
  1. La vitesse des joueurs ne declenche plus D seule : elle n'agit qu'en INTERACTION avec un
     indice venant du ballon (ballon immobile, ou invisible de facon prolongee).
  2. Ballon visible qui se deplace vite (une passe) = forte preuve de jeu VIVANT.
  3. Set piece Smart Tagging : D seulement AVANT la reprise (detectee dans la fenetre taguee),
     leger biais "vivant" apres.
  4. Calibrage NON supervise sur la litterature : le biais global de D est choisi pour que la
     fraction de jeu vivant sur TOUT le match vaille --alive-target (Barca ~60 % de ballon en
     jeu ; Siegle & Lames 2012 : arrets ~38 % du temps). Aucune verite terrain utilisee.
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
from v1 import (interpolate_ball, team_from_ball, team_from_tags, viterbi, write,  # noqa: E402
                before_kickoff_mask, sigmoid, LETTERS, RESULTS, EPS, TAG_WEIGHT_VISIBLE, H)
import load_smart_tagging  # noqa: E402

MAX_BALL_SPEED = 40.0     # m/s : au-dela = saut du tracker
MISSING_LONG = 2 * FPS    # trou de ballon > 2 s = "invisible de facon prolongee"


# ----------------------------------------------------------------------------------------------
# Indices sur le ballon
# ----------------------------------------------------------------------------------------------
def ball_speed(bx, by):
    """Vitesse du ballon (m/s) d'apres ses positions, lissee sur ~0.4 s. NaN si invisible."""
    v = np.hypot(np.diff(bx * LENGTH, prepend=np.nan), np.diff(by * WIDTH, prepend=np.nan)) * FPS
    v[v > MAX_BALL_SPEED] = np.nan
    return rolling_mean(v, FPS // 5)


def gap_length(bx):
    """Pour chaque frame ou le ballon est invisible : longueur du trou qui la contient (frames)."""
    miss = np.isnan(bx)
    out = np.zeros(len(bx), dtype=np.int64)
    i, n = 0, len(bx)
    while i < n:
        if miss[i]:
            j = i
            while j < n and miss[j]:
                j += 1
            out[i:j] = j - i
            i = j
        else:
            i += 1
    return out


def set_piece_phases(events, med_s, ball_moving, n):
    """Coupe chaque fenetre SET PIECES en [debut, reprise) = arret et [reprise, fin) = jeu.
    Reprise = 1re frame ou le ballon part vite apres 1 s ; sinon plus forte acceleration collective."""
    pre = np.zeros(n)
    post = np.zeros(n)
    for e in events:
        if (e["code"] or "").strip().lower() != "set pieces":
            continue
        s, t = max(0, e["start_frame"]), min(n, e["end_frame"])
        if t - s < FPS:
            continue
        moving = np.flatnonzero(ball_moving[s + FPS:t] > 0.5)
        if len(moving):
            r = s + FPS + moving[0]
        else:
            acc = np.diff(np.nan_to_num(med_s[s:t]))
            r = s + 1 + int(np.argmax(acc)) if len(acc) else (s + t) // 2
        pre[s:r] = 1.0
        post[r:t] = 1.0
    return pre, post


# ----------------------------------------------------------------------------------------------
def evidences():
    teams, px, py, ps, bx, by, bs = load_tracking()
    n = len(bx)
    events = load_smart_tagging.load()

    # equipe (inchange par rapport a v1)
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

    # arret de jeu : indices
    med, frac_slow, n_seen = speed_signals(ps)
    med_s = rolling_mean(med, FPS)
    slow = sigmoid((1.6 - np.nan_to_num(med_s, nan=1.0)) / 0.25)

    bv = ball_speed(bx, by)
    ball_moving = np.where(np.isnan(bv), 0.0, sigmoid((bv - 4.0) / 1.0))      # passe / conduite
    ball_static = np.where(np.isnan(bv), 0.0, sigmoid((1.0 - bv) / 0.3))      # ballon pose
    ball_static = rolling_mean(ball_static, FPS // 2)
    missing_long = (gap_length(bx) > MISSING_LONG).astype(float)
    out_recent = (rolling_mean(np.nan_to_num(ball_out(bx, by)), FPS) > 0.15).astype(float)
    sp_pre, sp_post = set_piece_phases(events, med_s, ball_moving, n)

    # logit SANS biais (le biais est calibre ensuite)
    z = (3.5 * out_recent
         + 2.0 * sp_pre
         - 1.0 * sp_post
         + slow * (1.8 * ball_static + 1.2 * missing_long)   # lent ET indice ballon
         + 0.4 * slow                                       # lent seul : quasi rien
         - 3.0 * ball_moving)                               # le ballon circule : vivant
    no_players = n_seen < 8
    return p_h, z, no_players, before_kickoff_mask(events, n)


def decode(p_h, z, no_players, kickoffs, bias, switch_team, switch_dead):
    p_dead = sigmoid(z + bias)
    p_dead[no_players] = 0.95
    p_dead = np.clip(p_dead, EPS, 1 - EPS)
    logp = np.stack([np.log(1 - p_dead) + np.log(p_h),
                     np.log(1 - p_dead) + np.log(1 - p_h),
                     np.log(p_dead)], axis=1)
    labels = np.full(len(p_h), "X", dtype="<U1")
    for half, first, last in half_video_blocks(GAME):
        lp = logp[first:last + 1].copy()
        ko = [k for k in kickoffs if first - 30 * FPS <= k <= last]
        if ko and ko[0] > first:
            lp[: ko[0] - first, :2] = -1e6
        labels[first:last + 1] = LETTERS[viterbi(lp, switch_team, switch_dead)]
    return labels


def alive_fraction(labels):
    graded = labels != "X"
    return np.isin(labels[graded], ["H", "A"]).mean()


def count_dead_segments(labels):
    d = (labels == "D").astype(np.int8)
    return int(np.sum(np.diff(np.concatenate(([0], d))) == 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--switch", type=float, default=30.0, help="penalite H<->A (nats)")
    ap.add_argument("--dead", type=float, default=20.0, help="penalite vivant<->D (nats)")
    ap.add_argument("--alive-target", type=float, default=0.58,
                    help="fraction de jeu vivant visee sur les frames notees (litterature)")
    ap.add_argument("--name", default="v2")
    args = ap.parse_args()

    p_h, z, no_players, kickoffs = evidences()

    # calibrage du biais par dichotomie : plus de biais -> plus de D -> moins de jeu vivant
    lo, hi = -10.0, 6.0
    for it in range(14):
        mid = (lo + hi) / 2
        frac = alive_fraction(decode(p_h, z, no_players, kickoffs, mid, args.switch, args.dead))
        print("  calibrage %2d : biais %+6.2f -> jeu vivant %.1f %%" % (it + 1, mid, 100 * frac))
        if frac > args.alive_target:
            lo = mid
        else:
            hi = mid
    bias = (lo + hi) / 2
    labels = decode(p_h, z, no_players, kickoffs, bias, args.switch, args.dead)

    out = os.path.join(RESULTS, args.name + ".csv")
    write(out, labels)
    graded_min = np.sum(labels != "X") / FPS / 60
    print("\nbiais retenu %+.2f | jeu vivant %.1f %% (%.1f min) | segments D : %d"
          % (bias, 100 * alive_fraction(labels), alive_fraction(labels) * graded_min,
             count_dead_segments(labels)))
    print("ecrit ->", os.path.relpath(out, ROOT))
    print("\nEnsuite :\n  python evaluation/validate_submission.py", os.path.relpath(out, ROOT),
          "\n  python ours/analyze.py --sub", os.path.relpath(out, ROOT),
          "\n  python evaluation/run_evaluation.py | head -16")
