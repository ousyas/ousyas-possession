"""v1 -- possession frame par frame : evidences par frame + decodage de Viterbi.

    python ours/v1.py                         # ecrit evaluation/results/v1.csv
    python ours/v1.py --switch 40 --dead 25   # autres penalites de changement

Pipeline (aucune verite terrain utilisee, aucun label fait main) :
  1. Ballon : interpolation lineaire des trous courts (<= 0.5 s).
  2. Equipe (H vs A) : distance du joueur le plus proche de chaque equipe au ballon (softmin),
     + a priori faible du Smart Tagging (phase offensive/defensive taguee par equipe).
  3. Ballon mort (D) : score logistique combinant vitesse collective basse, ballon hors terrain,
     ballon invisible, set piece Smart Tagging, et "avant le coup d'envoi" d'une mi-temps.
  4. Viterbi sur 3 etats (H, A, D) : maximise la somme des log-probas par frame moins une
     penalite a chaque changement d'etat -> des possessions entieres, pas du clignotement.
  5. Hors des mi-temps notees : X (ignore par le scorer).

Tous les seuils/poids sont fixes a priori sur des arguments physiques, PAS regles sur les
5 minutes publiques (interdit et trompeur, cf. EVALUATION.md).
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
import load_smart_tagging  # noqa: E402  (deja dans le path via features)

RESULTS = os.path.join(ROOT, "evaluation", "results")
H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])

# --- Parametres (a priori) ---------------------------------------------------------------------
MAX_GAP_INTERP = FPS // 2      # trous de ballon <= 0.5 s : interpolation lineaire
TEAM_TAU = 2.5                 # m : echelle du softmin sur les distances au ballon
TAG_STRENGTH = 0.65            # proba donnee a l'equipe indiquee par le Smart Tagging
TAG_WEIGHT_VISIBLE = 0.2       # poids du tag quand le ballon est visible
EPS = 1e-4

OFFENSIVE = {"build up", "progression", "creating chances", "attacking transition",
             "ball in final third", "ball in the box", "players in the box", "long ball"}
DEFENSIVE = {"defending in defensive third", "defending in middle third",
             "defending in attacking third", "defensive transition"}
TEAM_CODE = {"barcelona": H, "atletic club": A, "athletic club": A}


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


# ----------------------------------------------------------------------------------------------
# 1. Ballon
# ----------------------------------------------------------------------------------------------
def interpolate_ball(bx, by, max_gap=MAX_GAP_INTERP):
    """Remplit lineairement les trous de <= max_gap frames entre deux positions vues."""
    bx, by = bx.astype(np.float64).copy(), by.astype(np.float64).copy()
    seen = np.flatnonzero(~np.isnan(bx))
    for a, b in zip(seen[:-1], seen[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            w = np.arange(1, gap + 1) / (gap + 1)
            bx[a + 1:b] = bx[a] + w * (bx[b] - bx[a])
            by[a + 1:b] = by[a] + w * (by[b] - by[a])
    return bx, by


# ----------------------------------------------------------------------------------------------
# 2. Equipe
# ----------------------------------------------------------------------------------------------
def team_from_ball(px, py, bx, by, teams):
    """P(H) par frame d'apres le joueur le plus proche de chaque equipe. NaN si pas de ballon."""
    dx = (px - bx[:, None]) * LENGTH
    dy = (py - by[:, None]) * WIDTH
    dist = np.hypot(dx, dy)
    d_h = np.nanmin(np.where(teams[None, :] == H, dist, np.nan), axis=1)
    d_a = np.nanmin(np.where(teams[None, :] == A, dist, np.nan), axis=1)
    p_h = sigmoid((d_a - d_h) / TEAM_TAU)          # plus H est proche que A, plus P(H) -> 1
    p_h[np.isnan(d_h) & ~np.isnan(d_a)] = 0.2
    p_h[np.isnan(d_a) & ~np.isnan(d_h)] = 0.8
    return p_h                                     # NaN si ballon ou tous les joueurs absents


def team_from_tags(events, n):
    """P(H) a priori depuis les phases Smart Tagging (NaN hors phase)."""
    p = np.full(n, np.nan)
    for e in events:
        code = (e["code"] or "").strip().lower()
        team = TEAM_CODE.get((e["tags"].get("Team") or "").strip().lower())
        if team is None or code not in OFFENSIVE | DEFENSIVE:
            continue
        has_ball = team if code in OFFENSIVE else 1 - team
        val = TAG_STRENGTH if has_ball == H else 1 - TAG_STRENGTH
        s, t = max(0, e["start_frame"]), min(n, e["end_frame"])
        p[s:t] = val
    return p


# ----------------------------------------------------------------------------------------------
# 3. Ballon mort
# ----------------------------------------------------------------------------------------------
def dead_probability(ps, bx_raw, by_raw, bx_i, events, n):
    med, frac_slow, n_seen = speed_signals(ps)
    med_s = rolling_mean(med, FPS)                  # vitesse mediane lissee sur ~2 s
    slow_s = rolling_mean(frac_slow, FPS // 2)

    slow_score = sigmoid((1.6 - np.nan_to_num(med_s, nan=1.0)) / 0.25)   # ~1 sous 1.3 m/s
    out = np.nan_to_num(ball_out(bx_raw, by_raw))
    out_recent = rolling_mean(out, FPS) > 0.15     # ballon vu dehors dans les ~2 s autour
    missing = np.isnan(bx_i).astype(float)

    in_sp = np.zeros(n)
    for e in events:
        if (e["code"] or "").strip().lower() == "set pieces":
            in_sp[max(0, e["start_frame"]):min(n, e["end_frame"])] = 1.0

    z = (-3.0
         + 2.5 * slow_score
         + 1.5 * np.nan_to_num(slow_s, nan=0.0)
         + 3.5 * out_recent
         + 0.8 * missing
         + 1.0 * in_sp)
    p_dead = sigmoid(z)
    p_dead[n_seen < 8] = 0.95                      # presque personne de tracke : pas de jeu
    return p_dead


def before_kickoff_mask(events, n):
    """D avant le coup d'envoi de chaque mi-temps, detecte via le tag 'Kick Off (Start)'.
    La reprise = premier pic de vitesse dans la fenetre taguee (fait plus bas)."""
    starts = sorted(e["start_frame"] for e in events
                    if (e["code"] or "").strip().lower() == "set pieces"
                    and (e["tags"].get("Type") or "").strip().lower() == "kick off (start)")
    return starts


# ----------------------------------------------------------------------------------------------
# 4. Viterbi
# ----------------------------------------------------------------------------------------------
def viterbi(logp, switch_team, switch_dead):
    """logp: (n, 3). Penalite switch_team pour H<->A, switch_dead pour vivant<->D."""
    n = len(logp)
    trans = np.array([[0.0, -switch_team, -switch_dead],
                      [-switch_team, 0.0, -switch_dead],
                      [-switch_dead, -switch_dead, 0.0]])
    score = logp[0].copy()
    back = np.zeros((n, 3), dtype=np.int8)
    for t in range(1, n):
        cand = score[:, None] + trans               # cand[i, j] : venir de i vers j
        back[t] = np.argmax(cand, axis=0)
        score = cand[back[t], [0, 1, 2]] + logp[t]
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax(score))
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


# ----------------------------------------------------------------------------------------------
def build(switch_team, switch_dead):
    teams, px, py, ps, bx, by, bs = load_tracking()
    n = len(bx)
    events = load_smart_tagging.load()

    bx_i, by_i = interpolate_ball(bx, by)
    p_ball = team_from_ball(px, py, bx_i, by_i, teams)
    p_tag = team_from_tags(events, n)

    # Combinaison des evidences d'equipe
    p_h = np.full(n, 0.5)
    vis = ~np.isnan(p_ball)
    tag = ~np.isnan(p_tag)
    p_h[vis] = p_ball[vis]
    both = vis & tag
    p_h[both] = (1 - TAG_WEIGHT_VISIBLE) * p_ball[both] + TAG_WEIGHT_VISIBLE * p_tag[both]
    p_h[~vis & tag] = p_tag[~vis & tag]
    p_h = np.clip(p_h, EPS, 1 - EPS)

    p_dead = np.clip(dead_probability(ps, bx, by, bx_i, events, n), EPS, 1 - EPS)

    logp = np.stack([np.log(1 - p_dead) + np.log(p_h),
                     np.log(1 - p_dead) + np.log(1 - p_h),
                     np.log(p_dead)], axis=1)

    labels = np.full(n, "X", dtype="<U1")
    kickoffs = before_kickoff_mask(events, n)
    for half, first, last in half_video_blocks(GAME):
        seg = slice(first, last + 1)
        lp = logp[seg].copy()
        # avant le coup d'envoi tague de cette mi-temps : on force D
        ko = [k for k in kickoffs if first - 30 * FPS <= k <= last]
        if ko and ko[0] > first:
            lp[: ko[0] - first, :2] = -1e6
        path = viterbi(lp, switch_team, switch_dead)
        labels[seg] = LETTERS[path]
    return labels


def write(path, labels):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("frame,label\n")
        f.writelines("%d,%s\n" % (i, l) for i, l in enumerate(labels))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--switch", type=float, default=30.0, help="penalite H<->A (nats)")
    ap.add_argument("--dead", type=float, default=20.0, help="penalite vivant<->D (nats)")
    ap.add_argument("--name", default="v1")
    args = ap.parse_args()

    labels = build(args.switch, args.dead)
    out = os.path.join(RESULTS, args.name + ".csv")
    write(out, labels)
    counts = {c: int(np.sum(labels == c)) for c in "HADX"}
    print("labels:", counts)
    print("ecrit ->", os.path.relpath(out, ROOT))
    print("\nEnsuite :\n  python evaluation/validate_submission.py", os.path.relpath(out, ROOT),
          "\n  python evaluation/run_evaluation.py")
