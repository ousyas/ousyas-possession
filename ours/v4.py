"""v4 -- equipe en possession par TOUCHES + CONTROLE (au lieu du joueur le plus proche).

    python ours/v4.py                  # ecrit evaluation/results/v4.csv
    python ours/v4.py --alive-target 0.58

Probleme vise : precision d'equipe ~0.77 sur IDSSE (toutes versions), parce que
"joueur le plus proche sur une frame" donne le ballon a l'adversaire sous lequel il passe.

Idees et sources :
  1. Detecteur de touches : max local d'acceleration du ballon lissee (>= ~4 m/s2, Link &
     Hoernig 2017) OU changement net de direction (Vidal-Codina 2022), attribue au joueur le
     plus proche s'il est a < 2 m (rayon elargi : ATD bruite).
  2. Ballon en vol : entre deux touches, l'equipe = celle de la derniere touche.
  3. Changement d'equipe seulement avec CONTROLE (CHALLENGE.md) : 2 touches consecutives de la
     nouvelle equipe, ou ballon pres d'un de ses joueurs >= 0.8 s ; le changement est date a la
     PREMIERE touche (frontiere precise).
  4. Duel (adversaire a < 1.5 m au moment de la touche) : dispute -> on garde l'equipe
     precedente (Vidal-Codina : "le duel l'emporte" ; CHALLENGE.md : ballon dispute = equipe
     precedente).
  5. Plan B (reception en course sans changement visible) : ballon colle a un joueur de
     l'autre equipe >= 1.2 s, aucun joueur de l'equipe en cours a < 3 m.
  6. Reprise apres un long trou (> 3 s sans touche) : la 1re touche suffit.
Arrets de jeu (D) : inchanges par rapport a v3.1.
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from features import FPS, LENGTH, WIDTH, rolling_mean  # noqa: E402
import v1  # noqa: E402
import v3  # noqa: E402

ACC_TOUCH = 4.0           # m/s2 (Link & Hoernig 2017)
ANGLE_TOUCH = 35.0        # degres de changement de direction
R_TOUCH = 2.0             # m : joueur qui touche
R_DUEL = 1.5              # m : adversaire assez proche -> duel
R_CTRL = 2.0              # m : controle apres la touche
CTRL_FRAMES = int(0.8 * FPS)
PROX_FRAMES = int(1.2 * FPS)
RESTART_GAP = 3 * FPS
NMS = int(0.3 * FPS)      # une seule touche par fenetre de 0.3 s
P_CONF = 0.85             # confiance donnee a l'etat d'equipe
SG7 = np.array([-2, 3, 6, 7, 6, 3, -2]) / 21.0   # Savitzky-Golay ordre 2, 7 frames


def smooth(x):
    """Savitzky-Golay (7 frames) robuste aux NaN (NaN conserves)."""
    y = np.convolve(np.nan_to_num(x), SG7, "same")
    bad = np.convolve(np.isnan(x).astype(float), np.ones(7), "same") > 0
    y[bad] = np.nan
    return y


def nearest_by_team(px, py, bx, by, teams):
    dx = (px - bx[:, None]) * LENGTH
    dy = (py - by[:, None]) * WIDTH
    d = np.hypot(dx, dy)
    d_h = np.where(teams[None, :] == 0, d, np.inf)
    d_a = np.where(teams[None, :] == 1, d, np.inf)
    d_h = np.where(np.isnan(d_h), np.inf, d_h).min(axis=1)
    d_a = np.where(np.isnan(d_a), np.inf, d_a).min(axis=1)
    return d_h, d_a


def detect_touches(bx_m, by_m, d_h, d_a):
    """Retourne (frames, equipe, dispute) des touches detectees."""
    vx = np.gradient(bx_m) * FPS
    vy = np.gradient(by_m) * FPS
    ax = smooth(np.gradient(vx) * FPS)
    ay = smooth(np.gradient(vy) * FPS)
    acc = np.hypot(ax, ay)
    # changement de direction entre t-3 et t+3
    v_prev = np.stack([v3.shift(vx, 3), v3.shift(vy, 3)])
    v_next = np.stack([v3.shift(vx, -3), v3.shift(vy, -3)])
    num = (v_prev * v_next).sum(0)
    den = np.hypot(*v_prev) * np.hypot(*v_next)
    with np.errstate(all="ignore"):
        ang = np.degrees(np.arccos(np.clip(num / den, -1, 1)))
    speed = np.hypot(vx, vy)

    d_min = np.minimum(d_h, d_a)
    acc_n = np.nan_to_num(acc)
    loc_max = (acc_n >= np.maximum(v3.shift(acc_n, 1), v3.shift(acc_n, -1)))
    cand = ((loc_max & (acc_n >= ACC_TOUCH)) |
            ((np.nan_to_num(ang) >= ANGLE_TOUCH) & (np.nan_to_num(speed) > 2))) & (d_min <= R_TOUCH)
    idx = np.flatnonzero(cand)

    # non-maximum suppression : garder la plus forte touche par fenetre
    keep, last = [], -10 ** 9
    for i in idx:
        if i - last > NMS:
            keep.append(i)
            last = i
        elif acc_n[i] > acc_n[keep[-1]]:
            keep[-1] = i
            last = i
    keep = np.array(keep, dtype=np.int64)
    team = np.where(d_h[keep] <= d_a[keep], 0, 1)
    d_opp = np.where(team == 0, d_a[keep], d_h[keep])
    contested = d_opp < R_DUEL
    return keep, team, contested


def team_state(n, touches, team, contested, d_h, d_a, ball_vis):
    """Etat d'equipe frame par frame (0=H, 1=A, -1 inconnu) + confiance."""
    state = np.full(n, -1, dtype=np.int8)
    # changements decides : liste de (frame_debut, equipe)
    switches = []
    cur, last_touch = -1, -10 ** 9
    pending, p_start, p_count = -1, 0, 0
    for k, (t, T, c) in enumerate(zip(touches, team, contested)):
        if cur == -1 or t - last_touch > RESTART_GAP:          # debut / reprise
            if cur != T:
                switches.append((t, T))
            cur, pending = T, -1
        elif T == cur:
            pending = -1                                       # l'equipe en cours confirme
        elif not c:
            if pending == T:
                p_count += 1
            else:
                pending, p_start, p_count = T, t, 1
            # controle : ballon pres d'un joueur de T pendant CTRL_FRAMES apres la touche
            dT = d_h if T == 0 else d_a
            win = dT[t:t + CTRL_FRAMES]
            controlled = len(win) == CTRL_FRAMES and np.all(win <= R_CTRL)
            if p_count >= 2 or controlled:
                switches.append((p_start, T))
                cur, pending = T, -1
        last_touch = t

    # Plan B : ballon colle a l'autre equipe longtemps, sans touche detectee
    for T, dT, dO in ((0, d_h, d_a), (1, d_a, d_h)):
        cond = ball_vis & (dT < 1.5) & (dO > 3.0)
        c = np.concatenate(([0], cond.astype(np.int8), [0]))
        starts = np.flatnonzero(np.diff(c) == 1)
        ends = np.flatnonzero(np.diff(c) == -1)
        for s, e in zip(starts, ends):
            if e - s >= PROX_FRAMES:
                switches.append((s, T))
    switches.sort()

    cur, j = -1, 0
    for t in range(n):
        while j < len(switches) and switches[j][0] <= t:
            cur = switches[j][1]
            j += 1
        state[t] = cur
    return state


def team_probability(px, py, bx, by, teams, p_tag=None):
    bx_i, by_i = v1.interpolate_ball(bx, by)
    bx_m, by_m = bx_i * LENGTH, by_i * WIDTH
    d_h, d_a = nearest_by_team(px, py, bx_i, by_i, teams)
    vis = ~np.isnan(bx_i)
    touches, team, contested = detect_touches(bx_m, by_m, d_h, d_a)
    state = team_state(len(bx), touches, team, contested, d_h, d_a, vis)

    p_h = np.full(len(bx), 0.5)
    p_h[state == 0] = P_CONF
    p_h[state == 1] = 1 - P_CONF
    # ballon invisible longtemps : confiance reduite (+ a priori Smart Tagging si present)
    gap_long = v3.gap_length(bx) > 2 * FPS
    p_h[gap_long] = 0.5 + (p_h[gap_long] - 0.5) * 0.6
    if p_tag is not None:
        tag = ~np.isnan(p_tag) & gap_long
        p_h[tag] = 0.7 * p_h[tag] + 0.3 * p_tag[tag]
    stats = {"touches": int(len(touches)), "disputees": int(contested.sum()),
             "changements": int(np.sum(np.diff(state[state >= 0]) != 0))}
    return np.clip(p_h, v1.EPS, 1 - v1.EPS), stats


def evidences():
    """Comme v3.evidences, avec l'equipe remplacee par le modele touches + controle."""
    p_h_old, z, no_players, kickoffs, cost_in, cost_out, st = v3.evidences()
    teams, px, py, ps, bx, by, bs = v3.load_tracking()
    events = v3.load_smart_tagging.load()
    p_tag = v1.team_from_tags(events, len(bx)) if events else None
    p_h, st4 = team_probability(px, py, bx, by, teams, p_tag)
    st.update(st4)
    return p_h, z, no_players, kickoffs, cost_in, cost_out, st


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--switch", type=float, default=30.0)
    ap.add_argument("--alive-target", type=float, default=0.58)
    ap.add_argument("--name", default="v4")
    args = ap.parse_args()

    p_h, z, nop, ko, ci, co, stats = evidences()
    print("indices :", stats)
    lo, hi = -10.0, 6.0
    for it in range(12):
        mid = (lo + hi) / 2
        lab = v3.decode(p_h, z, nop, ko, ci, co, mid, args.switch)
        frac = v3.alive_fraction(lab)
        lo, hi = (mid, hi) if frac > args.alive_target else (lo, mid)
    labels = v3.decode(p_h, z, nop, ko, ci, co, (lo + hi) / 2, args.switch)
    out = os.path.join(v1.RESULTS, args.name + ".csv")
    v1.write(out, labels)
    print("jeu vivant %.1f %% | segments D : %d | ecrit -> %s"
          % (100 * v3.alive_fraction(labels), v3.count_dead_segments(labels), os.path.relpath(out, ROOT)))
