"""v4.1 -- v4 + ballon DISPUTE : la possession reste a l'equipe precedente (CHALLENGE.md).

    python ours/v41.py               # ecrit evaluation/results/v4_1.csv

Idee : sur un ballon dispute, la definition du challenge (CHALLENGE.md) garde l'equipe
precedente ; v4 changeait d'equipe trop vite. (Origine : erreurs reperees en regardant la video
pendant le diagnostic de la fenetre publique ; aucun seuil n'a ete regle dessus.)
Decision prise sur IDSSE uniquement : v4.1 baisse (spellF1 0.386 contre 0.409 pour v4) -> REJETEE.
Corrections :
  1. Zone de duel elargie a 2.5 m (bruit de l'ATD : un adversaire au contact peut apparaitre a 2-3 m).
  2. "Controle" valide seulement si AUCUN adversaire a < 2.5 m pendant >= 80 % de la fenetre.
  3. En melee (>= 2 alternances d'equipe dans les touches des 3 dernieres secondes) :
     3 touches consecutives requises ("more than two consecutive passes", CHALLENGE.md).
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from features import FPS, LENGTH, WIDTH  # noqa: E402
import v1  # noqa: E402
import v3  # noqa: E402
import v4  # noqa: E402

R_DUEL = 2.5
CTRL_FREE_FRAC = 0.8
SCRAMBLE_WIN = 3 * FPS


def team_state(n, touches, team, contested, d_h, d_a, ball_vis):
    switches = []
    cur, last_touch = -1, -10 ** 9
    pending, p_start, p_count = -1, 0, 0
    hist = []                                   # (frame, equipe) des touches recentes
    for t, T, c in zip(touches, team, contested):
        hist = [(f, e) for f, e in hist if t - f <= SCRAMBLE_WIN] + [(t, T)]
        alternations = sum(1 for a, b in zip(hist[:-1], hist[1:]) if a[1] != b[1])
        need = 3 if alternations >= 2 else 2
        if cur == -1 or t - last_touch > v4.RESTART_GAP:
            if cur != T:
                switches.append((t, T))
            cur, pending = T, -1
        elif T == cur:
            pending = -1
        elif not c:
            if pending == T:
                p_count += 1
            else:
                pending, p_start, p_count = T, t, 1
            dT, dO = (d_h, d_a) if T == 0 else (d_a, d_h)
            win_T = dT[t:t + v4.CTRL_FRAMES]
            win_O = dO[t:t + v4.CTRL_FRAMES]
            controlled = (len(win_T) == v4.CTRL_FRAMES and np.all(win_T <= v4.R_CTRL)
                          and np.mean(win_O > R_DUEL) >= CTRL_FREE_FRAC)
            if p_count >= need or controlled:
                switches.append((p_start, T))
                cur, pending = T, -1
        last_touch = t
    for T, dT, dO in ((0, d_h, d_a), (1, d_a, d_h)):
        c = np.concatenate(([0], (ball_vis & (dT < 1.5) & (dO > 3.0)).astype(np.int8), [0]))
        for s, e in zip(np.flatnonzero(np.diff(c) == 1), np.flatnonzero(np.diff(c) == -1)):
            if e - s >= v4.PROX_FRAMES:
                switches.append((s, T))
    switches.sort()
    state = np.full(n, -1, dtype=np.int8)
    cur, j = -1, 0
    for t in range(n):
        while j < len(switches) and switches[j][0] <= t:
            cur = switches[j][1]
            j += 1
        state[t] = cur
    return state


def team_probability(px, py, bx, by, teams, p_tag=None):
    old_duel, old_state = v4.R_DUEL, v4.team_state
    v4.R_DUEL, v4.team_state = R_DUEL, team_state     # meme pipeline v4, regles v4.1
    try:
        return v4.team_probability(px, py, bx, by, teams, p_tag)
    finally:
        v4.R_DUEL, v4.team_state = old_duel, old_state


def evidences():
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
    ap.add_argument("--name", default="v4_1")
    args = ap.parse_args()
    p_h, z, nop, ko, ci, co, stats = evidences()
    print("indices :", stats)
    lo, hi = -10.0, 6.0
    for it in range(12):
        mid = (lo + hi) / 2
        frac = v3.alive_fraction(v3.decode(p_h, z, nop, ko, ci, co, mid, args.switch))
        lo, hi = (mid, hi) if frac > args.alive_target else (lo, mid)
    labels = v3.decode(p_h, z, nop, ko, ci, co, (lo + hi) / 2, args.switch)
    out = os.path.join(v1.RESULTS, args.name + ".csv")
    v1.write(out, labels)
    print("jeu vivant %.1f %% | segments D : %d | ecrit -> %s"
          % (100 * v3.alive_fraction(labels), v3.count_dead_segments(labels), os.path.relpath(out, ROOT)))
