"""v4.2 -- v4 + "duel gagne puis joue = controle".

    python ours/v42.py               # ecrit evaluation/results/v4_2.csv

Idee : quand une equipe gagne un duel puis joue le ballon, la possession lui revient des la
touche du duel ; v4 ignorait cette touche (disputee) et attendait 2 autres touches propres,
d'ou plusieurs secondes de retard. (Origine : un cas repere en regardant la video pendant le
diagnostic de la fenetre publique ; la regle est generique, aucun seuil lie a une frame.)
Decision prise sur IDSSE uniquement : spellF1 0.415 contre 0.409 pour v4 (7 matchs) -> retenue.
Regle ajoutee (une seule, le reste = v4) : une touche DISPUTEE de l'equipe T suivie, dans les
2 s, d'une touche PROPRE de la meme equipe T -> changement valide, date a la touche du duel.
(v4.1, qui durcissait les regles de duel, a ete rejetee : baisse sur IDSSE hors-transitions.)
"""

import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from features import FPS  # noqa: E402
import v1  # noqa: E402
import v3  # noqa: E402
import v4  # noqa: E402

DUEL_FOLLOW = 2 * FPS


def team_state(n, touches, team, contested, d_h, d_a, ball_vis):
    switches = []
    cur, last_touch = -1, -10 ** 9
    pending, p_start, p_count = -1, 0, 0
    duel_team, duel_t = -1, -10 ** 9              # derniere touche disputee d'une autre equipe
    for t, T, c in zip(touches, team, contested):
        if cur == -1 or t - last_touch > v4.RESTART_GAP:
            if cur != T:
                switches.append((t, T))
            cur, pending, duel_team = T, -1, -1
        elif T == cur:
            pending, duel_team = -1, -1
        elif c:
            duel_team, duel_t = T, t              # duel : on note, sans changer
        else:
            # duel gagne puis joue : touche propre de T juste apres son duel
            if duel_team == T and t - duel_t <= DUEL_FOLLOW:
                switches.append((duel_t, T))
                cur, pending, duel_team = T, -1, -1
                last_touch = t
                continue
            if pending == T:
                p_count += 1
            else:
                pending, p_start, p_count = T, t, 1
            dT = d_h if T == 0 else d_a
            win = dT[t:t + v4.CTRL_FRAMES]
            controlled = len(win) == v4.CTRL_FRAMES and np.all(win <= v4.R_CTRL)
            if p_count >= 2 or controlled:
                switches.append((p_start, T))
                cur, pending, duel_team = T, -1, -1
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
    old = v4.team_state
    v4.team_state = team_state
    try:
        return v4.team_probability(px, py, bx, by, teams, p_tag)
    finally:
        v4.team_state = old


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
    ap.add_argument("--name", default="v4_2")
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
