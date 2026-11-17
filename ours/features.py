"""Signaux d'arret de jeu, calcules a partir du tracking + Smart Tagging.

Usage (depuis la racine du repo, venv active) :
    python ours/features.py                      # parse le tracking (1 seule fois) + trace 10 min
    python ours/features.py --start 30000 --minutes 5

Ce que le script produit :
  - data/cache_tracking.npz  : le tracking parse en tableaux numpy (gitignore, ne pas publier)
  - ours/plots/signals_<start>.png : un graphe a 4 panneaux sur la fenetre choisie

Aucune verite terrain n'est utilisee ici : on regarde seulement les entrees.
"""

import argparse
import os
import sys
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, DATA_DIR, require, source_frames  # noqa: E402
from baseline import team_of_each_track  # noqa: E402
import load_smart_tagging  # noqa: E402

FPS = GAMES[GAME]["fps"]                      # 25
LENGTH, WIDTH = GAMES[GAME]["pitch"]          # metres
CACHE = os.path.join(DATA_DIR, "cache_tracking.npz")
PLOTS = os.path.join(HERE, "plots")
warnings.filterwarnings("ignore", category=RuntimeWarning)  # frames sans joueurs -> NaN, voulu

# Seuils "physiques", fixes a priori (PAS regles sur les 5 minutes publiques)
MAX_HUMAN_SPEED = 12.0     # m/s : au-dela, c'est un saut d'identite du tracker, pas un joueur
SLOW_SPEED = 2.0           # m/s : marche / trot leger
OUT_MARGIN = 0.0           # en fraction de terrain : ballon hors de [0,1] => sorti


# ----------------------------------------------------------------------------------------------
# 1. Parsing du tracking brut -> tableaux numpy (une seule fois, puis cache)
# ----------------------------------------------------------------------------------------------
def _xys(group):
    """'x,y,s' -> (x, y, s) en float, NaN si absent."""
    if not group or group[0] == "N":
        return np.nan, np.nan, np.nan
    vals = group.split(",")
    x, y = float(vals[0]), float(vals[1])
    s = float(vals[2]) if len(vals) > 2 and vals[2] and vals[2][0] != "N" else np.nan
    return x, y, s


def parse_tracking(raw_path, n_frames, n_slots):
    """Retourne px, py, ps (n_frames x n_slots) et bx, by, bs (n_frames), indexes en video frame."""
    px = np.full((n_frames, n_slots), np.nan, dtype=np.float32)
    py = np.full_like(px, np.nan)
    ps = np.full_like(px, np.nan)
    bx = np.full(n_frames, np.nan, dtype=np.float32)
    by = np.full_like(bx, np.nan)
    bs = np.full_like(bx, np.nan)

    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            f = int(parts[0]) - 1                         # frame_id compte depuis 1
            if not 0 <= f < n_frames:
                continue
            for slot, group in enumerate(parts[1].split(";")[:n_slots]):
                px[f, slot], py[f, slot], ps[f, slot] = _xys(group)
            bx[f], by[f], bs[f] = _xys(parts[2])
    return px, py, ps, bx, by, bs


def load_tracking():
    meta_path, raw_path = require(["atd_meta", "atd_raw"])
    teams = team_of_each_track(meta_path)              # 0 = H (Barca), 1 = A
    if os.path.exists(CACHE):
        d = np.load(CACHE)
        return teams, d["px"], d["py"], d["ps"], d["bx"], d["by"], d["bs"]
    print("parsing du tracking (1-2 min, une seule fois) ...")
    arrays = parse_tracking(raw_path, source_frames("tactical"), len(teams))
    np.savez_compressed(CACHE, **dict(zip(["px", "py", "ps", "bx", "by", "bs"], arrays)))
    print("  cache ecrit :", os.path.relpath(CACHE, ROOT))
    return (teams,) + arrays


# ----------------------------------------------------------------------------------------------
# 2. Signaux
# ----------------------------------------------------------------------------------------------
def rolling_mean(x, window):
    """Moyenne glissante centree, robuste aux NaN (ignores)."""
    x = np.asarray(x, dtype=np.float64)
    valid = ~np.isnan(x)
    csum = np.concatenate(([0.0], np.cumsum(np.where(valid, x, 0.0))))
    ccnt = np.concatenate(([0], np.cumsum(valid)))
    i = np.arange(len(x))
    lo = np.maximum(0, i - window)
    hi = np.minimum(len(x), i + window + 1)
    cnt = ccnt[hi] - ccnt[lo]
    out = np.full(len(x), np.nan)
    ok = cnt > 0
    out[ok] = (csum[hi] - csum[lo])[ok] / cnt[ok]
    return out


def speed_signals(ps):
    """Mediane des vitesses + part des joueurs lents, par frame."""
    s = np.where(ps > MAX_HUMAN_SPEED, np.nan, ps)      # vire les sauts d'identite
    n_seen = np.sum(~np.isnan(s), axis=1)
    with np.errstate(all="ignore"):
        median_speed = np.nanmedian(s, axis=1)
        frac_slow = np.nansum(s < SLOW_SPEED, axis=1) / n_seen
    frac_slow[n_seen == 0] = np.nan
    return median_speed, frac_slow, n_seen


def team_alignment(px, py, teams):
    """Synchronicite : pour chaque equipe, norme de la moyenne des vecteurs de direction unitaires.
    ~1 = tout le bloc va dans la meme direction, ~0 = directions desordonnees."""
    vx = np.diff(px * LENGTH, axis=0, prepend=np.nan)
    vy = np.diff(py * WIDTH, axis=0, prepend=np.nan)
    norm = np.hypot(vx, vy)
    moving = norm > (0.5 / FPS)                          # ignore les joueurs quasi immobiles
    ux = np.where(moving, vx / np.where(norm > 0, norm, 1), np.nan)
    uy = np.where(moving, vy / np.where(norm > 0, norm, 1), np.nan)
    out = {}
    for code, name in ((0, "H"), (1, "A")):
        cols = teams == code
        with np.errstate(all="ignore"):
            mx = np.nanmean(ux[:, cols], axis=1)
            my = np.nanmean(uy[:, cols], axis=1)
        out[name] = np.hypot(mx, my)
    return out


def ball_out(bx, by):
    """1 si le ballon est vu hors du terrain, 0 s'il est vu dedans, NaN s'il est invisible."""
    out = (bx < -OUT_MARGIN) | (bx > 1 + OUT_MARGIN) | (by < -OUT_MARGIN) | (by > 1 + OUT_MARGIN)
    return np.where(np.isnan(bx), np.nan, out.astype(float))


def set_piece_intervals():
    return [(e["start_frame"], e["end_frame"], e["tags"].get("Type") or "?")
            for e in load_smart_tagging.load()
            if (e["code"] or "").strip().lower() == "set pieces"]


# ----------------------------------------------------------------------------------------------
# 3. Graphe
# ----------------------------------------------------------------------------------------------
def plot(start, n, med, slow, align, out, ball_seen, set_pieces):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sl = slice(start, start + n)
    t = np.arange(start, start + n) / FPS / 60.0          # minutes de video
    fig, ax = plt.subplots(4, 1, figsize=(16, 10), sharex=True)

    ax[0].plot(t, med[sl], lw=0.6, color="0.7", label="mediane brute")
    ax[0].plot(t, rolling_mean(med, FPS // 2)[sl], lw=1.4, label="mediane lissee (1 s)")
    ax[0].set_ylabel("vitesse (m/s)")
    ax[0].legend(loc="upper right")

    ax[1].plot(t, rolling_mean(slow, FPS // 2)[sl], lw=1.2, color="tab:red")
    ax[1].set_ylabel("part joueurs\n< %.0f m/s" % SLOW_SPEED)
    ax[1].set_ylim(0, 1.05)

    for name, color in (("H", "tab:blue"), ("A", "tab:orange")):
        ax[2].plot(t, rolling_mean(align[name], FPS // 2)[sl], lw=1.2, color=color,
                   label="alignement " + name)
    ax[2].set_ylabel("synchronicite")
    ax[2].set_ylim(0, 1.05)
    ax[2].legend(loc="upper right")

    ax[3].fill_between(t, 0, ball_seen[sl], step="mid", color="0.85", label="ballon visible")
    ax[3].fill_between(t, 0, np.nan_to_num(out[sl]), step="mid", color="black",
                       label="ballon hors terrain")
    ax[3].set_ylabel("ballon")
    ax[3].set_yticks([])
    ax[3].legend(loc="upper right")
    ax[3].set_xlabel("temps video (minutes)")

    for a in ax:                                          # set pieces du Smart Tagging en vert
        for s0, s1, kind in set_pieces:
            if s1 >= start and s0 <= start + n:
                a.axvspan(s0 / FPS / 60.0, s1 / FPS / 60.0, color="tab:green", alpha=0.15)
    for s0, s1, kind in set_pieces:
        if start <= s0 <= start + n:
            ax[0].text(s0 / FPS / 60.0, ax[0].get_ylim()[1] * 0.92, kind, fontsize=7,
                       color="tab:green", rotation=90, va="top")

    fig.suptitle("Signaux d'arret de jeu -- frames %d a %d (vert = SET PIECES Smart Tagging)"
                 % (start, start + n))
    fig.tight_layout()
    os.makedirs(PLOTS, exist_ok=True)
    path = os.path.join(PLOTS, "signals_%d.png" % start)
    fig.savefig(path, dpi=130)
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=11185, help="premiere video frame (defaut: coup d'envoi)")
    p.add_argument("--minutes", type=float, default=10)
    args = p.parse_args()

    teams, px, py, ps, bx, by, bs = load_tracking()
    med, slow, n_seen = speed_signals(ps)
    align = team_alignment(px, py, teams)
    out = ball_out(bx, by)
    ball_seen = (~np.isnan(bx)).astype(float)
    sps = set_piece_intervals()

    print("frames: %d | slots: %d | joueurs vus par frame (mediane, en jeu): %.0f"
          % (len(med), px.shape[1], np.median(n_seen[11185:90652])))
    print("set pieces Smart Tagging: %d" % len(sps))
    path = plot(args.start, int(args.minutes * 60 * FPS), med, slow, align, out, ball_seen, sps)
    print("graphe ->", os.path.relpath(path, ROOT))
