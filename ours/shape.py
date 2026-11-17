"""Features de FORME collective (piste 1) : l'equipe en possession s'etale, celle qui defend se
resserre et se place entre le ballon et son but. Aucune identite de joueur necessaire.

Pour chaque equipe et chaque frame, dans le sens d'attaque de l'equipe (detecte par mi-temps) :
  width     : largeur (percentile 90 - 10 des y, m)            -- "le 5 du 3-2-5 occupe la largeur"
  length    : longueur hors gardien (p90 - p10 des x, m)
  front     : joueurs a <= 8 m de la ligne du dernier defenseur adverse (ligne d'attaque)
  back      : joueurs de champ a <= 6 m de son propre dernier defenseur (ligne arriere, 3 vs 4)
  goalside  : joueurs plus pres de leur propre but que le ballon (NaN si ballon invisible)
+ differences H - A, et moyennes glissantes 2 s des differences.
"""
import numpy as np

FPS = 25
FRONT_M, BACK_M = 8.0, 6.0
MIN_PLAYERS = 5
TEAM_FEATS = ("width", "length", "front", "back", "goalside")


def _sorted(x):
    """Tri par ligne, NaN a la fin ; renvoie (tri, nombre de valeurs)."""
    return np.sort(x, axis=1), np.sum(~np.isnan(x), axis=1)


def _pct(s, n, q, offset=0):
    """Percentile q des valeurs triees s[:, offset:n] (approx. par indice)."""
    m = np.maximum(n - offset, 1)
    idx = offset + np.floor(q * (m - 1)).astype(int)
    return np.take_along_axis(s, np.minimum(idx, s.shape[1] - 1)[:, None], 1)[:, 0]


def attack_sign(px, teams, blocks, n):
    """+1 si H attaque vers x croissant sur cette frame, -1 sinon (constant par mi-temps)."""
    sign = np.ones(n)
    for _, a, b in blocks:
        with np.errstate(all="ignore"):
            mx = np.nanmean(px[a:b + 1][:, teams == 0])
        sign[a:b + 1] = 1.0 if mx < 0.5 else -1.0
    return sign


def _rolling(x, w):
    x = np.nan_to_num(x)
    c = np.concatenate(([0.0], np.cumsum(x)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - w), np.minimum(len(x), i + w + 1)
    return (c[hi] - c[lo]) / (hi - lo)


def shape_features(px, py, teams, bx, by, blocks, length=105.0, width=68.0):
    n = len(px)
    sign_h = attack_sign(px, teams, blocks, n)
    out = {}
    xatt = {}
    for code, name in ((0, "h"), (1, "a")):
        s = sign_h if code == 0 else -sign_h
        X = px[:, teams == code].astype(np.float64) * length
        xatt[name] = np.where(s[:, None] > 0, X, length - X)        # vers le but adverse = +
    bxa = {"h": np.where(sign_h > 0, bx * length, length - bx * length)}
    bxa["a"] = length - bxa["h"]

    for code, name, opp in ((0, "h", "a"), (1, "a", "h")):
        Y = py[:, teams == code].astype(np.float64) * width
        sy, ny = _sorted(Y)
        sx, nx = _sorted(xatt[name])                 # sx[:,0] = le plus recule (gardien)
        ok = nx >= MIN_PLAYERS
        f = {}
        f["width"] = _pct(sy, ny, 0.9) - _pct(sy, ny, 0.1)
        f["length"] = _pct(sx, nx, 0.9, 1) - _pct(sx, nx, 0.1, 1)
        # dernier defenseur adverse = 2e plus recule de l'adversaire (dans SON sens), ramene ici
        so, no = _sorted(xatt[opp])
        last_def_opp = length - so[:, 1]
        f["front"] = np.sum(xatt[name] >= (last_def_opp - FRONT_M)[:, None], axis=1).astype(float)
        own_last = sx[:, 1]                          # son propre dernier defenseur de champ
        f["back"] = (np.sum(xatt[name] <= (own_last + BACK_M)[:, None], axis=1) - 1).astype(float)
        f["goalside"] = np.where(np.isnan(bx), np.nan,
                                 np.sum(xatt[name] < bxa[name][:, None], axis=1)).astype(float)
        for k in TEAM_FEATS:
            v = f[k]
            v[~ok] = np.nan
            out["sh_%s_%s" % (k, name)] = v
    for k in TEAM_FEATS:
        d = out["sh_%s_h" % k] - out["sh_%s_a" % k]
        out["sh_%s_diff" % k] = d
        out["sh_%s_diff_m2" % k] = _rolling(d, 2 * FPS)
    names = sorted(out)
    F = np.stack([np.nan_to_num(out[k], nan=-1.0 if "diff" not in k else 0.0) for k in names], 1)
    return F.astype(np.float32), names


def swap_shape(F, names):
    """Echange H <-> A (augmentation par symetrie)."""
    Fs = F.copy()
    idx = {k: i for i, k in enumerate(names)}
    for i, k in enumerate(names):
        if "_diff" in k:
            Fs[:, i] = -F[:, i]
        elif k.endswith("_h"):
            Fs[:, i] = F[:, idx[k[:-2] + "_a"]]
        elif k.endswith("_a"):
            Fs[:, i] = F[:, idx[k[:-2] + "_h"]]
    return Fs
