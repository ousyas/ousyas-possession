"""Features de MOBILITE collective a long horizon (pour detecter les arrets de jeu).

Constat (5 min publiques, diagnostic) : pendant un arret ou le ballon reste visible et roule
doucement (on le ramene, on le replace), v5/v8 croit au jeu, alors que tous les joueurs marchent.
Des joueurs lents 2 s, ca arrive en jeu (construction lente) ; lents 8-10 s d'affilee, presque
jamais. Les features v5 ne regardent qu'a 2 s : on ajoute des horizons longs.

  mob_med5 / mob_med10   : vitesse mediane des joueurs, moyenne sur +/- 5 s et +/- 10 s
  mob_slow_spell         : duree (s) de la periode "lente" continue qui contient t (plafond 30 s)
  mob_since_fast         : secondes depuis la derniere frame "rapide" (plafond 30 s)
  mob_until_fast         : secondes jusqu'a la prochaine frame "rapide" (plafond 30 s)
  mob_top3_1 / _5        : vitesse moyenne des 3 joueurs les plus rapides, lissee 1 s / 5 s
  mob_walk5              : part des joueurs a < 1 m/s, moyenne sur 5 s
  mob_drop3 / mob_rise3  : chute de vitesse sur les 3 s passees / hausse sur les 3 s suivantes
Toutes symetriques en H/A (les deux equipes melangees) : rien a echanger en augmentation.
"""
import numpy as np

FPS = 25
MAX_HUMAN = 12.0
SLOW = 1.3          # m/s : vitesse mediane "lente" (profil IDSSE : ~1.0 en arret, ~2.1 en jeu)
CAP = 30 * FPS


def _rolling(x, w):
    x = np.asarray(x, dtype=np.float64)
    ok = ~np.isnan(x)
    c = np.concatenate(([0.0], np.cumsum(np.where(ok, x, 0.0))))
    k = np.concatenate(([0], np.cumsum(ok)))
    i = np.arange(len(x))
    lo, hi = np.maximum(0, i - w), np.minimum(len(x), i + w + 1)
    cnt = k[hi] - k[lo]
    return np.where(cnt > 0, (c[hi] - c[lo]) / np.maximum(cnt, 1), np.nan)


def _run_lengths(mask):
    """Pour chaque frame : frames depuis le dernier False (inclus t), et jusqu'au prochain False."""
    n = len(mask)
    since = np.zeros(n, np.int64)
    until = np.zeros(n, np.int64)
    run = 0
    for t in range(n):
        run = run + 1 if mask[t] else 0
        since[t] = min(run, CAP)
    run = 0
    for t in range(n - 1, -1, -1):
        run = run + 1 if mask[t] else 0
        until[t] = min(run, CAP)
    return since, until


def _shift(x, k):
    out = np.full(len(x), np.nan)
    if k > 0:
        out[k:] = x[:-k]
    else:
        out[:k] = x[-k:]
    return out


def mobility_features(ps):
    s = np.where(ps > MAX_HUMAN, np.nan, ps).astype(np.float64)
    with np.errstate(all="ignore"):
        med = np.nanmedian(s, axis=1)
        top3 = np.nanmean(-np.sort(-np.nan_to_num(s, nan=-1.0), axis=1)[:, :3], axis=1)
        walk = np.nanmean(np.where(np.isnan(s), np.nan, s < 1.0), axis=1)
    top3[top3 < 0] = np.nan
    med1 = _rolling(med, FPS // 2)
    slow = np.nan_to_num(med1, nan=0.0) < SLOW
    since, until = _run_lengths(slow)
    f = {
        "mob_med5": _rolling(med, 5 * FPS),
        "mob_med10": _rolling(med, 10 * FPS),
        "mob_slow_spell": np.minimum(since + until - slow, CAP) / FPS,
        "mob_since_fast": since / FPS,
        "mob_until_fast": until / FPS,
        "mob_top3_1": _rolling(top3, FPS // 2),
        "mob_top3_5": _rolling(top3, 5 * FPS),
        "mob_walk5": _rolling(walk, 5 * FPS),
        "mob_drop3": _shift(med1, 3 * FPS) - med1,
        "mob_rise3": _shift(med1, -3 * FPS) - med1,
    }
    names = sorted(f)
    M = np.stack([np.nan_to_num(f[k], nan=0.0) for k in names], 1)
    return M.astype(np.float32), names
