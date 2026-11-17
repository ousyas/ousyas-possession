"""Broadcast replays / close-ups as label-free evidence for stoppages (v16 candidate).

The broadcast homography is invalid on 34 % of its frames: replays, close-ups and crowd shots
(DATA.md), which a TV director shows mostly while play is stopped. This is information the tracking
does not contain. The broadcast is not frame-aligned to the tactical video, but it is the same live
match with a delay, so one offset per half maps it: found automatically by matching the ball's x
position in the two ATD files (1 s means), then refined frame by frame. No label is used anywhere.

    python ours/broadcast_evidence.py        # sync + label-free checks, prints the decision inputs

Categories per tactical frame: 0 = no broadcast frame, 1 = broadcast shows the pitch,
2 = replay / close-up. The dead-ball rate of each category is estimated on the test game by the same
EM as gap_evidence.py, and the log-odds against the global rate are added to D at half weight.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import require, source_frames  # noqa: E402

FPS = 25


def broadcast_ball():
    """Ball (x, y) per broadcast frame from the broadcast ATD (NaN when absent)."""
    raw = require(["atd_raw"], source="broadcast")[0]
    n = source_frames("broadcast")
    bx, by = np.full(n, np.nan), np.full(n, np.nan)
    with open(raw) as fh:
        for line in fh:
            parts = line.rstrip("\n").split(":")
            if len(parts) != 3:
                continue
            f = int(parts[0]) - 1
            g = parts[2]
            if 0 <= f < n and g and g[0] != "N":
                v = g.split(",")
                bx[f], by[f] = float(v[0]), float(v[1])
    return bx, by


def _bins(x):
    m = len(x) // FPS
    xb = x[:m * FPS].reshape(m, FPS)
    with np.errstate(all="ignore"):
        return np.nanmean(xb, 1)


def sync(t_bx, b_bx, blocks, b_valid):
    """{half: offset k} with broadcast frame = tactical frame - k, per graded half. Tries both
    orientations of the broadcast x axis and keeps the better one (returned in the tuple)."""
    best_all = None
    for flip in (False, True):
        bx_ = 1 - b_bx if flip else b_bx
        r = _sync(t_bx, bx_, blocks, b_valid)
        tot = sum(v[1] for v in r.values())
        if best_all is None or tot < best_all[0]:
            best_all = (tot, {h: v + (flip,) for h, v in r.items()})
    return best_all[1]


def _sync(t_bx, b_bx, blocks, b_valid):
    tb, bb = _bins(t_bx), _bins(np.where(b_valid, b_bx, np.nan))
    out = {}
    for half, a, b in blocks:
        ta, tz = a // FPS, b // FPS
        seg = tb[ta:tz]
        best = (np.inf, None)
        for L in range(-len(bb), len(tb)):           # bin lag: broadcast bin = tactical bin - L
            lo, hi = max(ta, L), min(tz, L + len(bb))
            if hi - lo < 300:
                continue
            d = np.abs(tb[lo:hi] - bb[lo - L:hi - L])
            d = d[~np.isnan(d)]
            if len(d) < 300:
                continue
            score = np.median(d)
            if score < best[0]:
                best = (score, L)
        L = best[1]
        # frame-level refinement within +/- 1 s
        fr = []
        for k in range(L * FPS - FPS, L * FPS + FPS + 1):
            lo, hi = max(a, k), min(b + 1, k + len(b_bx))
            d = np.abs(t_bx[lo:hi] - np.where(b_valid, b_bx, np.nan)[lo - k:hi - k])
            d = d[~np.isnan(d)]
            fr.append((np.median(d) if len(d) > 1000 else np.inf, k))
        score, k = min(fr)
        out[half] = (k, float(score), float(best[0]))
        del seg
    return out


def replay_category(n, blocks, offsets, b_valid):
    cat = np.zeros(n, np.int8)
    for half, a, b in blocks:
        k = offsets[half][0]
        t = np.arange(a, b + 1)
        tb = t - k
        ok = (tb >= 0) & (tb < len(b_valid))
        cat[t[ok]] = np.where(b_valid[tb[ok]], 1, 2)
    return cat


def category_log_ratio(P, cat, graded, prior, weight=0.5, min_frames=30 * FPS):
    """Same estimator as gap_evidence.gap_log_ratio, for any categorical signal."""
    from gap_evidence import em_dead_rate, _logit
    glob = em_dead_rate(P[graded, 2], prior[2])
    lr = np.zeros(len(P))
    info = {"global": glob}
    for c in np.unique(cat):
        m = graded & (cat == c)
        if m.sum() < min_frames:
            continue
        rate = em_dead_rate(P[m, 2], prior[2])
        lr[cat == c] = weight * (_logit(rate) - _logit(glob))
        info[int(c)] = (int(m.sum()), rate)
    return lr, info


def main():
    import v3
    from load_homography import load
    from features import speed_signals
    teams, px, py, ps, bx, by, bs = v3.load_tracking()
    blocks = v3.half_video_blocks(v3.GAME)
    _, _, b_valid = load(source="broadcast")
    b_bx, _ = broadcast_ball()
    print("broadcast: %d frames, homography valid %.1f %%, ball seen %.1f %%"
          % (len(b_valid), 100 * b_valid.mean(), 100 * np.mean(~np.isnan(b_bx))))
    offs = sync(bx, b_bx, blocks, b_valid)
    for h, (k, s, s1, flip) in offs.items():
        print("half %s: broadcast frame = tactical frame - %d  (median ball-x gap %.4f of pitch length, 1 s bins %.4f, x flipped %s)"
              % (h, k, s, s1, flip))
    cat = replay_category(len(bx), blocks, offs, b_valid)
    g = np.zeros(len(bx), bool)
    for _, a, b in blocks:
        g[a:b + 1] = True
    med, _, _ = speed_signals(ps)
    for c, name in ((0, "no broadcast frame"), (1, "broadcast on the pitch"), (2, "replay / close-up")):
        m = g & (cat == c)
        if m.any():
            print("%-24s %5.1f %% of graded frames | median player speed %.2f m/s | ball visible (tactical) %.0f %%"
                  % (name, 100 * m.mean() / g.mean(), np.nanmedian(med[m]), 100 * np.mean(~np.isnan(bx[m]))))
    np.save(os.path.join(os.environ.get("MTAH_DATA_DIR", os.path.join(ROOT, "data")), "replay_category.npy"), cat)


if __name__ == "__main__":
    main()
