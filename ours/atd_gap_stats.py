"""How does the challenge ATD lose the ball? Label-free statistics, to set the bench's gap world.

    python ours/atd_gap_stats.py                 # BAR-ATH tactical ATD (needs the data bundle)
    python ours/atd_gap_stats.py --idsse         # the same numbers on degraded IDSSE, per world

Reads only the tracking inputs of the graded halves -- no ground truth, public or private. The
numbers feed ours/idsse_bench.py (LONG_GAP_SHARE, LONG_GAP_MEDIAN_S, LONG_GAP_SIGMA); the link
between gaps and stoppages is NOT set from here: ours/gap_evidence.py estimates it on the test game.
The last column is a label-free hint of that link: the median collective speed of players while
the ball is visible vs inside long gaps (low speed = players waiting = likely a stoppage).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
from features import FPS, speed_signals  # noqa: E402
from v2 import gap_length  # noqa: E402


def stats(bx, ps, graded):
    miss = np.isnan(bx) & graded
    gl = gap_length(np.where(graded, bx, 0.0)) / FPS          # gaps cut at the graded edges
    med, _, _ = speed_signals(ps)
    out = {"visible": float(1 - miss.sum() / graded.sum())}
    lens = gl[miss]
    out["gap_median_s (frame-weighted)"] = float(np.median(lens)) if len(lens) else 0.0
    for t in (1, 3, 10, 30):
        out["missing in gaps > %d s" % t] = float(np.mean(lens > t)) if len(lens) else 0.0
    out["speed visible"] = float(np.nanmedian(med[graded & ~np.isnan(bx)]))
    long_ = miss & (gl > 10)
    out["speed in gaps > 10 s"] = float(np.nanmedian(med[long_])) if long_.any() else float("nan")
    return out


def show(name, s):
    print("%-12s " % name + "  ".join("%s %.2f" % (k, v) for k, v in s.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idsse", action="store_true")
    args = ap.parse_args()
    if not args.idsse:
        import v3
        teams, px, py, ps, bx, by, bs = v3.load_tracking()
        graded = np.zeros(len(bx), bool)
        for _, a, b in v3.half_video_blocks(v3.GAME):
            graded[a:b + 1] = True
        show("BAR-ATH", stats(bx, ps, graded))
        return
    import idsse_bench as bench
    import v5_train as T
    for w, cfg in (("legacy", dict(world="legacy")), ("blocks_a1", dict(world="blocks", affinity=1.0))):
        bench.set_world(**cfg)
        for i, mid in enumerate(T.MATCHES[:2]):
            m = bench.load_match(mid)
            graded = np.zeros(len(m["labels"]), bool)
            for _, a, b in bench.blocks_from_period(m["period"]):
                graded[a:b + 1] = True
            teams, px, py, ps, bx, by, bs = bench.degrade(m, i)
            show("%s %s" % (mid, w), stats(bx, ps, graded))
    bench.set_world("legacy")


if __name__ == "__main__":
    main()
