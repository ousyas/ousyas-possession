"""Metrica Sports sample games 1-2 as extra training / validation matches (same tracker family as
the challenge ATD). Public data: github.com/metrica-sports/sample-data ("please acknowledge the
source"); files in data/ext/metrica/ (git-ignored), fetched by:

    for g in 1 2; do for f in RawEventsData RawTrackingData_Home_Team RawTrackingData_Away_Team; do
      curl -sL -o data/ext/metrica/Sample_Game_${g}_$f.csv \
        https://raw.githubusercontent.com/metrica-sports/sample-data/master/data/Sample_Game_$g/Sample_Game_${g}_$f.csv
    done; done

Frame labels come from Metrica's own event data (no labelling by us), with the challenge's notion of
controlled possession:
  D  from BALL OUT / FAULT RECEIVED / a goal until the next SET PIECE (the restart), and before each
     period's kick-off;
  H/A the team of the last SET PIECE, PASS, RECOVERY or SHOT; CHALLENGE and BALL LOST change nothing
     until someone (re)gains the ball.
Importing this module lets idsse_bench.load_match("MET1") / ("MET2") return these matches in the
same format as the IDSSE ones, so every bench tool works on them unchanged.
"""
import os

import numpy as np
import pandas as pd

import idsse_bench as bench

MET_DIR = os.path.join(bench.EXT, "metrica")
MET_MATCHES = ["MET1", "MET2"]
_load_idsse = bench.load_match


def _tracking(path):
    raw = pd.read_csv(path, skiprows=2)
    cols = list(raw.columns)
    names = [c for c in cols[3:] if not c.startswith("Unnamed")]
    xy = {}
    for c in names:
        i = cols.index(c)
        xy[c] = (raw.iloc[:, i].to_numpy(np.float64), raw.iloc[:, i + 1].to_numpy(np.float64))
    return raw["Period"].to_numpy(), raw["Frame"].to_numpy(), xy


def labels_from_events(ev, n, period):
    lab = np.full(n, "D", "<U1")
    ev = ev.sort_values(["Start Frame", "End Frame"]).reset_index(drop=True)
    team = {"Home": "H", "Away": "A"}
    points = []                                       # (frame, value) change points
    for _, e in ev.iterrows():
        t, s = str(e["Type"]).upper(), str(e["Subtype"]).upper()
        f0 = int(e["Start Frame"]) - 1
        f1 = max(f0, int(e["End Frame"]) - 1)
        side = team.get(str(e["Team"]))
        if t == "SET PIECE" and side:
            points.append((f0, side))
        elif t in ("PASS", "RECOVERY") and side:
            points.append((f0, side))
        elif t == "SHOT" and side:
            points.append((f0, side))
            if "GOAL" in s and "OUT" not in s:
                points.append((f1, "D"))
        elif t == "BALL OUT":
            points.append((f1, "D"))
        elif t == "FAULT RECEIVED":
            points.append((f0, "D"))
    cur, last = "D", 0
    for f, v in sorted(points, key=lambda p: p[0]):   # D lasts until a team plays the ball again
        f = min(max(f, 0), n)
        lab[last:f] = cur
        cur, last = v, f
    lab[last:] = cur
    for p in np.unique(period):                       # before each period's kick-off: dead
        idx = np.flatnonzero(period == p)
        ko = [int(e["Start Frame"]) - 1 for _, e in ev.iterrows()
              if str(e["Type"]).upper() == "SET PIECE" and "KICK" in str(e["Subtype"]).upper()
              and int(e["Period"]) == p]
        if ko:
            lab[idx[0]:max(idx[0], ko[0])] = "D"
    return lab


def load_metrica(g):
    cache = os.path.join(bench.EXT, "metrica_MET%d.npz" % g)
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return {k: d[k] for k in d.files}
    stem = os.path.join(MET_DIR, "Sample_Game_%d_" % g)
    per_h, fr_h, xy_h = _tracking(stem + "RawTrackingData_Home_Team.csv")
    per_a, fr_a, xy_a = _tracking(stem + "RawTrackingData_Away_Team.csv")
    n = min(len(fr_h), len(fr_a))
    home = [c for c in xy_h if c != "Ball"]
    away = [c for c in xy_a if c != "Ball"]
    px = np.stack([xy_h[c][0][:n] for c in home] + [xy_a[c][0][:n] for c in away], 1).astype(np.float32)
    py = np.stack([xy_h[c][1][:n] for c in home] + [xy_a[c][1][:n] for c in away], 1).astype(np.float32)
    bx, by = xy_h["Ball"][0][:n].astype(np.float32), xy_h["Ball"][1][:n].astype(np.float32)
    teams = np.array([0] * len(home) + [1] * len(away), np.int8)
    period = per_h[:n].astype(np.int64)
    ev = pd.read_csv(stem + "RawEventsData.csv")
    labels = labels_from_events(ev, n, period)
    out = dict(px=px, py=py, bx=bx, by=by, teams=teams, labels=labels, period=period, source=np.array("metrica"))
    np.savez_compressed(cache, **out)
    return out


_degrade_idsse = bench.degrade


def degrade(m, seed):
    """Metrica's ball is already lost ~40 % of the time by the real tracker: keep the ball and the
    dead-ball behaviour as recorded (no simulation), keep positions, add only player dropouts."""
    if "source" not in m or str(m["source"]) != "metrica":
        return _degrade_idsse(m, seed)
    rng = np.random.default_rng(seed)
    px, py = m["px"].astype(np.float64).copy(), m["py"].astype(np.float64).copy()
    FPS = bench.FPS
    vx = np.diff(px * bench.LEN_M, axis=0, prepend=np.nan) * FPS
    vy = np.diff(py * bench.WID_M, axis=0, prepend=np.nan) * FPS
    ker = np.ones(5) / 5
    ps = np.apply_along_axis(lambda c: np.convolve(c, ker, "same"), 0, np.hypot(vx, vy))
    for j in range(px.shape[1]):
        gone = bench.markov_gaps(len(px), bench.PLAYER_DROP_START, bench.PLAYER_GAP_MEAN, rng)
        px[gone, j] = np.nan
        py[gone, j] = np.nan
    ps = np.where(np.isnan(px), np.nan, ps)
    n = len(px)
    return (m["teams"], px.astype(np.float32), py.astype(np.float32), ps.astype(np.float32),
            m["bx"].astype(np.float32), m["by"].astype(np.float32), np.full(n, np.nan, np.float32))


def load_match(mid):
    if str(mid).startswith("MET"):
        return load_metrica(int(str(mid)[3:]))
    return _load_idsse(mid)


bench.load_match = load_match
bench.degrade = degrade
