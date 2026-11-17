"""Banc d'essai IDSSE : mesurer nos versions sur 7 VRAIS matchs labellises (legal, CC-BY).

    pip install kloppy
    python ours/idsse_bench.py                    # 7 matchs, toutes les versions
    python ours/idsse_bench.py --matches J03WPY   # un seul match (test rapide)
    python ours/idsse_bench.py --versions v3_1    # une seule version

Ce que fait le script, pour chaque match IDSSE (Bundesliga 2022/23, tracking TRACAB 25 fps) :
  1. Telecharge le tracking via kloppy (1re fois : quelques minutes ; ensuite cache local).
  2. Extrait AUTOMATIQUEMENT les labels deja presents : ball_state mort -> D,
     sinon ball_owning_team domicile -> H / exterieur -> A.  (aucun label fait main)
  3. DEGRADE le tracking pour imiter l'ATD automatique du challenge : bruit de position,
     joueurs qui disparaissent par moments, ballon absent ~41 % du temps (plus souvent
     dans les zones chargees), pas d'identites.
  4. Lance nos versions (baseline50, v1, v2, v3_1) avec EXACTEMENT le meme code que pour
     le challenge (Smart Tagging absent ici -> on mesure le pipeline sans lui).
  5. Score avec le scorer OFFICIEL (evaluation/possession_metrics.py) : spellF1, macroF1...

Resultats : tableau a l'ecran + ajout dans ours/bench_results.csv (journal d'experiences).
Les donnees IDSSE vont dans data/ext/ (gitignore : rien n'est publie).
"""

import argparse
import csv
import datetime
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
sys.path.insert(0, os.path.join(ROOT, "starter"))
import possession_metrics as pm  # noqa: E402
from games import DATA_DIR  # noqa: E402

MATCHES = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPF", "J03WPY", "J03WQF"]
EXT = os.path.join(DATA_DIR, "ext")
LOG = os.path.join(HERE, "bench_results2.csv")
FPS = 25

# Degradation (cible : statistiques de l'ATD du challenge, cf. DATA.md)
BALL_MISSING_TARGET = 0.41      # ballon absent sur 41 % des frames notees
BALL_GAP_MEAN = 30              # longueur moyenne d'un trou de ballon (frames)
CROWD_BOOST = 3.0               # trous 3x plus probables quand >= 3 joueurs a < 3 m du ballon
PLAYER_DROP_START = 0.002       # debut d'un trou joueur (par frame)
PLAYER_GAP_MEAN = 50
NOISE_PLAYER_M, NOISE_BALL_M = 0.3, 0.15
LEN_M, WID_M = 105.0, 68.0

# Ballon pendant les arrets (banc v2). Dans IDSSE propre, le ballon est FIGE pendant 100 % des
# arrets (souvent juste derriere la ligne) ; dans un tracking automatique, on voit le ballon
# rouler, etre ramene par un joueur vers le point de reprise, puis pose. Sans cette simulation,
# un modele apprend "ballon qui bouge dans le terrain = jeu" (faux sur l'ATD du challenge).
DEAD_SIM = True
DEAD_KEEP_FROZEN = 0.15         # part des arrets laisses tels quels (ballon fige / hors terrain)
DEAD_MIN_FRAMES = FPS           # arrets plus courts : inchanges


# ----------------------------------------------------------------------------------------------
# 1-2. Chargement + labels
# ----------------------------------------------------------------------------------------------
def load_match(mid):
    os.makedirs(EXT, exist_ok=True)
    cache = os.path.join(EXT, "idsse_%s.npz" % mid)
    if os.path.exists(cache):
        d = np.load(cache, allow_pickle=True)
        return {k: d[k] for k in d.files}

    import glob
    from kloppy import sportec
    raw_dir = os.path.join(EXT, "raw")
    meta = glob.glob(os.path.join(raw_dir, "**", "*matchinformation*%s*.xml" % mid), recursive=True)
    pos = glob.glob(os.path.join(raw_dir, "**", "*positions_raw_observed*%s*.xml" % mid), recursive=True)
    if meta and pos:
        print("  lecture locale de %s (quelques minutes, une seule fois)..." % mid)
        ds = sportec.load_tracking(meta_data=meta[0], raw_data=pos[0], only_alive=False)
    else:
        print("  telechargement + lecture de %s (quelques minutes, une seule fois)..." % mid)
        try:
            ds = sportec.load_open_tracking_data(match_id=mid, only_alive=False)
        except Exception as e:
            sys.exit("\nTelechargement impossible (%s).\n"
                     "Telecharge les fichiers IDSSE a la main depuis figshare :\n"
                     "  https://doi.org/10.6084/m9.figshare.28196177\n"
                     "et mets-les (dezippes) dans : %s\n"
                     "Fichiers necessaires par match : *matchinformation*%s*.xml et "
                     "*positions_raw_observed*%s*.xml" % (type(e).__name__, raw_dir, mid, mid))
    home, away = ds.metadata.teams[0], ds.metadata.teams[1]
    df = ds.to_df(engine="pandas")

    cols, teams = [], []
    for code, team in ((0, home), (1, away)):
        for p in team.players:
            if "%s_x" % p.player_id in df.columns:
                cols.append(p.player_id)
                teams.append(code)
    px = df[["%s_x" % c for c in cols]].to_numpy(dtype=np.float32)
    py = df[["%s_y" % c for c in cols]].to_numpy(dtype=np.float32)
    bx = df["ball_x"].to_numpy(dtype=np.float32)
    by = df["ball_y"].to_numpy(dtype=np.float32)

    # coordonnees en 0-1 comme l'ATD (kloppy le fait par defaut ; sinon on normalise)
    if np.nanmax(np.abs(px)) > 2:
        pd_ = ds.metadata.pitch_dimensions
        x0, x1, y0, y1 = pd_.x_dim.min, pd_.x_dim.max, pd_.y_dim.min, pd_.y_dim.max
        px, bx = (px - x0) / (x1 - x0), (bx - x0) / (x1 - x0)
        py, by = (py - y0) / (y1 - y0), (by - y0) / (y1 - y0)

    # labels H/A/D depuis les flags du fournisseur
    state = df["ball_state"].astype(str).str.lower().to_numpy()
    owner = df["ball_owning_team_id"].astype(str).to_numpy()
    home_id = str(home.team_id)
    labels = np.where(owner == home_id, "H", "A").astype("<U1")
    labels[~np.char.startswith(state.astype(str), "alive")] = "D"
    period = df["period_id"].to_numpy()

    out = dict(px=px, py=py, bx=bx, by=by, teams=np.array(teams, np.int8), labels=labels,
               period=period)
    np.savez_compressed(cache, **out)
    return out


def synthetic_match(seed=0, n=20000):
    """Faux match pour tester le banc sans reseau (--synthetic)."""
    rng = np.random.default_rng(seed)
    labels = np.empty(n, "<U1")
    i, cur = 0, "H"
    while i < n:
        L = int(rng.integers(100, 600))
        labels[i:i + L] = cur
        i += L
        cur = rng.choice([c for c in "HAD" if c != cur])
    # joueurs : marche aleatoire lisse (~3 m/s), pas de teleportation
    px = np.clip(rng.random(22) + np.cumsum(rng.normal(0, 0.0012, (n, 22)), axis=0), 0, 1).astype(np.float32)
    py = np.clip(rng.random(22) + np.cumsum(rng.normal(0, 0.0018, (n, 22)), axis=0), 0, 1).astype(np.float32)
    teams = np.array([0] * 11 + [1] * 11, np.int8)
    bx = np.empty(n, np.float32)
    by = np.empty(n, np.float32)
    for t in range(n):
        side = 0 if labels[t] == "H" else 11
        bx[t], by[t] = px[t, side] + 0.003, py[t, side]
    return dict(px=px, py=py, bx=bx, by=by, teams=teams, labels=labels,
                period=np.where(np.arange(n) < n // 2, 1, 2))


# ----------------------------------------------------------------------------------------------
# 3. Degradation "a la ATD"
# ----------------------------------------------------------------------------------------------
def markov_gaps(n, p_start, mean_len, rng):
    """Masque True = observation perdue ; trous de longueur ~ geometrique(mean_len)."""
    p_start = np.broadcast_to(p_start, (n,))
    miss = np.zeros(n, bool)
    t = 0
    starts = rng.random(n) < p_start
    while t < n:
        if starts[t]:
            L = int(rng.geometric(1.0 / mean_len))
            miss[t:t + L] = True
            t += L
        else:
            t += 1
    return miss


def simulate_dead_ball(bx, by, labels, rng):
    """Remplace le ballon fige des arrets par : roule (0-2 s), ramene vers la reprise, pose (2-5 s).
    Positions en coordonnees 0-1 ; modifie bx, by en place."""
    D = labels == "D"
    d = np.diff(np.concatenate(([0], D.astype(np.int8), [0])))
    n = len(bx)
    for s, e in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)):
        L = e - s
        if L < DEAD_MIN_FRAMES or s == 0 or e >= n or rng.random() < DEAD_KEEP_FROZEN:
            continue
        x0, y0, x1, y1 = bx[s - 1], by[s - 1], bx[e], by[e]
        if np.isnan([x0, y0, x1, y1]).any():
            continue
        p0 = np.array([np.clip(x0, 0.01, 0.99) * LEN_M, np.clip(y0, 0.01, 0.99) * WID_M])
        p1 = np.array([np.clip(x1, 0.0, 1.0) * LEN_M, np.clip(y1, 0.0, 1.0) * WID_M])
        place = int(min(L // 2, rng.integers(2 * FPS, 5 * FPS + 1)))
        roll = int(rng.integers(0, min(2 * FPS, L // 4) + 1))
        carry = L - place - roll
        traj = np.empty((L, 2))
        # 1) le ballon roule encore un peu (vitesse ~2 m/s qui decroit)
        ang = rng.uniform(0, 2 * np.pi)
        sp = 2.0 * np.exp(-np.arange(roll) / max(1, roll / 2)) / FPS
        traj[:roll] = p0 + np.cumsum(sp)[:, None] * [np.cos(ang), np.sin(ang)] if roll else traj[:0]
        start = traj[roll - 1] if roll else p0
        # 2) ramene vers le point de reprise (mouvement doux + petites irregularites)
        if carry > 0:
            w = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, carry))
            wob = np.cumsum(rng.normal(0, 0.05, (carry, 2)), 0)
            wob -= np.linspace(0, 1, carry)[:, None] * wob[-1]
            traj[roll:roll + carry] = start + w[:, None] * (p1 - start) + wob
        # 3) pose au point de reprise
        traj[roll + carry:] = p1 + rng.normal(0, 0.03, (place, 2))
        traj[:, 0] = np.clip(traj[:, 0], -1.0, LEN_M + 1.0)
        traj[:, 1] = np.clip(traj[:, 1], -1.0, WID_M + 1.0)
        bx[s:e], by[s:e] = traj[:, 0] / LEN_M, traj[:, 1] / WID_M


def degrade(m, seed):
    rng = np.random.default_rng(seed)
    px, py, bx, by = (m[k].astype(np.float64).copy() for k in ("px", "py", "bx", "by"))
    if DEAD_SIM:
        simulate_dead_ball(bx, by, m["labels"], np.random.default_rng(seed + 7))
    n, k = px.shape
    vx0 = np.diff(px * LEN_M, axis=0, prepend=np.nan) * FPS
    vy0 = np.diff(py * WID_M, axis=0, prepend=np.nan) * FPS
    ker = np.ones(5) / 5
    ps_clean = np.apply_along_axis(lambda c: np.convolve(c, ker, "same"), 0, np.hypot(vx0, vy0))

    def smooth_noise(shape, win=2 * FPS):
        w = rng.normal(0, 1, shape)
        k = np.ones(win) / np.sqrt(win)
        if w.ndim == 1:
            return np.convolve(w, k, "same")
        return np.apply_along_axis(lambda c: np.convolve(c, k, "same"), 0, w)

    px += smooth_noise(px.shape) * NOISE_PLAYER_M / LEN_M
    py += smooth_noise(py.shape) * NOISE_PLAYER_M / WID_M
    bx += smooth_noise(n) * NOISE_BALL_M / LEN_M
    by += smooth_noise(n) * NOISE_BALL_M / WID_M

    for j in range(k):                                    # joueurs qui "sautent"
        gone = markov_gaps(n, PLAYER_DROP_START, PLAYER_GAP_MEAN, rng)
        px[gone, j] = np.nan
        py[gone, j] = np.nan

    # ballon : trous plus probables dans les zones chargees ; dichotomie pour viser 41 %
    d = np.hypot((px - bx[:, None]) * LEN_M, (py - by[:, None]) * WID_M)
    crowded = np.sum(d < 3.0, axis=1) >= 3
    weight = np.where(crowded, CROWD_BOOST, 1.0)
    lo, hi = 1e-4, 0.2
    for _ in range(12):
        mid = (lo + hi) / 2
        miss = markov_gaps(n, np.minimum(1, mid * weight), BALL_GAP_MEAN, np.random.default_rng(seed + 1))
        if miss.mean() < BALL_MISSING_TARGET:
            lo = mid
        else:
            hi = mid
    bx[miss] = np.nan
    by[miss] = np.nan

    # vitesses (m/s) recalculees depuis les positions bruitees, comme un tracker automatique
    ps = np.where(np.isnan(px), np.nan, ps_clean)
    bs = np.full(n, np.nan)
    return (m["teams"], px.astype(np.float32), py.astype(np.float32), ps.astype(np.float32),
            bx.astype(np.float32), by.astype(np.float32), bs.astype(np.float32))


def blocks_from_period(period):
    blocks = []
    for p in np.unique(period):
        idx = np.flatnonzero(period == p)
        if len(idx) > FPS * 60:
            blocks.append((int(p), int(idx[0]), int(idx[-1])))
    return blocks


# ----------------------------------------------------------------------------------------------
# 4. Nos versions, branchees sur ces donnees (meme code que pour le challenge)
# ----------------------------------------------------------------------------------------------
def patch(modules, tracking, blocks):
    import load_smart_tagging
    load_smart_tagging.load = lambda *a, **k: []        # pas de Smart Tagging dans IDSSE
    for mod in modules:
        if hasattr(mod, "load_tracking"):
            mod.load_tracking = lambda: tracking
        if hasattr(mod, "half_video_blocks"):
            mod.half_video_blocks = lambda *a, **k: blocks


def calibrate_and_decode(ev_fn, dec_fn, alive_target, extra=()):
    lo, hi = -10.0, 6.0
    for _ in range(12):
        mid = (lo + hi) / 2
        lab = dec_fn(mid)
        g = lab != "X"
        frac = np.isin(lab[g], ["H", "A"]).mean()
        lo, hi = (mid, hi) if frac > alive_target else (lo, mid)
    return dec_fn((lo + hi) / 2)


def run_version(name, tracking, blocks, n, alive_target):
    import v1, v2, v3, v4, v41, v42
    patch([v1, v2, v3, v4, v41, v42], tracking, blocks)
    if name == "baseline50":
        teams, px, py, ps, bx, by, bs = tracking
        d = np.hypot((px - bx[:, None]) * LEN_M, (py - by[:, None]) * WID_M)
        best = np.nanargmin(np.where(np.isnan(d), np.inf, d), axis=1)
        codes = np.where(np.isnan(bx), -1, teams[best]).astype(np.int8)
        h = np.concatenate(([0], np.cumsum(codes == 0)))
        a = np.concatenate(([0], np.cumsum(codes == 1)))
        i = np.arange(n)
        lo_, hi_ = np.maximum(0, i - 50), np.minimum(n, i + 51)
        hh, aa = h[hi_] - h[lo_], a[hi_] - a[lo_]
        lab = np.where(hh + aa == 0, "D", np.where(hh >= aa, "H", "A")).astype("<U1")
    elif name == "v1":
        lab = v1.build(30.0, 20.0)
    elif name == "v2":
        p_h, z, nop, ko = v2.evidences()
        lab = calibrate_and_decode(None, lambda b: v2.decode(p_h, z, nop, ko, b, 30.0, 20.0), alive_target)
    elif name == "v3_1":
        p_h, z, nop, ko, ci, co, _ = v3.evidences()
        lab = calibrate_and_decode(None, lambda b: v3.decode(p_h, z, nop, ko, ci, co, b, 30.0), alive_target)
    elif name == "v4_2":
        p_h, z, nop, ko, ci, co, _ = v42.evidences()
        lab = calibrate_and_decode(None, lambda b: v3.decode(p_h, z, nop, ko, ci, co, b, 30.0), alive_target)
    elif name == "v4_1":
        p_h, z, nop, ko, ci, co, _ = v41.evidences()
        lab = calibrate_and_decode(None, lambda b: v3.decode(p_h, z, nop, ko, ci, co, b, 30.0), alive_target)
    elif name == "v4":
        p_h, z, nop, ko, ci, co, _ = v4.evidences()
        lab = calibrate_and_decode(None, lambda b: v3.decode(p_h, z, nop, ko, ci, co, b, 30.0), alive_target)
    else:
        raise ValueError(name)
    out = np.full(n, "X", "<U1")
    for _, a, b in blocks:
        out[a:b + 1] = lab[a:b + 1]
    return out


# ----------------------------------------------------------------------------------------------
def team_acc_far(gt, pred, margin=FPS):
    alive = np.isin(gt, ["H", "A"]) & np.isin(pred, ["H", "A"])
    ch = np.flatnonzero(gt[1:] != gt[:-1]) + 1
    if len(ch) < 2:
        return float(np.mean(gt[alive] == pred[alive]))
    i = np.arange(len(gt))
    pos = np.clip(np.searchsorted(ch, i), 1, len(ch) - 1)
    dist = np.minimum(np.abs(i - ch[pos - 1]), np.abs(i - ch[pos]))
    m = alive & (dist > margin)
    return float(np.mean(gt[m] == pred[m])) if m.any() else float("nan")


def gt_stats(labels, blocks):
    g = np.zeros(len(labels), bool)
    for _, a, b in blocks:
        g[a:b + 1] = True
    lab = labels[g]
    alive = np.isin(lab, ["H", "A"]).mean()
    d = (lab == "D").astype(int)
    n_dead = int(np.sum(np.diff(np.concatenate(([0], d))) == 1))
    return alive, n_dead


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matches", nargs="+", default=MATCHES)
    ap.add_argument("--versions", nargs="+", default=["baseline50", "v4", "v4_2"])
    ap.add_argument("--alive-target", type=float, default=0.58,
                    help="meme regle que pour le challenge (litterature), PAS lue dans les labels")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic", action="store_true", help="faux match pour tester le banc")
    args = ap.parse_args()

    rows = []
    for mi, mid in enumerate(args.matches):
        t0 = time.time()
        m = synthetic_match(mi) if args.synthetic else load_match(mid)
        blocks = blocks_from_period(m["period"])
        n = len(m["labels"])
        gt = np.full(n, "X", "<U1")
        for _, a, b in blocks:
            gt[a:b + 1] = m["labels"][a:b + 1]
        tracking = degrade(m, args.seed + mi)
        alive, n_dead = gt_stats(m["labels"], blocks)
        gt_codes = pm.label_codes(gt)
        print("\n%s | %d frames | VERITE : jeu vivant %.1f %%, %d arrets | ballon visible apres degradation %.1f %%"
              % (mid, n, 100 * alive, n_dead, 100 * np.mean(~np.isnan(tracking[4]))))
        for v in args.versions:
            pred = run_version(v, tracking, blocks, n, args.alive_target)
            r = pm.evaluate(gt_codes, pm.label_codes(pred), bootstrap_n=50)
            rows.append(dict(match=mid, version=v, spellF1=r["spell_f1_mean"], macroF1=r["macro_f1"],
                             poss_gt=r["n_gt_spells"], poss_pred=r["n_pred_spells"],
                             dur_gt=r["gt_mean_spell_duration_s"], dur_pred=r["mean_spell_duration_s"],
                             alive_dead_f1=r["alive_dead_f1"], team_acc=r["team_acc_both_alive"], team_far=team_acc_far(gt, pred)))
            print("  %-10s spellF1 %.4f  macroF1 %.4f  poss %4d/%4d  dur %5.1f/%5.1f s  vivant/mort F1 %.3f  equipe %.3f"
                  % (v, r["spell_f1_mean"], r["macro_f1"], r["n_pred_spells"], r["n_gt_spells"],
                     r["mean_spell_duration_s"], r["gt_mean_spell_duration_s"],
                     r["alive_dead_f1"], r["team_acc_both_alive"]) + "  hors-trans %.3f" % rows[-1]["team_far"])
        print("  (%.0f s)" % (time.time() - t0))

    print("\n" + "=" * 78)
    print("MOYENNE SUR %d MATCHS  (ecart-type entre matchs)" % len(args.matches))
    print("=" * 78)
    for v in args.versions:
        s = np.array([r["spellF1"] for r in rows if r["version"] == v])
        mf = np.array([r["macroF1"] for r in rows if r["version"] == v])
        ad = np.array([r["alive_dead_f1"] for r in rows if r["version"] == v])
        ta = np.array([r["team_acc"] for r in rows if r["version"] == v])
        tf = np.array([r["team_far"] for r in rows if r["version"] == v])
        print("  %-10s spellF1 %.4f (+/- %.4f)  macroF1 %.4f  vivant/mort F1 %.3f  equipe %.3f"
              % (v, s.mean(), s.std(), mf.mean(), ad.mean(), ta.mean()) + "  hors-trans %.3f" % tf.mean())

    if not args.synthetic:
        new = not os.path.exists(LOG)
        with open(LOG, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["date"] + list(rows[0].keys()))
            if new:
                w.writeheader()
            stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
            for r in rows:
                w.writerow(dict(date=stamp, **r))
        print("\njournal -> %s" % os.path.relpath(LOG, ROOT))


if __name__ == "__main__":
    main()
