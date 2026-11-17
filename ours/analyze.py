"""Analyse d'erreurs d'une soumission -- AVANT de construire la version suivante.

    python ours/analyze.py                         # analyse evaluation/results/v1.csv
    python ours/analyze.py --sub evaluation/results/v1.csv --list 30

Partie A -- MATCH COMPLET, sans verite terrain :
    plausibilite de ce que la soumission affirme, comparee a la litterature
    (nombre et duree des arrets, temps de jeu effectif, possessions).
Partie B -- POURQUOI le modele dit D : attribution de chaque segment D a ses indices.
Partie C -- FENETRE PUBLIQUE (5 min), DIAGNOSTIC SEULEMENT :
    types d'erreurs + leur contexte + liste des segments fautifs a aller regarder en video.
    => sert a comprendre les erreurs, JAMAIS a regler des parametres (RULES.md).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)
from games import GAME, half_video_blocks  # noqa: E402
from features import FPS, load_tracking, rolling_mean, speed_signals, ball_out  # noqa: E402
import load_smart_tagging  # noqa: E402

GT_PUBLIC = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")

# Reperes de la litterature (match masculin de haut niveau) -- cf. sources dans la conversation
REF = {
    "arrets / match": "~108 (Siegle & Lames 2012)",
    "duree moy. arret": "~19-23 s (Siegle & Lames 2012 ; Augste & Cordes 2016)",
    "duree moy. sequence de jeu": "~32 s (Siegle & Lames 2012)",
    "temps de jeu effectif": "~54-56 min en LaLiga ; Barca ~60 %",
    "touches / corners / CF / 6 m": "~40 / ~10 / ~25-35 / ~17-25 par match",
}


def runs(labels):
    """[(label, start, end_exclusive)] des sequences de labels identiques."""
    out, start = [], 0
    for i in range(1, len(labels) + 1):
        if i == len(labels) or labels[i] != labels[start]:
            out.append((labels[start], start, i))
            start = i
    return out


def mmss(frame):
    s = frame / FPS
    return "%d:%04.1f" % (s // 60, s % 60)


def pct(x):
    return "%5.1f%%" % (100.0 * x)


def describe(arr):
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0:
        return "n=0"
    return "n=%d  mediane %.1f s  moy %.1f s  p10 %.1f  p90 %.1f" % (
        len(arr), np.median(arr), arr.mean(), np.percentile(arr, 10), np.percentile(arr, 90))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sub", default=os.path.join(ROOT, "evaluation", "results", "v1.csv"))
    ap.add_argument("--list", type=int, default=25, help="nb de segments fautifs a lister")
    args = ap.parse_args()

    sub = pd.read_csv(args.sub)["label"].astype(str).str.strip().str.upper().to_numpy()
    n = len(sub)
    teams, px, py, ps, bx, by, bs = load_tracking()
    events = load_smart_tagging.load()

    # --- indices par frame (memes definitions que v1) ---
    med, frac_slow, n_seen = speed_signals(ps)
    med_s = rolling_mean(med, FPS)
    slow = np.nan_to_num(med_s, nan=1.0) < 1.6
    out = np.nan_to_num(ball_out(bx, by)) > 0
    out_near = rolling_mean(out.astype(float), FPS) > 0.15
    missing = np.isnan(bx)
    in_sp = np.zeros(n, bool)
    sp_kind = np.full(n, "", dtype=object)
    for e in events:
        if (e["code"] or "").strip().lower() == "set pieces":
            in_sp[max(0, e["start_frame"]):min(n, e["end_frame"])] = True
            sp_kind[max(0, e["start_frame"]):min(n, e["end_frame"])] = e["tags"].get("Type") or "?"

    graded = np.zeros(n, bool)
    for _, a, b in half_video_blocks(GAME):
        graded[a:b + 1] = True
    match_min = graded.sum() / FPS / 60

    # ============================== PARTIE A ==============================
    print("=" * 78)
    print("A. MATCH COMPLET -- plausibilite (sans verite terrain)  |  %s" % os.path.basename(args.sub))
    print("=" * 78)
    d_runs, poss = [], []
    for _, a, b in half_video_blocks(GAME):
        for lab, s, e in runs(sub[a:b + 1]):
            if lab == "D":
                d_runs.append((s + a, e + a))
            elif lab in "HA":
                poss.append((lab, s + a, e + a))
    alive = np.isin(sub[graded], ["H", "A"]).sum() / FPS / 60
    d_dur = np.array([(e - s) / FPS for s, e in d_runs])
    p_dur = np.array([(e - s) / FPS for _, s, e in poss])
    print("duree notee          : %.1f min" % match_min)
    print("temps de jeu (H+A)   : %.1f min (%s)      ref: %s" % (
        alive, pct(alive / match_min), REF["temps de jeu effectif"]))
    print("segments D (arrets)  : %d                 ref: %s" % (len(d_runs), REF["arrets / match"]))
    print("  durees D           : %s" % describe(d_dur))
    print("  D < 2 s (suspects) : %d   |   D > 60 s : %d" % ((d_dur < 2).sum(), (d_dur > 60).sum()))
    print("possessions (H/A)    : %d   (H %d, A %d)" % (
        len(poss), sum(1 for p in poss if p[0] == "H"), sum(1 for p in poss if p[0] == "A")))
    print("  durees possessions : %s" % describe(p_dur))
    print("  possessions < 2 s  : %d  (souvent du clignotement)" % (p_dur < 2).sum())
    hh = sum(1 for (l1, _, e1), (l2, s2, _) in zip(poss[:-1], poss[1:]) if l1 != l2 and e1 == s2)
    print("changements H<->A directs (sans D entre) : %d" % hh)
    print("set pieces Smart Tagging : %d   (vs ~100+ arrets attendus -> le tagging en rate beaucoup)"
          % sum(1 for e in events if (e["code"] or "").strip().lower() == "set pieces"))

    # ============================== PARTIE B ==============================
    print("\n" + "=" * 78)
    print("B. POURQUOI le modele dit D -- indice present dans chaque segment D")
    print("=" * 78)
    cats = {"ballon sorti": 0, "set piece (sans sortie)": 0, "vitesse basse seule": 0,
            "ballon invisible seul": 0, "rien de net": 0}
    for s, e in d_runs:
        seg = slice(s, e)
        if out_near[seg].any():
            cats["ballon sorti"] += 1
        elif in_sp[seg].mean() > 0.5:
            cats["set piece (sans sortie)"] += 1
        elif slow[seg].mean() > 0.5:
            cats["vitesse basse seule"] += 1
        elif missing[seg].mean() > 0.5:
            cats["ballon invisible seul"] += 1
        else:
            cats["rien de net"] += 1
    for k, v in cats.items():
        print("  %-26s %4d  (%s)" % (k, v, pct(v / max(1, len(d_runs)))))
    print("  -> 'vitesse basse seule' / 'invisible seul' = D les moins fiables (jeu lent du Barca ?)")

    # ============================== PARTIE C ==============================
    print("\n" + "=" * 78)
    print("C. FENETRE PUBLIQUE -- DIAGNOSTIC (ne pas regler de parametres dessus)")
    print("=" * 78)
    gt = pd.read_csv(GT_PUBLIC)["label"].astype(str).str.strip().str.upper().to_numpy()
    m = gt != "X"
    print("frames notees publiques : %d" % m.sum())

    def context(mask):
        k = mask.sum()
        if k == 0:
            return "  (aucune)"
        return ("  n=%d | dans set piece %s | ballon invisible %s | vitesse basse %s | ballon sorti %s"
                % (k, pct(in_sp[mask].mean()), pct(missing[mask].mean()),
                   pct(slow[mask].mean()), pct(out_near[mask].mean())))

    gt_alive = m & np.isin(gt, ["H", "A"])
    false_d = gt_alive & (sub == "D")
    missed_d = m & (gt == "D") & np.isin(sub, ["H", "A"])
    wrong_team = gt_alive & np.isin(sub, ["H", "A"]) & (sub != gt)
    print("\n1) FAUX ARRETS (jeu vivant -> predit D) :")
    print(context(false_d))
    print("   reference, jeu vivant bien predit :")
    print(context(gt_alive & np.isin(sub, ["H", "A"])))
    print("\n2) ARRETS RATES (D -> predit H/A) :")
    print(context(missed_d))
    print("\n3) MAUVAISE EQUIPE (H<->A) :")
    print(context(wrong_team))

    # liste des segments fautifs, a regarder en video
    err = np.where(false_d, "faux D", np.where(missed_d, "D rate",
                   np.where(wrong_team, "mauvaise equipe", "")))
    segs = [(lab, s, e) for lab, s, e in runs(err) if lab and e - s >= FPS // 2]
    segs.sort(key=lambda r: -(r[2] - r[1]))
    print("\nSegments fautifs >= 0.5 s, du plus long au plus court (video = frame / 25) :")
    print("  %-16s %8s %8s %6s  %-8s %s" % ("type", "debut", "fin", "duree", "temps", "contexte"))
    for lab, s, e in segs[: args.list]:
        ctx = []
        if in_sp[s:e].mean() > 0.5:
            ctx.append("set piece:" + str(sp_kind[s]))
        if missing[s:e].mean() > 0.5:
            ctx.append("ballon invisible")
        if slow[s:e].mean() > 0.5:
            ctx.append("lent")
        if out_near[s:e].any():
            ctx.append("sortie")
        print("  %-16s %8d %8d %5.1fs  %-8s %s" % (lab, s, e, (e - s) / FPS, mmss(s), ", ".join(ctx)))

    print("\nReperes litterature :")
    for k, v in REF.items():
        print("  %-30s %s" % (k, v))


if __name__ == "__main__":
    main()
