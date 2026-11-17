"""IDSSE validation: does label-free ball-gap evidence (ours/gap_evidence.py) help when the test
tracking loses the ball the way the challenge ATD does?

    python ours/validate_gap.py                                   # train legacy, test all worlds
    python ours/validate_gap.py --train-world blocks_a1           # train with realistic gap shapes
    python ours/validate_gap.py --test-worlds legacy blocks_a10   # subset

Protocol, fixed BEFORE running:
  * exactly v11 (main model + turnover detector + time-varying Viterbi + EM alive rate, same
    hyper-parameters), trained on 6 IDSSE matches in the TRAIN world, scored on the 7th in each
    TEST world (leave-one-match-out, official scorer);
  * the worlds differ only in how the ball goes missing (ours/idsse_bench.py, BALL_GAP_WORLD).
    "atd" worlds copy the gap shape measured on BAR-ATH (atd_gap_stats.py, inputs only):
        legacy      the current bench: short gaps, more frequent in crowded live play
        atd_a1      BAR-ATH gap shape, independent of the label
        atd_a10     same, long blocks 10x more likely to start in a stoppage
        atd_stop    same, and 80 % of stoppages lose the ball until the restart: the closest
                    match to BAR-ATH's label-free speed ratio (players at 58 % speed in long gaps)
    41 % of frames miss the ball in every world, as in the challenge ATD;
  * variants: v11 | v11+gap (weight 0.5, fixed a priori) | v11+gap1 (weight 1, sensitivity only);
  * adoption rule: v11+gap must beat v11 on mean spellF1 in atd_a10 and atd_stop, and lose at
    most 0.002 in legacy and atd_a1, where there is nothing to find.
Features are cached per (world, match) in data/ext/gapcache/ (delete it after changing a world's
parameters). Log -> ours/bench_results_gap.csv.
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
import idsse_bench as bench  # noqa: E402
import v5_train as T  # noqa: E402
import shape  # noqa: E402
import events  # noqa: E402
import v8  # noqa: E402
import gap_evidence as G  # noqa: E402
from validate_v8 import train_turnover  # noqa: E402
from validate_v11 import train_main  # noqa: E402

WORLDS = {                      # "atd": gap shape measured on BAR-ATH (idsse_bench.LONG_GAP_*)
    "legacy": dict(world="legacy"),
    "atd_a1": dict(world="blocks", affinity=1.0),
    "atd_a10": dict(world="blocks", affinity=10.0),
    "atd_stop": dict(world="blocks", affinity=1.0, align=0.8),
}
CACHE = os.path.join(bench.EXT, "gapcache")
LOG = os.path.join(HERE, "bench_results_gap.csv")
COLS = ["spellF1", "macroF1", "alive_dead_f1", "team_acc", "team_far"]
VARIANTS = {"v11": None, "v11+gap": 0.5, "v11+gap1": 1.0}
TRAIN_KEYS = ("X", "E", "XT", "XTs", "turn", "y", "graded", "names")
TEST_KEYS = ("X", "E", "XT", "graded", "gt", "bx", "blocks", "no_players", "cost_in", "cost_out")


def prepare_world(mid, i, wname):
    """Same features as v11_train.py for match `mid` degraded in world `wname` (seed = i, as in
    v5_train.prepare). Cached on disk."""
    path = os.path.join(CACHE, "%s_%s.npz" % (wname, mid))
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        d = {k: z[k] for k in z.files}
        d["names"] = [str(s) for s in d["names"]]
        return d
    import v1, v2, v3, v4, v42, v5  # noqa: F401
    bench.set_world(**WORLDS[wname])
    try:
        m = bench.load_match(mid)
        blocks = bench.blocks_from_period(m["period"])
        n = len(m["labels"])
        tracking = bench.degrade(m, i)
        bench.patch([v1, v2, v3, v4, v42, v5], tracking, blocks)
        X, names, extras = v5.build_features()
    finally:
        bench.set_world("legacy")
    graded = np.zeros(n, bool)
    for _, a, b in blocks:
        graded[a:b + 1] = True
    y = np.full(n, -1)
    for k, c in enumerate("HAD"):
        y[m["labels"] == c] = k
    gt = np.full(n, "X", "<U1")
    gt[graded] = m["labels"][graded]
    teams, px, py, ps, bx, by, bs = tracking
    S, sn = shape.shape_features(px, py, teams, bx, by, blocks)
    E, _ = events.event_features(px, py, ps, bx, by)
    XT = np.hstack([v8.turnover_features(X, names, S, sn), E]).astype(np.float32)
    XTs = np.hstack([v8.turnover_features(v5.swap_features(X, names), names, shape.swap_shape(S, sn), sn),
                     E]).astype(np.float32)
    d = dict(X=X, E=E, XT=XT, XTs=XTs, turn=v8.turnover_labels(gt).astype(np.int8), y=y.astype(np.int8),
             graded=graded, gt=gt, bx=bx.astype(np.float32), blocks=np.array(blocks, np.int64),
             no_players=np.asarray(extras["no_players"]), cost_in=np.asarray(extras["cost_in"], np.float32),
             cost_out=np.asarray(extras["cost_out"], np.float32), names=np.array(names))
    assert not extras["kickoffs"], "IDSSE has no Smart Tagging, hence no kick-off anchors"
    os.makedirs(CACHE, exist_ok=True)
    np.savez(path, **d)
    d["names"] = list(names)
    return d


def decode(P, d, switch, alive):
    import v1, v2, v3, v4, v42, v5  # noqa: F401
    blocks = [tuple(int(v) for v in b) for b in d["blocks"]]
    bench.patch([v1, v2, v3, v4, v42, v5, v8], None, blocks)
    extras = dict(no_players=d["no_players"], kickoffs=[], cost_in=d["cost_in"], cost_out=d["cost_out"])
    return np.where(d["graded"], v8.decode_probs(P, extras, switch, alive), "X")


def world_stats(d):
    g, miss = d["graded"], np.isnan(d["bx"])
    dead = d["gt"] == "D"
    gl = G.gap_length(d["bx"])[g & miss] / G.FPS
    return dict(miss_dead=float(miss[g & dead].mean()), miss_alive=float(miss[g & ~dead].mean()),
                gap_median_s=float(np.median(gl)) if len(gl) else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-world", default="legacy", choices=list(WORLDS))
    ap.add_argument("--test-worlds", nargs="+", default=list(WORLDS), choices=list(WORLDS))
    ap.add_argument("--matches", nargs="+", default=T.MATCHES)
    args = ap.parse_args()
    mids = args.matches

    t0 = time.time()
    train = {}
    for i, mid in enumerate(T.MATCHES):
        if mid not in mids:
            continue
        d = prepare_world(mid, i, args.train_world)
        train[mid] = {k: d[k] for k in TRAIN_KEYS}
        print("features %-6s %-10s (%.0f s)" % (mid, args.train_world, time.time() - t0), flush=True)
    test = {}
    for w in args.test_worlds:
        for i, mid in enumerate(T.MATCHES):
            if mid not in mids:
                continue
            d = prepare_world(mid, i, w)
            test[w, mid] = {k: d[k] for k in TEST_KEYS}
            print("features %-6s %-10s (%.0f s)" % (mid, w, time.time() - t0), flush=True)
    for w in args.test_worlds:
        s = [world_stats(test[w, m]) for m in mids]
        print("world %-10s  P(ball missing | dead) %.2f  | alive) %.2f  median gap %.1f s"
              % (w, np.mean([x["miss_dead"] for x in s]), np.mean([x["miss_alive"] for x in s]),
                 np.mean([x["gap_median_s"] for x in s])))

    rows = []
    datas = [train[m] for m in mids]
    for k, mid in enumerate(mids):
        idx = [j for j in range(len(mids)) if j != k]
        t1 = time.time()
        main_m = train_main(datas, idx, True)
        turn_m = train_turnover(datas, idx)
        d_tr = np.mean(np.concatenate([datas[j]["y"][datas[j]["graded"] & (datas[j]["y"] >= 0)] == 2
                                       for j in idx]))
        prior = np.array([(1 - d_tr) / 2, (1 - d_tr) / 2, d_tr])
        for w in args.test_worlds:
            d = test[w, mid]
            g = d["graded"]
            P = main_m.predict_proba(np.hstack([d["X"], d["E"]]))
            sw = v8.switch_cost(turn_m.predict_proba(d["XT"])[:, 1])
            alive = v8.em_alive(P, g, prior)                     # global alive rate: unchanged
            dead_true = d["gt"] == "D"
            cat = G.gap_category(d["bx"])
            for name, weight in VARIANTS.items():
                if weight is None:
                    Pv, info = P, {}
                else:
                    lr, info = G.gap_log_ratio(P, d["bx"], g, prior, weight=weight)
                    Pv = G.apply_log_ratio(P, lr)
                pred = decode(Pv, d, sw, alive)
                s = T.score(d["gt"], pred)
                rec_d = float(np.mean(pred[dead_true] == "D"))
                row = dict(train_world=args.train_world, world=w, match=mid, version=name, alive_em=alive,
                           recall_D=rec_d, **dict(zip(COLS, s)))
                if name == "v11+gap":                            # EM estimate vs truth, per gap class
                    for c in range(G.N_CAT):
                        mc = g & (cat == c)
                        row["true_D_c%d" % c] = float(dead_true[mc].mean()) if mc.any() else float("nan")
                        row["em_D_c%d" % c] = info[c][1] if c in info else float("nan")
                rows.append(row)
                print("  %-6s %-10s %-9s spellF1 %.4f macroF1 %.4f viv/mort %.3f rappel D %.3f"
                      % (mid, w, name, s[0], s[1], s[2], rec_d), flush=True)
        print("  fold %s (%.0f s)" % (mid, time.time() - t1), flush=True)

    print("\nSUMMARY  train world = %s, %d matches, leave-one-match-out" % (args.train_world, len(mids)))
    for w in args.test_worlds:
        base = np.array([r["spellF1"] for r in rows if r["world"] == w and r["version"] == "v11"])
        for name in VARIANTS:
            rs = [r for r in rows if r["world"] == w and r["version"] == name]
            a = np.array([[r[c] for c in COLS + ["recall_D"]] for r in rs])
            extra = "" if name == "v11" else "  | vs v11 %+.4f, wins %d/%d" % (
                (a[:, 0] - base).mean(), int((a[:, 0] > base).sum()), len(base))
            print("  %-10s %-9s spellF1 %.4f (+/- %.4f) macroF1 %.4f viv/mort %.3f rappel D %.3f%s"
                  % (w, name, a[:, 0].mean(), a[:, 0].std(), a[:, 1].mean(), a[:, 2].mean(), a[:, 5].mean(), extra))
        rs = [r for r in rows if r["world"] == w and r["version"] == "v11+gap"]
        tr = np.nanmean([[r["true_D_c%d" % c] for c in range(G.N_CAT)] for r in rs], 0)
        em = np.nanmean([[r["em_D_c%d" % c] for c in range(G.N_CAT)] for r in rs], 0)
        print("  %-10s P(D | gap class vis,<=1,1-3,3-10,10-30,>30 s): true %s  EM %s"
              % (w, np.round(tr, 2), np.round(em, 2)))

    fields = sorted({k for r in rows for k in r}, key=lambda k: (k not in rows[0], k))
    new = not os.path.exists(LOG)
    with open(LOG, "a", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["date"] + fields)
        if new:
            wr.writeheader()
        stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        for r in rows:
            wr.writerow(dict(date=stamp, **r))
    print("\nlog -> %s  (%.0f min)" % (os.path.relpath(LOG, ROOT), (time.time() - t0) / 60))


if __name__ == "__main__":
    main()
