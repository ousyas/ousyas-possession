import os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import idsse_bench as bench  # noqa: E402
import v5_train as T  # noqa: E402
from v5_em import em_prior  # noqa: E402

NORMAL = dict(BALL_MISSING_TARGET=0.41, NOISE_PLAYER_M=bench.NOISE_PLAYER_M,
              NOISE_BALL_M=bench.NOISE_BALL_M, PLAYER_DROP_START=bench.PLAYER_DROP_START)
STRESS = {
    "ballon 55%": dict(NORMAL, BALL_MISSING_TARGET=0.55),
    "bruit x3": dict(NORMAL, NOISE_PLAYER_M=3 * NORMAL["NOISE_PLAYER_M"], NOISE_BALL_M=3 * NORMAL["NOISE_BALL_M"]),
    "joueurs x4": dict(NORMAL, PLAYER_DROP_START=4 * NORMAL["PLAYER_DROP_START"]),
}

def setp(params):
    for k, v in params.items():
        setattr(bench, k, v)

def main():
    import v1, v2, v3, v4, v42, v5
    setp(NORMAL)
    train_d = [T.prepare(mid, i) for i, mid in enumerate(T.MATCHES)]
    print("donnees normales pretes")
    models, priors = [], []
    for k in range(len(T.MATCHES)):
        idx = [j for j in range(len(train_d)) if j != k]
        models.append(T.train(train_d, idx))
        d_tr = np.mean(np.concatenate([train_d[j]["y"][train_d[j]["graded"] & (train_d[j]["y"] >= 0)] == 2 for j in idx]))
        priors.append(np.array([(1 - d_tr) / 2, (1 - d_tr) / 2, d_tr]))
    for sname, params in [("normal", NORMAL)] + list(STRESS.items()):
        setp(params)
        print("\n=== test : %s ===" % sname)
        print("  %-8s %7s %7s %7s | %8s %8s" % ("match", "vrai", "EM", "biais", "fixe58", "em"))
        bias, f, e = [], [], []
        for k, mid in enumerate(T.MATCHES):
            d = T.prepare(mid, k, seed=100) if sname != "normal" else train_d[k]
            bench.patch([v1, v2, v3, v4, v42, v5], d["tracking"], d["blocks"])
            P = models[k].predict_proba(d["X"])
            g = d["graded"]
            a_em = 1 - em_prior(P[g], priors[k])[2]
            a_true = np.mean(d["gt"][g] != "D")
            s_f = T.score(d["gt"], np.where(g, v5.decode_probs(P, d["extras"], 0.58), "X"))[0]
            s_e = T.score(d["gt"], np.where(g, v5.decode_probs(P, d["extras"], a_em), "X"))[0]
            bias.append(a_em - a_true); f.append(s_f); e.append(s_e)
            print("  %-8s %6.1f%% %6.1f%% %+6.1f | %8.4f %8.4f" % (mid, 100*a_true, 100*a_em, 100*(a_em-a_true), s_f, s_e))
        print("  MOYENNE biais EM %+.1f pts | spellF1 fixe %.4f  em %.4f" % (100*np.mean(bias), np.mean(f), np.mean(e)))
    setp(NORMAL)

if __name__ == "__main__":
    main()
