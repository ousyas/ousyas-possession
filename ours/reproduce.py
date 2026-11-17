"""Regenere la soumission de bout en bout (donnees -> modeles -> CSV -> verification).

    py ours/reproduce.py                  # entraine les 2 modeles sur IDSSE puis ecrit le CSV
    py ours/reproduce.py --skip-train     # reutilise ours/models/*.joblib
    py ours/reproduce.py --name equipe    # -> evaluation/results/equipe.csv

Steps (default --version v13r = the submission):
  1. ours/v11_train.py --version v13r   frame classifier P(H/A/D) + H<->A turnover detector +
                         temporal GRU head, 7 IDSSE matches
                         -> models/v13r.joblib, models/v13r_turnover.joblib, models/v13r_gru.joblib
  2. ours/v11.py --version v13r         applied to the challenge match -> evaluation/results/<name>.csv
  3. evaluation/validate_submission.py  same checks as the official scorer
Everything is seeded (random_state, bench degradation, torch seed, single-threaded deterministic
torch). --skip-train reuses the committed models and rebuilds the submitted file byte for byte on
any OS (the CSV is written with CRLF everywhere); retraining the GRU on another CPU can differ in
the last bits of its weights, which is why the weights are committed.
"""
import argparse
import hashlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def run(args):
    print("\n$ python " + " ".join(args), flush=True)
    subprocess.run([sys.executable, "-W", "ignore"] + args, cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--name", default="Ousyas")
    ap.add_argument("--gap-evidence", action="store_true",
                    help="label-free ball-gap evidence on D (ours/GAP_EVIDENCE.md); off = submitted file")
    ap.add_argument("--version", default="v13r", choices=["v11", "v12", "v13", "v13r"],
                    help="v13r = submission (v11 + GRU head); v11 = first submission; v12/v13 experiments")
    a = ap.parse_args()
    if not a.skip_train:
        run(["ours/v11_train.py", "--version", a.version])
    run(["ours/v11.py", "--name", a.name, "--version", a.version] + (["--gap-evidence"] if a.gap_evidence else []))
    out = os.path.join(ROOT, "evaluation", "results", a.name + ".csv")
    run(["evaluation/validate_submission.py", out])
    with open(out, "rb") as f:
        print("\nSHA-256 %s  %s" % (hashlib.sha256(f.read()).hexdigest(), os.path.relpath(out, ROOT)))


if __name__ == "__main__":
    main()
