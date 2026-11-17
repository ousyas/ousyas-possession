"""Regenere la soumission de bout en bout (donnees -> modeles -> CSV -> verification).

    py ours/reproduce.py                  # entraine les 2 modeles sur IDSSE puis ecrit le CSV
    py ours/reproduce.py --skip-train     # reutilise ours/models/*.joblib
    py ours/reproduce.py --name equipe    # -> evaluation/results/equipe.csv

Etapes :
  1. ours/v11_train.py   modele principal P(H/A/D) + detecteur de changements H<->A, 7 matchs IDSSE
                         -> models/v11.joblib, models/v11_turnover.joblib
  2. ours/v11.py         application au match du challenge -> evaluation/results/<name>.csv
  3. evaluation/validate_submission.py  memes controles que le scorer officiel
Tout est deterministe (random_state fixes, degradation du banc a graine fixe) : le CSV produit
est identique octet pour octet a celui soumis (empreinte SHA-256 affichee a la fin).
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
    a = ap.parse_args()
    if not a.skip_train:
        run(["ours/v11_train.py"])
    run(["ours/v11.py", "--name", a.name])
    out = os.path.join(ROOT, "evaluation", "results", a.name + ".csv")
    run(["evaluation/validate_submission.py", out])
    with open(out, "rb") as f:
        print("\nSHA-256 %s  %s" % (hashlib.sha256(f.read()).hexdigest(), os.path.relpath(out, ROOT)))


if __name__ == "__main__":
    main()
