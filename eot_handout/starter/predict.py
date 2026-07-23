"""predict.py — run EOT inference on a new, unseen data folder.

Imports extract_features() and the causal accumulator loop from train_model.py
(which in turn imports extract_features from train.py — unchanged).
Does NOT require a 'label' column in labels.csv.

Usage
-----
    python predict.py --data_dir ../eot_data/english --out predictions.csv
                      [--model eot_model.joblib]

Output
------
  predictions.csv with exactly three columns: turn_id, pause_index, p_eot
"""
import argparse
import os
import sys
import csv

import joblib
import numpy as np

# load_data() contains the causal accumulator loop — single definition in
# train_model.py, imported here so there is no copy-paste divergence risk.
from train_model import load_data, FEATURE_NAMES, FEATURE_DIM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True,
                    help="Folder containing labels.csv and audio/")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--model", default="eot_model.joblib",
                    help="joblib bundle written by train_model.py")
    args = ap.parse_args()

    # ── load model bundle ──────────────────────────────────────────────────── #
    if not os.path.exists(args.model):
        sys.exit(
            f"ERROR: model file '{args.model}' not found.\n"
            f"Run train_model.py first or pass --model <path>."
        )
    bundle = joblib.load(args.model)
    model       = bundle["model"]
    saved_names = bundle["feature_names"]
    saved_dim   = bundle["feature_dim"]

    # ── feature-order parity check ────────────────────────────────────────── #
    # FEATURE_NAMES is imported from train_model.py — same object that was
    # saved into the bundle at training time. If train.py's extract_features()
    # ever changes its output order, the bundle will mismatch and we catch it.
    if saved_names != FEATURE_NAMES or saved_dim != FEATURE_DIM:
        sys.exit(
            f"FEATURE MISMATCH\n"
            f"  model was saved with : {saved_names}\n"
            f"  train_model.py now has: {FEATURE_NAMES}\n"
            f"Retrain the model before running predict.py."
        )

    # ── locate data dir ───────────────────────────────────────────────────── #
    if not os.path.isdir(args.data_dir):
        sys.exit(f"ERROR: data directory not found: '{args.data_dir}'")

    print(f"data_dir : {os.path.abspath(args.data_dir)}")
    print(f"model    : {os.path.abspath(args.model)}"
          f"  [trained on {bundle.get('trained_on', [])}]")

    # ── extract features — same loop as train_model.py, no copy-paste ─────── #
    # require_label=False: predict.py works on data without ground-truth labels
    X, y, groups, keys = load_data(args.data_dir, require_label=False)
    print(f"pauses   : {len(keys)}  finite={np.all(np.isfinite(X))}")

    if len(X) == 0:
        sys.exit("ERROR: no pauses found — nothing to predict.")

    # ── predict ───────────────────────────────────────────────────────────── #
    p_eot = model.predict_proba(X)[:, 1]

    # ── write output — exactly three columns, no extras ───────────────────── #
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), p in zip(keys, p_eot):
            w.writerow([tid, pi, f"{p:.4f}"])

    print(f"wrote {len(keys)} predictions → {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
