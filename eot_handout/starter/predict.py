"""predict.py — run EOT inference on a new data folder.

Usage
-----
    python predict.py --data_dir ../eot_data/english --out predictions.csv
                      [--model eot_model.joblib]

Contract
--------
  • Reads: <data_dir>/labels.csv  (does NOT require the 'label' column)
           <data_dir>/audio/*.wav (same structure as training data)
  • Writes: <out>  with exactly three columns: turn_id, pause_index, p_eot
  • Errors loudly if the saved model's feature_names don't match the
    current feature_extractor.FEATURE_NAMES (catches silent column drift).
  • Uses the SAME extract_features() as train_model.py — imported from
    feature_extractor.py, never reimplemented here.
"""
import argparse
import csv
import os
import sys

import joblib
import numpy as np

from feature_extractor import (
    FEATURE_NAMES, FEATURE_DIM, build_feature_matrix
)


def main():
    ap = argparse.ArgumentParser(description="EOT inference on a new data folder")
    ap.add_argument("--data_dir", required=True,
                    help="Folder containing labels.csv and audio/")
    ap.add_argument("--out", default="predictions.csv",
                    help="Output CSV path")
    ap.add_argument("--model", default="eot_model.joblib",
                    help="Path to the joblib model bundle from train_model.py")
    args = ap.parse_args()

    # ── load model bundle ─────────────────────────────────────────────────── #
    if not os.path.exists(args.model):
        sys.exit(
            f"ERROR: model file '{args.model}' not found.\n"
            f"Run train_model.py first, or pass --model <path>."
        )
    bundle = joblib.load(args.model)
    model         = bundle["model"]
    saved_names   = bundle["feature_names"]
    saved_dim     = bundle["feature_dim"]

    # ── feature-name parity check ─────────────────────────────────────────── #
    if saved_names != FEATURE_NAMES or saved_dim != FEATURE_DIM:
        sys.exit(
            f"FEATURE MISMATCH — model was trained with:\n"
            f"  {saved_names}\n"
            f"but feature_extractor.py now defines:\n"
            f"  {FEATURE_NAMES}\n"
            f"Retrain the model before running predict.py."
        )

    # ── load data and extract features ───────────────────────────────────────#
    if not os.path.isdir(args.data_dir):
        sys.exit(f"ERROR: --data_dir '{args.data_dir}' does not exist.")

    print(f"Loading data from : {os.path.abspath(args.data_dir)}")
    print(f"Model             : {os.path.abspath(args.model)}  "
          f"[{bundle.get('model_kind','?')}, "
          f"trained on {bundle.get('trained_on',[])}]")

    # require_labels=False: predict.py must not crash if 'label' is absent
    X, keys, _, groups, meta = build_feature_matrix(
        args.data_dir, require_labels=False
    )

    print(f"Pauses            : {len(keys)}")
    print(f"Feature matrix finite: {np.all(np.isfinite(X))}")

    if len(X) == 0:
        sys.exit("ERROR: no pauses found in labels.csv — nothing to predict.")

    # ── predict ───────────────────────────────────────────────────────────── #
    p_eot = model.predict_proba(X)[:, 1]

    # ── write output ──────────────────────────────────────────────────────── #
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])  # exact column order
        for (tid, pi), p in zip(keys, p_eot):
            w.writerow([tid, pi, f"{p:.4f}"])

    print(f"Wrote {len(keys)} predictions → {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
