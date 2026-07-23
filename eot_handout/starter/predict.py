"""predict.py — run EOT inference on a new, unseen data folder.

Imports extract_features() and helpers from train.py (unchanged).
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
import csv
import os
import sys
from collections import defaultdict

import joblib
import numpy as np

# ── import the validated feature extractor and helpers from train.py (unchanged) ──
from train import (
    extract_features,
    _voiced_seg_durations,
    HOP_S,
)
from features import load_wav, speech_before, f0_contour


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
    model         = bundle["model"]
    saved_names   = bundle["feature_names"]
    saved_dim     = bundle["feature_dim"]

    # ── feature-order parity check ────────────────────────────────────────── #
    # extract_features() is imported from train.py; saved_names from the bundle.
    # If train.py changed since the model was saved, we catch it here.
    from train_model import FEATURE_NAMES, FEATURE_DIM
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

    labels_path = os.path.join(args.data_dir, "labels.csv")
    if not os.path.exists(labels_path):
        sys.exit(f"ERROR: labels.csv not found in '{args.data_dir}'")

    rows = list(csv.DictReader(open(labels_path, newline="")))
    if not rows:
        sys.exit(f"ERROR: labels.csv in '{args.data_dir}' is empty")
    # 'label' column is NOT required — predict.py works on unseen data
    # that may not have ground-truth labels.

    print(f"data_dir : {os.path.abspath(args.data_dir)}")
    print(f"model    : {os.path.abspath(args.model)}"
          f"  [{bundle.get('model_kind', '?')}, trained on {bundle.get('trained_on', [])}]")

    # ── causal accumulator loop — identical structure to train.py's main() ── #
    turn_rows = defaultdict(list)
    for r in rows:
        turn_rows[r["turn_id"]].append(r)
    for tid in turn_rows:
        turn_rows[tid].sort(key=lambda r: int(r["pause_index"]))

    cache = {}
    X, keys = [], []

    for tid, t_rows in turn_rows.items():
        seg_durations   = []
        voiced_s_so_far = 0.0
        total_s_so_far  = 0.0

        for pause_idx, r in enumerate(t_rows):
            path = os.path.join(args.data_dir, r["audio_file"])
            if path not in cache:
                cache[path] = load_wav(path)
            x_wav, sr = cache[path]
            pause_start = float(r["pause_start"])

            ctx = {
                "pause_index"     : pause_idx,
                "seg_durations"   : list(seg_durations),
                "voiced_s_so_far" : voiced_s_so_far,
                "total_s_so_far"  : total_s_so_far,
            }
            feat = extract_features(x_wav, sr, pause_start, ctx)
            X.append(feat)
            keys.append((r["turn_id"], r["pause_index"]))

            # update context AFTER extracting — strictly causal
            seg = speech_before(x_wav, sr, pause_start, window_s=1.5)
            f0  = f0_contour(seg, sr)
            seg_durations.extend(_voiced_seg_durations(f0))
            voiced_s_so_far += float((f0 > 0).sum()) * HOP_S
            total_s_so_far   = pause_start   # match train.py exactly

    X_mat = np.array(X, dtype=np.float32)
    print(f"pauses   : {len(keys)}  finite={np.all(np.isfinite(X_mat))}")

    if len(X_mat) == 0:
        sys.exit("ERROR: no pauses found — nothing to predict.")

    # ── predict ───────────────────────────────────────────────────────────── #
    p_eot = model.predict_proba(X_mat)[:, 1]

    # ── write output — exactly three columns, no extras ───────────────────── #
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), p in zip(keys, p_eot):
            w.writerow([tid, pi, f"{p:.4f}"])

    print(f"wrote {len(keys)} predictions → {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
