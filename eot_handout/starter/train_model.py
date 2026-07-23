"""train_model.py — fit and save the EOT classifier.

Imports extract_features() and helpers directly from train.py (unchanged).

Usage
-----
    # Train on a single language:
    python train_model.py --data_dir ../eot_data/english --out eot_model.joblib

    # Train on both (run twice and compare, or extend --data_dir to accept multiple):
    python train_model.py --data_dir ../eot_data/english ../eot_data/hindi --out eot_model.joblib

Output
------
  <out>  joblib bundle: {model, feature_names, feature_dim, random_state}
"""
import argparse
import csv
import os
import sys
from collections import defaultdict

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

# ── import the validated feature extractor and helpers from train.py (unchanged) ──
from train import (
    extract_features,
    _voiced_seg_durations,
    HOP_S,
)
from features import load_wav, speech_before, f0_contour

# feature order must stay in sync with extract_features() in train.py
FEATURE_NAMES = [
    "f0_slope_rel",        #  0
    "f0_rise_fall",        #  1
    "f0_var_rel",          #  2
    "energy_slope",        #  3
    "energy_residual_rel", #  4
    "energy_var_tail",     #  5
    "filler_score",        #  6
    "lengthening_ratio",   #  7
    "voicing_density",     #  8
    "pause_index_norm",    #  9
    "turn_voicing_ratio",  # 10
    "energy_mean_db",      # 11
    "f0_range_rel",        # 12
    "final_voiced_dur",    # 13
    "voiced_frame_count",  # 14
    "pause_start_s",       # 15
]
FEATURE_DIM = 16
RANDOM_STATE = 42


# ── data loader — same causal accumulator as train.py's main() ───────────── #

def load_data(data_dir, require_label=True):
    """Load features and labels from one data_dir folder.

    Returns X (n, 16), y (n,) or None, groups (n,), keys (n,).
    """
    labels_path = os.path.join(data_dir, "labels.csv")
    if not os.path.exists(labels_path):
        sys.exit(f"ERROR: labels.csv not found in '{data_dir}'")

    rows = list(csv.DictReader(open(labels_path, newline="")))
    has_label = "label" in (rows[0] if rows else {})
    if require_label and not has_label:
        sys.exit(f"ERROR: 'label' column missing in {labels_path}")

    turn_rows = defaultdict(list)
    for r in rows:
        turn_rows[r["turn_id"]].append(r)
    for tid in turn_rows:
        turn_rows[tid].sort(key=lambda r: int(r["pause_index"]))

    cache = {}
    X, y_list, groups, keys = [], [], [], []

    for tid, t_rows in turn_rows.items():
        seg_durations   = []
        voiced_s_so_far = 0.0
        total_s_so_far  = 0.0

        for pause_idx, r in enumerate(t_rows):
            path = os.path.join(data_dir, r["audio_file"])
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
            groups.append(tid)
            keys.append((r["turn_id"], r["pause_index"]))
            if has_label:
                y_list.append(1 if r["label"] == "eot" else 0)

            # update context AFTER extracting — strictly causal
            seg = speech_before(x_wav, sr, pause_start, window_s=1.5)
            f0  = f0_contour(seg, sr)
            seg_durations.extend(_voiced_seg_durations(f0))
            voiced_s_so_far += float((f0 > 0).sum()) * HOP_S
            total_s_so_far   = pause_start   # mirror train.py's accumulator exactly

    X_mat = np.array(X, dtype=np.float32)
    y_arr = np.array(y_list, dtype=int) if y_list else None
    nan_ct = int(np.sum(~np.isfinite(X_mat)))
    if nan_ct:
        print(f"WARNING: {nan_ct} non-finite values in feature matrix for {data_dir}")
    return X_mat, y_arr, groups, keys


# ── AUC helper ───────────────────────────────────────────────────────────── #

def _auc(y_true, scores):
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n1, n0 = int(y_true.sum()), int((1 - y_true).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y_true == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ── scorer (mirrors score.py logic) ─────────────────────────────────────── #

def _mean_delay_at_budget(meta_rows, p_val, budget=0.05):
    """Best mean response delay achievable at <= budget interrupted-turn rate.
    meta_rows: list of {turn_id, pause_end, pause_start, label}
    """
    try:
        from score import evaluate, THRESHOLDS, DELAYS, TIMEOUT_S
    except ImportError:
        return None, {}

    pauses = [{"turn_id": m["turn_id"],
               "dur"    : m["pause_end"] - m["pause_start"],
               "label"  : m["label"],
               "p"      : float(p)} for m, p in zip(meta_rows, p_val)]

    best = None
    for t in THRESHOLDS:
        for d in DELAYS:
            cut, lat = evaluate(pauses, t, d)
            if cut <= budget and (best is None or lat < best["lat"]):
                best = {"lat": lat, "cut": cut, "t": t, "d": d}

    if best is None:
        best = {"lat": TIMEOUT_S, "cut": 0.0, "t": 1.0, "d": TIMEOUT_S}
    return best["lat"], best


# ── main ──────────────────────────────────────────────────────────────────── #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", nargs="+", required=True,
                    help="One or more data dirs (e.g. ../eot_data/english ../eot_data/hindi)")
    ap.add_argument("--out", default="eot_model.joblib")
    ap.add_argument("--val_frac", type=float, default=0.25)
    ap.add_argument("--random_state", type=int, default=RANDOM_STATE)
    args = ap.parse_args()

    rs = args.random_state
    print(f"feature_names : {FEATURE_NAMES}")
    print(f"feature_dim   : {FEATURE_DIM}")
    print(f"random_state  : {rs}\n")

    # ── check all data dirs exist before loading anything ────────────────── #
    for d in args.data_dir:
        if not os.path.isdir(d):
            sys.exit(f"ERROR: data directory not found: '{d}'")

    # ── load and combine all data dirs ───────────────────────────────────── #
    all_X, all_y, all_groups, all_meta = [], [], [], []
    lang_idx = {}   # lang_name → list of row indices in the combined set

    for data_dir in args.data_dir:
        lang = os.path.basename(data_dir.rstrip("/"))
        print(f"Loading {lang} ({data_dir}) ...", end=" ", flush=True)
        X, y, groups, keys = load_data(data_dir, require_label=True)

        # rebuild meta for scoring
        labels_path = os.path.join(data_dir, "labels.csv")
        meta_rows_raw = list(csv.DictReader(open(labels_path, newline="")))
        meta_by_key = {(r["turn_id"], r["pause_index"]): r for r in meta_rows_raw}

        start = len(all_X)
        for i, (key, feat, label) in enumerate(zip(keys, X, y)):
            raw = meta_by_key.get(key, {})
            all_meta.append({
                "turn_id"    : raw.get("turn_id", ""),
                "pause_start": float(raw.get("pause_start", 0)),
                "pause_end"  : float(raw.get("pause_end", 0)),
                "label"      : raw.get("label", ""),
            })
        all_X.extend(X)
        all_y.extend(y)
        all_groups.extend([f"{lang}__{g}" for g in groups])
        lang_idx[lang] = list(range(start, len(all_X)))
        print(f"{len(y)} pauses  (eot={y.sum()}, hold={len(y)-y.sum()})  finite={np.all(np.isfinite(X))}")

    X_all  = np.array(all_X, dtype=np.float32)
    y_all  = np.array(all_y, dtype=int)
    print(f"\nCombined: {len(y_all)} pauses  (eot={y_all.sum()}, hold={len(y_all)-y_all.sum()})\n")

    # ── GroupShuffleSplit keyed on turn_id (never splits a turn) ─────────── #
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_frac,
                                 random_state=rs)
    tr_idx, va_idx = next(splitter.split(X_all, y_all, all_groups))
    va_set = set(va_idx)

    X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
    X_va, y_va = X_all[va_idx], y_all[va_idx]
    meta_va = [all_meta[i] for i in va_idx]
    print(f"Train: {len(tr_idx)} | Val: {len(va_idx)}")
    for lang, idxs in lang_idx.items():
        n_val = sum(1 for i in idxs if i in va_set)
        print(f"  val {lang}: {n_val} pauses")

    # ── train calibrated logistic regression ─────────────────────────────── #
    # class_weight=None (uniform): diagnostic showed 40/60 imbalance is mild
    # enough that 'balanced' over-predicts EOT (~51% predicted vs ~38% actual),
    # inflating false positives and reducing held-out accuracy.
    print("\nFitting CalibratedClassifierCV(LogisticRegression, uniform weights) ...")
    base = LogisticRegression(class_weight=None, max_iter=2000,
                               random_state=rs)
    clf = CalibratedClassifierCV(base, method="isotonic", cv=5)
    clf.fit(X_tr, y_tr)

    p_va  = clf.predict_proba(X_va)[:, 1]
    acc   = float((clf.predict(X_va) == y_va).mean())
    auc   = _auc(y_va, p_va)
    chance = max(float(y_va.mean()), 1 - float(y_va.mean()))
    print(f"  val accuracy : {acc:.3f}  (chance {chance:.3f})")
    print(f"  val AUC      : {auc:.3f}")

    # overall delay score
    lat, best = _mean_delay_at_budget(meta_va, p_va)
    if lat is not None:
        print(f"  val mean delay @5% (overall) : {lat*1000:.0f} ms"
              f"  [t={best['t']}, d={best['d']*1000:.0f}ms,"
              f" interrupted={best['cut']*100:.1f}%]")

    # per-language delay score
    for lang, idxs in lang_idx.items():
        va_lang = [i for i in idxs if i in va_set]
        if not va_lang:
            continue
        local_pos = [list(va_idx).index(i) for i in va_lang]
        meta_l = [meta_va[p] for p in local_pos]
        p_l    = p_va[local_pos]
        y_l    = y_va[local_pos]
        lat_l, _ = _mean_delay_at_budget(meta_l, p_l)
        auc_l    = _auc(y_l, p_l)
        if lat_l is not None:
            print(f"  val mean delay @5% ({lang:7s}) : {lat_l*1000:.0f} ms  AUC={auc_l:.3f}")
        else:
            print(f"  val AUC ({lang}): {auc_l:.3f}")

    # ── refit on everything, then save ───────────────────────────────────── #
    print("\nRefitting on full combined dataset ...")
    clf.fit(X_all, y_all)

    bundle = {
        "model"        : clf,
        "feature_names": FEATURE_NAMES,
        "feature_dim"  : FEATURE_DIM,
        "random_state" : rs,
        "trained_on"   : list(lang_idx.keys()),
    }
    joblib.dump(bundle, args.out)
    print(f"Saved model bundle → {os.path.abspath(args.out)}")
    print("Paste the above val numbers into RUNLOG.md.")


if __name__ == "__main__":
    main()
