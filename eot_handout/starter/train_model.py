"""train_model.py — train EOT detector on English + Hindi combined.

Usage
-----
    python train_model.py [--data_root ../eot_data] [--out eot_model.joblib]
                          [--model {logreg,gbt,both}] [--val_frac 0.25]
                          [--random_state 42]

Output
------
  <out>          joblib bundle: {model, feature_names, random_state, meta}
  stdout         holdout accuracy, AUC, per-language mean delay @5% cutoff

Raises loudly if eot_data/english or eot_data/hindi are missing.
"""
import argparse
import os
import sys
import warnings

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

# shared feature pipeline — single source of truth
from feature_extractor import (
    FEATURE_NAMES, FEATURE_DIM, build_feature_matrix
)

# import scorer logic if available
try:
    from score import evaluate, THRESHOLDS, DELAYS, TIMEOUT_S
    _SCORER_OK = True
except ImportError:
    _SCORER_OK = False
    warnings.warn("score.py not importable; per-language delay scores not reported.")

RANDOM_STATE = 42


# ── scorer helper ─────────────────────────────────────────────────────────── #

def _mean_delay_at_budget(meta_val, p_val, budget=0.05):
    """Compute best mean response delay @ <=budget interrupted-turn rate.

    meta_val : list of pause dicts (turn_id, pause_end, pause_start, label)
    p_val    : array of predicted p_eot for those pauses
    """
    if not _SCORER_OK:
        return None, None

    pauses = []
    for m, p in zip(meta_val, p_val):
        pauses.append({
            "turn_id": m["turn_id"],
            "dur"    : m["pause_end"] - m["pause_start"],
            "label"  : m["label"],
            "p"      : float(p),
        })

    best = None
    for t in THRESHOLDS:
        for d in DELAYS:
            cut, lat = evaluate(pauses, t, d)
            if cut <= budget and (best is None or lat < best["lat"]):
                best = {"lat": lat, "cut": cut, "t": t, "d": d}

    if best is None:
        best = {"lat": TIMEOUT_S, "cut": 0.0, "t": 1.0, "d": TIMEOUT_S}
    return best["lat"], best


def _auc(y_true, scores):
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n1 = y_true.sum(); n0 = len(y_true) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((ranks[y_true == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


# ── build classifier ──────────────────────────────────────────────────────── #

def _make_calibrated(kind, random_state):
    if kind == "logreg":
        base = LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=random_state
        )
    else:  # gbt
        base = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, random_state=random_state,
            subsample=0.8, learning_rate=0.05,
        )
    return CalibratedClassifierCV(base, method="isotonic", cv=5)


# ── main ──────────────────────────────────────────────────────────────────── #

def main():
    ap = argparse.ArgumentParser(description="Train EOT model on English + Hindi")
    ap.add_argument("--data_root", default=os.path.join(os.path.dirname(__file__),
                                                         "..", "eot_data"),
                    help="Path that contains english/ and hindi/ subdirectories.")
    ap.add_argument("--out", default="eot_model.joblib",
                    help="Path to save the model bundle.")
    ap.add_argument("--model", choices=["logreg", "gbt", "both"], default="both",
                    help="Which model(s) to train; 'both' picks the one with higher val AUC.")
    ap.add_argument("--val_frac", type=float, default=0.25)
    ap.add_argument("--random_state", type=int, default=RANDOM_STATE)
    args = ap.parse_args()

    rs = args.random_state

    # ── locate language folders ───────────────────────────────────────────── #
    lang_dirs = {}
    for lang in ("english", "hindi"):
        d = os.path.join(args.data_root, lang)
        if not os.path.isdir(d):
            sys.exit(
                f"ERROR: expected '{d}' but directory not found.\n"
                f"Check --data_root (currently '{args.data_root}')."
            )
        lang_dirs[lang] = d
    print(f"Data root  : {os.path.abspath(args.data_root)}")
    print(f"Languages  : {list(lang_dirs.keys())}")
    print(f"Features   : {FEATURE_DIM}  ({FEATURE_NAMES})")
    print(f"Random state: {rs}")
    print()

    # ── load and combine ──────────────────────────────────────────────────── #
    all_X, all_y, all_groups, all_meta, all_lang = [], [], [], [], []
    lang_slices = {}   # lang -> (start_idx, end_idx)

    for lang, d in lang_dirs.items():
        print(f"Loading {lang} ...", end=" ", flush=True)
        X, keys, y, groups, meta = build_feature_matrix(d, require_labels=True)
        print(f"{len(y)} pauses  "
              f"(eot={y.sum()}, hold={len(y)-y.sum()})  "
              f"finite={np.all(np.isfinite(X))}")

        start = len(all_X)
        all_X.extend(X)
        all_y.extend(y)
        all_groups.extend([f"{lang}__{g}" for g in groups])
        all_meta.extend(meta)
        all_lang.extend([lang] * len(y))
        lang_slices[lang] = (start, len(all_X))

    X_all  = np.array(all_X, dtype=np.float32)
    y_all  = np.array(all_y, dtype=int)
    lang_arr = np.array(all_lang)
    print(f"\nCombined   : {len(y_all)} pauses  "
          f"(eot={y_all.sum()}, hold={len(y_all)-y_all.sum()})")
    print(f"Feature matrix finite: {np.all(np.isfinite(X_all))}")
    print()

    # ── GroupShuffleSplit on turn_id ──────────────────────────────────────── #
    splitter = GroupShuffleSplit(
        n_splits=1, test_size=args.val_frac, random_state=rs
    )
    tr_idx, va_idx = next(splitter.split(X_all, y_all, all_groups))

    X_tr, y_tr = X_all[tr_idx], y_all[tr_idx]
    X_va, y_va = X_all[va_idx], y_all[va_idx]
    meta_va   = [all_meta[i] for i in va_idx]
    lang_va   = lang_arr[va_idx]

    print(f"Train: {len(tr_idx)} pauses | Val: {len(va_idx)} pauses")
    for lang in lang_dirs:
        n = (lang_va == lang).sum()
        print(f"  val {lang}: {n} pauses")

    # ── train models ─────────────────────────────────────────────────────── #
    candidates = [args.model] if args.model != "both" else ["logreg", "gbt"]
    results = {}

    for kind in candidates:
        print(f"\n── Training {kind} ...")
        clf = _make_calibrated(kind, rs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(X_tr, y_tr)

        p_va = clf.predict_proba(X_va)[:, 1]
        acc  = float((clf.predict(X_va) == y_va).mean())
        auc  = _auc(y_va, p_va)
        print(f"   val  accuracy : {acc:.3f}  (chance {max(y_va.mean(), 1-y_va.mean()):.3f})")
        print(f"   val  AUC      : {auc:.3f}")

        # overall score.py metric
        lat_all, best_all = _mean_delay_at_budget(meta_va, p_va)
        if lat_all is not None:
            print(f"   val  mean delay @5% cutoff (overall) : {lat_all*1000:.0f} ms"
                  f"  [t={best_all['t']}, d={best_all['d']*1000:.0f}ms, "
                  f"interrupted={best_all['cut']*100:.1f}%]")

        # per-language score
        for lang in lang_dirs:
            mask = lang_va == lang
            if mask.sum() == 0:
                continue
            meta_l = [m for m, ml in zip(meta_va, lang_va) if ml == lang]
            p_l    = p_va[mask]
            y_l    = y_va[mask]
            lat_l, best_l = _mean_delay_at_budget(meta_l, p_l)
            auc_l  = _auc(y_l, p_l)
            if lat_l is not None:
                print(f"   val  mean delay @5% ({lang:7s}) : {lat_l*1000:.0f} ms"
                      f"  AUC={auc_l:.3f}  n={mask.sum()}")
            else:
                print(f"   val  AUC ({lang}): {auc_l:.3f}  n={mask.sum()}")

        results[kind] = {"clf": clf, "auc": auc, "lat": lat_all}

    # ── pick best model if both ───────────────────────────────────────────── #
    if len(results) == 2:
        best_kind = max(results, key=lambda k: results[k]["auc"])
        print(f"\n── Selecting '{best_kind}' (higher val AUC: "
              f"{results[best_kind]['auc']:.3f})")
    else:
        best_kind = candidates[0]

    final_clf = results[best_kind]["clf"]

    # Refit on ALL data for deployment
    print(f"\n── Refitting '{best_kind}' on full combined data ...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_clf.fit(X_all, y_all)

    # ── save model bundle ─────────────────────────────────────────────────── #
    bundle = {
        "model"        : final_clf,
        "feature_names": FEATURE_NAMES,
        "feature_dim"  : FEATURE_DIM,
        "random_state" : rs,
        "model_kind"   : best_kind,
        "trained_on"   : list(lang_dirs.keys()),
    }
    out_path = args.out
    joblib.dump(bundle, out_path)
    print(f"\n── Saved model bundle → {os.path.abspath(out_path)}")
    print(f"   feature_names : {FEATURE_NAMES}")
    print(f"   model kind    : {best_kind}")
    print(f"   trained on    : {bundle['trained_on']}")
    print("\nDone. Add the above val delay numbers to RUNLOG.md.")


if __name__ == "__main__":
    main()
