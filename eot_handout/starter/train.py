"""Prosodic features + classifier for end-of-turn detection.

    python train.py --data_dir ../eot_data/english --out predictions.csv
    python score.py --data_dir ../eot_data/english --pred predictions.csv

CAUSALITY CONTRACT: for a pause, features may ONLY use audio in
[0, pause_start). Neither pause_end nor pause duration is ever read here.

Feature set (16 features) — all relative/normalised; no hardcoded Hz or dB
thresholds so the same function generalises to Hindi and English alike.

  0  f0_slope_rel        relative F0 slope over last N voiced frames
  1  f0_rise_fall        non-monotonic shape flag: rise-then-fall in voiced window
  2  f0_var_rel          relative pitch variance (filler proxy component 1)
  3  energy_slope        dB/frame slope over last ~400 ms (negative = decay)
  4  energy_residual_rel dB of last frame relative to window mean (low = smooth EOT)
  5  energy_var_tail     energy variance in trailing ~300 ms (filler proxy component 2)
  6  filler_score        combined filler proxy: low e_var + low f0_var + long voiced run
  7  lengthening_ratio   final voiced-seg duration / running turn average
  8  voicing_density     voiced frames / total frames in window
  9  pause_index_norm    soft-bounded pause index for position context
  10 turn_voicing_ratio  voiced / total seconds so far in this turn
  11 energy_mean_db      mean dB in window (speaker-level energy context)
  12 f0_range_rel        relative F0 excursion in window
  13 final_voiced_dur    duration (s) of the last contiguous voiced run
  14 voiced_frame_count  raw count of voiced frames in window
  15 pause_start_s       absolute pause start time in seconds (causal)
"""
import argparse
import csv
import os
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit

from features import load_wav, speech_before, frame_energy_db, f0_contour


# --------------------------------------------------------------------------- #
#  Constants (kept in sync with features.py)                                   #
# --------------------------------------------------------------------------- #
HOP_MS   = 10           # features.py HOP_MS
HOP_S    = HOP_MS / 1000.0
FLOOR_DB = -80.0        # floor for fully-silent frames → kills -inf from log(~0)
N_FEATS  = 16


# --------------------------------------------------------------------------- #
#  Helpers                                                                      #
# --------------------------------------------------------------------------- #

def _safe_polyfit_slope(x_vals, y_vals):
    """Linear regression slope; returns 0.0 if inputs are degenerate."""
    if len(x_vals) < 3:
        return 0.0
    xi = np.array(x_vals, dtype=float)
    yi = np.array(y_vals, dtype=float)
    xi -= xi.mean()
    # Degenerate if all x-values are identical (e.g. single frame repeated)
    if np.std(xi) < 1e-9:
        return 0.0
    coeffs = np.polyfit(xi, yi, 1)
    return float(coeffs[0])


def _last_voiced_run(voiced_mask):
    """Return the length (in frames) of the last contiguous voiced run."""
    run = 0
    saved = 0
    for v in reversed(voiced_mask.tolist()):
        if v:
            run += 1
        else:
            if run > 0:
                saved = run
                break
    # If the whole window ended on voiced frames and we never hit a silence:
    return saved if saved > 0 else run


def _voiced_seg_durations(f0_arr):
    """Durations (s) of every contiguous voiced run in f0_arr."""
    durations, run = [], 0
    for v in f0_arr:
        if v > 0:
            run += 1
        else:
            if run > 0:
                durations.append(run * HOP_S)
                run = 0
    if run > 0:
        durations.append(run * HOP_S)
    return durations


# --------------------------------------------------------------------------- #
#  Feature extractor — STRICTLY CAUSAL                                         #
# --------------------------------------------------------------------------- #

def extract_features(x, sr, pause_start, turn_context=None):
    """Return a 16-D float32 vector from audio STRICTLY BEFORE pause_start.

    Parameters
    ----------
    x            : full-turn audio waveform (float32 ndarray)
    sr           : sample rate (int) — taken from soundfile, never hardcoded
    pause_start  : float, seconds; only [0, pause_start) is used
    turn_context : dict with running turn-level stats from *prior* pauses:
                     'pause_index'     – 0-based index of this pause
                     'seg_durations'   – voiced-seg durations (s) so far
                     'voiced_s_so_far' – total voiced seconds before this pause
                     'total_s_so_far'  – total seconds processed before this pause

    Returns
    -------
    np.ndarray, shape (N_FEATS,), dtype float32.
    GUARANTEED: no NaN, no inf.
    """
    # ── defaults ──────────────────────────────────────────────────────────── #
    if turn_context is None:
        turn_context = {}

    pause_index     = int(turn_context.get("pause_index", 0))
    seg_durations   = turn_context.get("seg_durations", [])       # prior segs
    voiced_s_so_far = float(turn_context.get("voiced_s_so_far", 0.0))
    total_s_so_far  = float(turn_context.get("total_s_so_far", 0.0))

    # ── EDGE CASE A: almost no audio before this pause ─────────────────────  #
    # < 100 ms of audio → we simply cannot say anything; return neutral zero.
    end_sample = int(pause_start * sr)
    if end_sample < sr // 10:           # 100 ms
        return np.zeros(N_FEATS, dtype=np.float32)

    # ── pull pre-pause window (≤ 1.5 s) ───────────────────────────────────  #
    seg = speech_before(x, sr, pause_start, window_s=1.5)

    # ── EDGE CASE B: energy -inf on fully silent frames → clip to FLOOR_DB ─ #
    e_db = np.clip(frame_energy_db(seg, sr), FLOOR_DB, None)

    # ── F0 contour (0.0 = unvoiced) ────────────────────────────────────────  #
    f0          = f0_contour(seg, sr)   # uses sr, not a hardcoded 16k
    voiced_mask = f0 > 0
    voiced_vals = f0[voiced_mask]
    voiced_idx  = np.where(voiced_mask)[0]

    n_voiced = len(voiced_vals)
    f0_mean  = float(voiced_vals.mean()) if n_voiced > 0 else 1.0  # avoid /0

    # ====================================================================== #
    # FEATURE 0 — Relative F0 slope over the last N voiced frames             #
    # Slope normalised by the window's own mean F0 → dimensionless, no Hz    #
    # threshold → language-agnostic.                                          #
    # Grounding: falling slope should be an EOT cue; BUT user noted that two  #
    # genuine hold pauses also had falling pitch, so this feature alone       #
    # should not dominate — it's one signal among many.                       #
    # EDGE CASE C: < 3 voiced frames → 0.0 (neutral, not NaN).               #
    # ====================================================================== #
    N_SLOPE = 8
    if n_voiced >= 3:
        uv = voiced_vals[-N_SLOPE:]
        ui = voiced_idx[-N_SLOPE:].astype(float)
        slope_abs   = _safe_polyfit_slope(ui, uv)
        f0_slope_rel = slope_abs / (f0_mean + 1e-6)
    else:
        f0_slope_rel = 0.0               # EDGE CASE: zero/few voiced frames

    # ====================================================================== #
    # FEATURE 1 — Non-monotonic F0 shape flag (rise-then-fall)               #
    # User observed one hold pause with a rise-then-fall shape from a self-  #
    # correction. We detect this by splitting the voiced window in two halves #
    # and checking whether the first half's mean > last frame's value         #
    # and the midpoint > start. Captures concave shape, not just endpoint.   #
    # EDGE CASE: < 4 voiced frames → 0.0.                                    #
    # ====================================================================== #
    if n_voiced >= 4:
        half = n_voiced // 2
        first_half_mean = float(voiced_vals[:half].mean())
        second_half_mean = float(voiced_vals[half:].mean())
        peak_in_middle  = voiced_vals[half // 2:3 * half // 2]
        mid_mean = float(peak_in_middle.mean()) if len(peak_in_middle) > 0 else first_half_mean
        # rise-then-fall: mid > start AND mid > end
        start_mean = float(voiced_vals[:max(1, n_voiced // 4)].mean())
        end_mean   = float(voiced_vals[-max(1, n_voiced // 4):].mean())
        f0_rise_fall = float(mid_mean > start_mean and mid_mean > end_mean)
    else:
        f0_rise_fall = 0.0

    # ====================================================================== #
    # FEATURE 2 — Relative pitch variance (filler-likelihood proxy, part 1)  #
    # Filler sounds ("ummm") are flat: low F0 variance over a long voiced run.#
    # Normalise by mean F0² so the scale is language-agnostic.               #
    # EDGE CASE: < 2 voiced frames → 0.0.                                    #
    # ====================================================================== #
    FILLER_WINDOW_S = 0.35
    n_filler = max(1, int(FILLER_WINDOW_S / HOP_S))
    filler_f0 = voiced_vals[-n_filler:] if n_voiced >= 2 else np.array([])

    if len(filler_f0) >= 2:
        f0_var     = float(np.var(filler_f0))
        f0_var_rel = f0_var / (f0_mean ** 2 + 1e-6)
    else:
        f0_var_rel = 0.0

    # ====================================================================== #
    # FEATURE 3 — Energy decay slope into the pause (last ~400 ms, dB/frame) #
    # Smooth decay to near-silence → negative slope → EOT.                   #
    # Hold pauses from mid-utterance cutoffs often have moderate residual     #
    # energy → slope is less steep or even flat/positive.                    #
    # EDGE CASE D: < 3 frames → 0.0 fallback.                                #
    # ====================================================================== #
    n_decay  = max(1, int(0.40 / HOP_S))    # ~40 frames
    e_decay  = e_db[-n_decay:]
    xi_decay = np.arange(len(e_decay), dtype=float)
    energy_slope = _safe_polyfit_slope(xi_decay, e_decay)

    # ====================================================================== #
    # FEATURE 4 — Residual energy at pause_start, relative to window mean    #
    # A low final dB level (well below window mean) signals smooth decay →   #
    # EOT. Moderate final energy → mid-utterance hold.                       #
    # ====================================================================== #
    energy_mean_db    = float(e_db.mean()) if len(e_db) > 0 else FLOOR_DB
    energy_final_db   = float(e_db[-1])    if len(e_db) > 0 else FLOOR_DB
    energy_residual_rel = energy_final_db - energy_mean_db  # negative = low residual

    # ====================================================================== #
    # FEATURE 5 — Energy variance in trailing ~300 ms (filler proxy, part 2) #
    # Filler ("ummm") is sustained, hence LOW energy variance.               #
    # Breaths, clicks, or consonant bursts near silence → higher variance.    #
    # EDGE CASE: < 2 frames → 0.0.                                           #
    # ====================================================================== #
    n_tail   = max(1, int(0.30 / HOP_S))
    e_tail   = e_db[-n_tail:]
    energy_var_tail = float(np.var(e_tail)) if len(e_tail) >= 2 else 0.0

    # ====================================================================== #
    # FEATURE 6 — Combined filler score                                       #
    # Fillers = LOW f0_var + LOW e_var + voiced run >= ~200 ms.              #
    # This featurises the interaction that none of the individual features   #
    # captures alone. High filler_score → likely hold.                       #
    # ====================================================================== #
    last_run_frames = _last_voiced_run(voiced_mask)
    last_run_dur    = last_run_frames * HOP_S

    FILLER_DUR_THRESH_S = 0.20    # relative, not about Hz
    # Low var in both pitch and energy AND sustained pitch = filler candidate
    is_flat_pitch  = float(f0_var_rel < 0.005)   # relative threshold
    is_flat_energy = float(energy_var_tail < 10.0)  # dB² — absolute but generous
    is_long_voiced = float(last_run_dur >= FILLER_DUR_THRESH_S)
    filler_score   = is_flat_pitch * is_flat_energy * is_long_voiced

    # ====================================================================== #
    # FEATURE 7 — Final voiced-segment lengthening ratio                      #
    # Duration of last contiguous voiced run / causal running mean duration.  #
    # Pre-boundary lengthening is a cross-linguistically robust EOT cue.     #
    # EDGE CASE E: pause_index == 0 / no prior segments → ratio = 1.0       #
    # (neutral: we have no reference, so assume the current seg is average). #
    # ====================================================================== #
    if len(seg_durations) > 0:
        avg_seg_dur = float(np.mean(seg_durations))
    else:
        # First pause — no prior data. Use the current seg if available,
        # so ratio = 1.0 (by definition, last = average).
        avg_seg_dur = last_run_dur if last_run_dur > 1e-6 else HOP_S * 10

    lengthening_ratio = last_run_dur / (avg_seg_dur + 1e-6)

    # ====================================================================== #
    # FEATURE 8 — Voicing density / speaking rate in window                   #
    # voiced_frames / total_frames: 0–1 ratio, language-agnostic.            #
    # ====================================================================== #
    n_total = len(f0)
    voicing_density = (float(voiced_mask.sum()) / n_total
                       if n_total > 0 else 0.0)

    # ====================================================================== #
    # FEATURE 9 — Pause index (soft-normalised)                               #
    # Later pauses in a turn are more likely to be EOT.                      #
    # ====================================================================== #
    pause_index_norm = float(pause_index) / (float(pause_index) + 5.0)

    # ====================================================================== #
    # FEATURE 10 — Turn voicing ratio (cumulative causal)                     #
    # Voiced fraction of all audio *before* this pause in the turn.          #
    # ====================================================================== #
    if total_s_so_far > 0:
        turn_voicing_ratio = voiced_s_so_far / total_s_so_far
    else:
        turn_voicing_ratio = voicing_density   # fallback: use window estimate

    # ====================================================================== #
    # FEATURES 11–15 — Supporting statistics                                  #
    # ====================================================================== #
    # 12: relative F0 excursion (pitch movement range, normalised)
    if n_voiced >= 2:
        f0_range_rel = float(voiced_vals.max() - voiced_vals.min()) / (f0_mean + 1e-6)
    else:
        f0_range_rel = 0.0

    # 13: final voiced-run duration in seconds (raw, not ratio)
    final_voiced_dur = last_run_dur

    # 14: raw voiced frame count (gives the classifier a sense of data density)
    voiced_frame_count = float(n_voiced)

    # 15: pause_start — absolutely causal; useful as a position signal
    pause_start_s = float(pause_start)

    # ── assemble ──────────────────────────────────────────────────────────── #
    feat = np.array([
        f0_slope_rel,        # 0
        f0_rise_fall,        # 1
        f0_var_rel,          # 2
        energy_slope,        # 3
        energy_residual_rel, # 4
        energy_var_tail,     # 5
        filler_score,        # 6
        lengthening_ratio,   # 7
        voicing_density,     # 8
        pause_index_norm,    # 9
        turn_voicing_ratio,  # 10
        energy_mean_db,      # 11
        f0_range_rel,        # 12
        final_voiced_dur,    # 13
        voiced_frame_count,  # 14
        pause_start_s,       # 15
    ], dtype=np.float32)

    # ── EDGE CASE F: assert no NaN/inf (safety net after all guards above) ─ #
    bad = ~np.isfinite(feat)
    if bad.any():
        feat = np.where(bad, 0.0, feat).astype(np.float32)

    assert np.all(np.isfinite(feat)), "BUG: non-finite value in feature vector"
    return feat


# --------------------------------------------------------------------------- #
#  Main — builds turn-context causal accumulator, runs train/eval split       #
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(args.data_dir, "labels.csv"))))
    cache = {}

    # Group by turn, sort by pause_index — the only way accumulation stays causal
    turn_rows = defaultdict(list)
    for r in rows:
        turn_rows[r["turn_id"]].append(r)
    for tid in turn_rows:
        turn_rows[tid].sort(key=lambda r: int(r["pause_index"]))

    X, y, groups, keys = [], [], [], []

    for tid, t_rows in turn_rows.items():
        seg_durations   = []    # voiced-seg durations from PRIOR pauses only
        voiced_s_so_far = 0.0
        total_s_so_far  = 0.0

        for pause_idx, r in enumerate(t_rows):
            path = os.path.join(args.data_dir, r["audio_file"])
            if path not in cache:
                cache[path] = load_wav(path)
            x_wav, sr = cache[path]

            pause_start = float(r["pause_start"])

            # Pass a *copy* of seg_durations so extract_features can't mutate it
            turn_context = {
                "pause_index"     : pause_idx,
                "seg_durations"   : list(seg_durations),
                "voiced_s_so_far" : voiced_s_so_far,
                "total_s_so_far"  : total_s_so_far,
            }

            feat = extract_features(x_wav, sr, pause_start, turn_context)
            X.append(feat)
            y.append(1 if r["label"] == "eot" else 0)
            groups.append(tid)
            keys.append((tid, r["pause_index"]))

            # Update context AFTER extracting features — strictly causal
            seg = speech_before(x_wav, sr, pause_start, window_s=1.5)
            f0  = f0_contour(seg, sr)
            seg_durations.extend(_voiced_seg_durations(f0))
            voiced_s_so_far += float((f0 > 0).sum()) * HOP_S
            total_s_so_far  += pause_start

    X, y = np.array(X), np.array(y)

    # Sanity check: no NaN in the full matrix
    nan_count = int(np.sum(~np.isfinite(X)))
    if nan_count > 0:
        print(f"WARNING: {nan_count} non-finite values found in feature matrix!")
    else:
        print("Feature matrix: all finite ✓")

    # Hold-out on turns, never split a turn across train/test
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0)
                  .split(X, y, groups))
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X[tr], y[tr])
    print(f"held-out turn accuracy: {clf.score(X[te], y[te]):.3f} "
          f"(chance ~ {max(np.mean(y), 1-np.mean(y)):.3f})")

    # Refit on everything, write predictions
    clf.fit(X, y)
    p = clf.predict_proba(X)[:, 1]
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), pi_p in zip(keys, p):
            w.writerow([tid, pi, f"{pi_p:.4f}"])
    print(f"wrote {len(keys)} predictions -> {args.out}")
    print("NOTE for your final predict.py: it must load a SAVED model and "
          "predict on unseen data, not refit like this sanity script.")


if __name__ == "__main__":
    main()
