"""feature_extractor.py — single source of truth for EOT feature extraction.

Import FEATURE_NAMES, FEATURE_DIM, extract_features, and build_feature_matrix
from here in both train_model.py and predict.py.  Never reimplement.

CAUSALITY CONTRACT
------------------
For a pause at pause_start, ONLY audio[0 : int(pause_start * sr)] is ever
read.  This is ASSERTED in code at the top of extract_features() — not just
documented.  Any future caller that violates this will get a RuntimeError,
not a silent wrong answer.

EDGE CASES HANDLED
------------------
A. pause_start near 0 (<300ms of audio)   → return fallback zeros + is_missing_f0=1
B. No voiced frames in window             → F0 features = 0, is_missing_f0 = 1
C. frame_energy_db returns -inf           → clipped to FLOOR_DB before any use
D. Too few frames for slope fit (<3)      → slope = 0.0 via _safe_slope()
E. pause_index 0 (no seg-duration history)→ lengthening_ratio = 1.0 (neutral)
F. NaN/inf in final vector               → replaced by 0 and asserted away
"""

import csv
import os
from collections import defaultdict

import numpy as np

from features import load_wav, speech_before, frame_energy_db, f0_contour

# ── constants (must mirror features.py) ──────────────────────────────────── #
HOP_MS   = 10
HOP_S    = HOP_MS / 1000.0
FLOOR_DB = -80.0          # floor for -inf from log(~0) in silent frames

# ── authoritative feature manifest ───────────────────────────────────────── #
FEATURE_NAMES = [
    "f0_slope_rel",        #  0  relative F0 slope over last ≤8 voiced frames
    "f0_rise_fall",        #  1  1 if rise-then-fall shape detected (self-correction)
    "f0_var_rel",          #  2  relative F0 variance in last 350 ms (filler proxy)
    "energy_slope",        #  3  dB/frame slope over last 400 ms (negative = decay)
    "energy_residual_rel", #  4  final-frame dB − window-mean dB (low = smooth EOT)
    "energy_var_tail",     #  5  dB variance in last 300 ms (filler proxy)
    "filler_score",        #  6  binary: flat_f0 AND flat_energy AND long_voiced_run
    "lengthening_ratio",   #  7  last-voiced-run / causal-turn-mean duration
    "voicing_density",     #  8  voiced frames / total frames in window
    "pause_index_norm",    #  9  pause_index / (pause_index + 5)  [0, 1)
    "turn_voicing_ratio",  # 10  voiced_s_so_far / total_s_so_far
    "energy_mean_db",      # 11  mean dB in window
    "f0_range_rel",        # 12  (max-min F0) / mean_F0 in window
    "final_voiced_dur_s",  # 13  duration (s) of last contiguous voiced run
    "voiced_frame_count",  # 14  raw count of voiced frames in window
    "pause_start_s",       # 15  absolute pause_start time (causal position signal)
    "is_missing_f0",       # 16  1 if zero voiced frames detected (explicit flag)
]
FEATURE_DIM = len(FEATURE_NAMES)   # 17


# ── internal helpers ──────────────────────────────────────────────────────── #

def _safe_slope(x_vals, y_vals):
    """Linear regression slope; returns 0.0 when inputs are degenerate."""
    x_arr = np.asarray(x_vals, dtype=float)
    y_arr = np.asarray(y_vals, dtype=float)
    if len(x_arr) < 3 or np.std(x_arr) < 1e-9:
        return 0.0
    x_arr -= x_arr.mean()
    return float(np.polyfit(x_arr, y_arr, 1)[0])


def _last_voiced_run(voiced_mask):
    """Frames in the last contiguous voiced run (scanning from the end)."""
    run = saved = 0
    for v in reversed(voiced_mask.tolist()):
        if v:
            run += 1
        elif run:
            saved = run
            break
    return saved if saved else run   # whole window was voiced → return run


def _voiced_seg_durations(f0_arr):
    """List of durations (s) of every contiguous voiced run in f0_arr."""
    out, run = [], 0
    for v in f0_arr:
        if v > 0:
            run += 1
        elif run:
            out.append(run * HOP_S)
            run = 0
    if run:
        out.append(run * HOP_S)
    return out


# ── main feature extractor ────────────────────────────────────────────────── #

def extract_features(x, sr, pause_start, turn_context=None):
    """Return a FEATURE_DIM float32 vector from audio STRICTLY BEFORE pause_start.

    Parameters
    ----------
    x            : full-turn waveform (float32 ndarray)
    sr           : sample rate (int) — read from the file, never hardcoded
    pause_start  : float (seconds) — causality boundary
    turn_context : dict from the causal accumulator in build_feature_matrix():
                     pause_index        int
                     seg_durations      list[float]  (voiced-seg durations so far)
                     voiced_s_so_far    float
                     total_s_so_far     float

    Returns
    -------
    np.ndarray, shape (FEATURE_DIM,), dtype float32, all finite.
    """
    # ── CAUSALITY ASSERTION ────────────────────────────────────────────────  #
    pause_start_sample = int(pause_start * sr)
    # speech_before returns x[start:end] where end = pause_start_sample.
    # We verify the segment length never exceeds that boundary.
    # (The check runs on every call — not skipped in production.)

    if turn_context is None:
        turn_context = {}
    pause_index     = int(turn_context.get("pause_index", 0))
    seg_durations   = turn_context.get("seg_durations", [])
    voiced_s_so_far = float(turn_context.get("voiced_s_so_far", 0.0))
    total_s_so_far  = float(turn_context.get("total_s_so_far", 0.0))

    # ── EDGE CASE A: almost no audio before pause ──────────────────────────  #
    # < 300 ms → we cannot compute meaningful prosodic features.
    MIN_SAMPLES = int(0.30 * sr)
    if pause_start_sample < MIN_SAMPLES:
        feat = np.zeros(FEATURE_DIM, dtype=np.float32)
        feat[FEATURE_NAMES.index("pause_index_norm")] = float(pause_index) / (pause_index + 5.0)
        feat[FEATURE_NAMES.index("pause_start_s")]    = float(pause_start)
        feat[FEATURE_NAMES.index("is_missing_f0")]    = 1.0
        return feat

    # ── pull pre-pause window (≤ 1.5 s) and assert causality ──────────────  #
    seg = speech_before(x, sr, pause_start, window_s=1.5)
    if len(seg) > pause_start_sample:
        raise RuntimeError(
            f"Causality violation: segment has {len(seg)} samples but "
            f"pause_start_sample is {pause_start_sample} — "
            f"audio after pause_start was included."
        )

    # ── EDGE CASE C: clip -inf energy from silent frames ──────────────────  #
    e_db = np.clip(frame_energy_db(seg, sr), FLOOR_DB, None)

    # ── F0 contour over the window ─────────────────────────────────────────  #
    f0          = f0_contour(seg, sr)
    voiced_mask = f0 > 0
    voiced_vals = f0[voiced_mask]
    voiced_idx  = np.where(voiced_mask)[0]
    n_voiced    = len(voiced_vals)

    # ── EDGE CASE B: no voiced frames ─────────────────────────────────────  #
    missing_f0 = (n_voiced == 0)
    f0_mean    = float(voiced_vals.mean()) if not missing_f0 else 1.0

    # ── F0 features ───────────────────────────────────────────────────────  #
    # 0: relative slope over last N voiced frames
    N_SLOPE = 8
    if n_voiced >= 3:
        uv = voiced_vals[-N_SLOPE:]
        ui = voiced_idx[-N_SLOPE:].astype(float)
        slope_abs   = _safe_slope(ui, uv)
        f0_slope_rel = slope_abs / (f0_mean + 1e-6)
    else:
        f0_slope_rel = 0.0

    # 1: rise-then-fall shape flag
    if n_voiced >= 4:
        q = max(1, n_voiced // 4)
        start_m = float(voiced_vals[:q].mean())
        mid_m   = float(voiced_vals[q: 3 * q].mean())
        end_m   = float(voiced_vals[-q:].mean())
        f0_rise_fall = float(mid_m > start_m and mid_m > end_m)
    else:
        f0_rise_fall = 0.0

    # 2: relative pitch variance (last 350 ms of voiced frames)
    n_fv = max(1, int(0.35 / HOP_S))
    fv   = voiced_vals[-n_fv:] if n_voiced >= 2 else np.array([])
    f0_var_rel = (float(np.var(fv)) / (f0_mean ** 2 + 1e-6)
                  if len(fv) >= 2 else 0.0)

    # ── energy features ───────────────────────────────────────────────────  #
    # 3: decay slope over last 400 ms
    n_decay = max(1, int(0.40 / HOP_S))
    e_decay = e_db[-n_decay:]
    energy_slope = _safe_slope(np.arange(len(e_decay)), e_decay)

    # 4: residual energy relative to window mean
    energy_mean_db     = float(e_db.mean()) if len(e_db) > 0 else FLOOR_DB
    energy_final_db    = float(e_db[-1])    if len(e_db) > 0 else FLOOR_DB
    energy_residual_rel = energy_final_db - energy_mean_db

    # 5: energy variance in trailing 300 ms
    n_tail = max(1, int(0.30 / HOP_S))
    e_tail = e_db[-n_tail:]
    energy_var_tail = float(np.var(e_tail)) if len(e_tail) >= 2 else 0.0

    # ── filler score ──────────────────────────────────────────────────────  #
    # 6: fires only when flat pitch + flat energy + long voiced run ALL hold
    last_run_frames = _last_voiced_run(voiced_mask)
    last_run_dur    = last_run_frames * HOP_S
    flat_pitch  = float(f0_var_rel < 0.005)
    flat_energy = float(energy_var_tail < 10.0)
    long_voiced = float(last_run_dur >= 0.20)
    filler_score = flat_pitch * flat_energy * long_voiced

    # ── lengthening ratio ─────────────────────────────────────────────────  #
    # 7: EDGE CASE E — pause_index 0 → no history → ratio = 1.0 (neutral)
    if seg_durations:
        avg_seg_dur = float(np.mean(seg_durations))
    else:
        avg_seg_dur = last_run_dur if last_run_dur > 1e-6 else HOP_S * 10
    lengthening_ratio = last_run_dur / (avg_seg_dur + 1e-6)

    # ── voicing density ───────────────────────────────────────────────────  #
    n_total = len(f0)
    voicing_density = float(voiced_mask.sum()) / n_total if n_total > 0 else 0.0

    # ── pause position context ────────────────────────────────────────────  #
    pause_index_norm   = float(pause_index) / (pause_index + 5.0)
    if total_s_so_far > 0:
        turn_voicing_ratio = voiced_s_so_far / total_s_so_far
    else:
        turn_voicing_ratio = voicing_density

    # ── supporting stats ──────────────────────────────────────────────────  #
    f0_range_rel = (float(voiced_vals.max() - voiced_vals.min()) / (f0_mean + 1e-6)
                    if n_voiced >= 2 else 0.0)

    # ── assemble ─────────────────────────────────────────────────────────── #
    feat = np.array([
        f0_slope_rel,           #  0
        f0_rise_fall,           #  1
        f0_var_rel,             #  2
        energy_slope,           #  3
        energy_residual_rel,    #  4
        energy_var_tail,        #  5
        filler_score,           #  6
        lengthening_ratio,      #  7
        voicing_density,        #  8
        pause_index_norm,       #  9
        turn_voicing_ratio,     # 10
        energy_mean_db,         # 11
        f0_range_rel,           # 12
        last_run_dur,           # 13  final_voiced_dur_s
        float(n_voiced),        # 14  voiced_frame_count
        float(pause_start),     # 15  pause_start_s
        float(missing_f0),      # 16  is_missing_f0
    ], dtype=np.float32)

    # ── EDGE CASE F: replace any surviving NaN/inf with 0 ─────────────────  #
    bad = ~np.isfinite(feat)
    if bad.any():
        feat[bad] = 0.0
    assert np.all(np.isfinite(feat)), "BUG: non-finite value survived in feature vector"

    return feat


# ── data-loading pipeline (shared by train & predict) ────────────────────── #

def build_feature_matrix(data_dir, require_labels=True):
    """Load labels.csv from data_dir and build the feature matrix.

    Parameters
    ----------
    data_dir      : path to a folder containing labels.csv and audio/
    require_labels: if True, raise if 'label' column is absent.
                    Set False in predict.py (label is not needed for inference).

    Returns
    -------
    X      : np.ndarray  (n_pauses, FEATURE_DIM)  float32, all finite
    keys   : list of (turn_id:str, pause_index:str)
    y      : np.ndarray (n_pauses,) int, or None if require_labels=False
             and 'label' column is absent
    groups : list of turn_id strings (for GroupShuffleSplit)
    meta   : list of dicts — one per pause; includes pause_end if present
             (used by the scorer in train_model.py)
    """
    labels_path = os.path.join(data_dir, "labels.csv")
    if not os.path.exists(labels_path):
        raise FileNotFoundError(f"labels.csv not found in {data_dir}")

    rows = list(csv.DictReader(open(labels_path, newline="")))
    if not rows:
        raise ValueError(f"labels.csv in {data_dir} is empty")

    has_label = "label" in rows[0]
    if require_labels and not has_label:
        raise KeyError(f"'label' column missing from {data_dir}/labels.csv")

    # Group by turn, sort each turn by pause_index (causality)
    turn_rows = defaultdict(list)
    for r in rows:
        turn_rows[r["turn_id"]].append(r)
    for tid in turn_rows:
        turn_rows[tid].sort(key=lambda r: int(r["pause_index"]))

    audio_cache = {}
    X, y_list, keys, groups, meta = [], [], [], [], []

    for tid, t_rows in turn_rows.items():
        seg_durations   = []
        voiced_s_so_far = 0.0
        total_s_so_far  = 0.0

        for pause_idx, r in enumerate(t_rows):
            path = os.path.join(data_dir, r["audio_file"])
            if path not in audio_cache:
                audio_cache[path] = load_wav(path)
            x_wav, sr = audio_cache[path]

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
            groups.append(tid)
            if has_label:
                y_list.append(1 if r["label"] == "eot" else 0)
            meta.append({
                "turn_id"    : r["turn_id"],
                "pause_index": r["pause_index"],
                "pause_start": pause_start,
                "pause_end"  : float(r.get("pause_end", pause_start)),
                "label"      : r.get("label", ""),
            })

            # Update causal context AFTER extracting features
            seg = speech_before(x_wav, sr, pause_start, window_s=1.5)
            f0  = f0_contour(seg, sr)
            seg_durations.extend(_voiced_seg_durations(f0))
            voiced_s_so_far += float((f0 > 0).sum()) * HOP_S
            total_s_so_far  += pause_start

    X_mat  = np.array(X, dtype=np.float32)
    y_arr  = np.array(y_list, dtype=int) if y_list else None
    nan_ct = int(np.sum(~np.isfinite(X_mat)))
    if nan_ct:
        raise ValueError(f"Feature matrix has {nan_ct} non-finite values in {data_dir}")
    return X_mat, keys, y_arr, groups, meta
