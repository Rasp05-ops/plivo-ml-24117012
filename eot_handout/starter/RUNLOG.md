# RUNLOG.md — EOT Detector: Error-Driven Iteration Log

All runs: `train_model.py --data_dir ../eot_data/english ../eot_data/hindi`
then `predict.py` + `score.py` on each language. Model throughout: Pipeline(drop_col15 →
StandardScaler → LR C=0.1, uniform class weights), trained on EN+HI combined (496 pauses).

---

## Baseline (Iteration 0)

**Model**: 15-feature LR pipeline (pause_start_s dropped, StandardScaler, C=0.1)

| Language | AUC   | Mean Delay | Interrupted |
|----------|-------|------------|-------------|
| English  | 0.672 | 1250 ms    | 5.0%        |
| Hindi    | 0.660 |  850 ms    | 5.0%        |

**Val split (held-out 25%)**: AUC=0.631, overall delay=1156ms

**Error analysis (English, 248 pauses)**:
- False positives: **27** hold pauses predicted as EOT
- False negatives: **66** EOT pauses predicted as hold
- Top-10 worst errors: ALL are `pause_index=0` EOT turns (pi_norm=0.000 pushes model toward hold)
- `filler_score` fires **0 times** — `energy_var_tail < 10.0` never satisfied (actual mean: eot≈251, hold≈307 dB²)
- `lengthening_ratio=1.000` for every `pause_index=0` — neutral fallback adds no discriminative signal
- `energy_slope` (400ms window) averages early-stable and late-dropping energy together, diluting the terminal decay

---

## Iteration 1 — Fix dead filler_score (`energy_var_tail` threshold)

**What I heard in the errors**: The model was generating 27 false positives (hold pauses with sustained flat voiced runs, which are fillers "ummm" / "ahhh"). The `filler_score` feature was intended to detect these and suppress EOT probability, but the `energy_var_tail < 10.0` threshold meant it never fired — actual energy variance is 250–300 dB². Relaxing to 200 dB² (below the data mean of ~280) lets the detector actually trigger on genuinely flat-energy sustained voiced runs.

**Change** (`train.py`, line 249):
```python
# was: is_flat_energy = float(energy_var_tail < 10.0)
is_flat_energy = float(energy_var_tail < 200.0)
```

**Result**:

| Language | AUC   | Mean Delay | Interrupted | Δ vs baseline |
|----------|-------|------------|-------------|----------------|
| English  | 0.672 | **1180 ms**| 5.0%        | **−70 ms** ✓  |
| Hindi    | 0.663 |  850 ms    | 5.0%        | 0 ms           |

Val AUC: 0.641 (+0.010 vs baseline). Val mean delay: 1156ms (no change — val split has fewer fillers).

**Verdict**: Effective for English. filler_score now fires on ~18% of hold pauses (previously 0%). Reduces false positives from 27 to ~22.

---

## Iteration 2 — Replace 400ms energy_slope with energy_drop_last_200ms

**What I heard in the errors**: The 10 worst false negatives all had strong energy decays (-1.4 to -1.9 dB/frame) into the pause. These were being averaged into a 400ms window that included earlier (more stable) energy, diluting the terminal signal. The full-window slope sometimes showed near-zero slope even when the final 200ms had a sharp drop.

**Change** (`train.py`, feature 3 block):
Replace linear regression over 400ms with a direct drop comparison:
```python
# energy_slope = mean(mid-200ms energy) − mean(final-200ms energy)
# Positive = energy dropped sharply at end = EOT cue
n_half = max(1, int(0.20 / HOP_S))   # ~20 frames
energy_slope = mid_e.mean() - tail_e.mean()
```

**Result**:

| Language | AUC   | Mean Delay | Interrupted | Δ vs Iter 1   |
|----------|-------|------------|-------------|----------------|
| English  | 0.668 | 1215 ms    | 5.0%        | +35 ms ✗      |
| Hindi    | 0.659 |  850 ms    | 5.0%        | 0 ms           |

Val AUC: 0.630 (−0.011). Val mean delay: 1156ms unchanged.

**Verdict**: Slight regression on English score.py. The 200ms window is noisier than 400ms at the overall level — it increases variance for mid-turn hold pauses that also have short local energy dips. Feature retained (the theoretical motivation holds and val difference is small), but noted as mixed.

---

## Iteration 3 — Better lengthening_ratio fallback for pause_index=0

**What I heard in the errors**: Every one of the 10 worst false negatives was `pause_index=0`. At the first pause, `seg_durations=[]` so `lengthening_ratio` used to fall back to a self-referential ratio of 1.0 (last segment = its own average). This gave the model exactly zero information about final-segment length for first-pause turns. Replace the fallback with normalization by a fixed typical syllable duration (150ms), matching conversational English/Hindi prosody.

**Change** (`train.py`, lengthening_ratio else-branch):
```python
# was: avg_seg_dur = last_run_dur  → ratio always 1.0
TYPICAL_SYL_DUR_S = 0.15   # 150ms typical conversational syllable
avg_seg_dur = TYPICAL_SYL_DUR_S
lengthening_ratio = np.clip(last_run_dur / avg_seg_dur, 0.0, 4.0)
```

**Result**:

| Language | AUC   | Mean Delay | Interrupted | Δ vs Iter 2   |
|----------|-------|------------|-------------|----------------|
| English  | 0.666 | 1215 ms    | 5.0%        | 0 ms           |
| Hindi    | 0.658 |  850 ms    | 5.0%        | 0 ms           |

Val AUC: 0.626. Val overall mean delay: **1060 ms** (−96ms vs baseline 1156ms) — improvement is visible on val but score.py (in-sample) doesn't move.

**Verdict**: Neutral on score.py; meaningful improvement on held-out val overall delay. The feature now correctly ranks first-pause lengthening (long final segment = high ratio = stronger EOT signal) instead of always outputting 1.0.

---

## Summary

| | Baseline | Iter 1 | Iter 2 | Iter 3 |
|--|----------|--------|--------|--------|
| EN score.py (ms) | 1250 | **1180** ← best | 1215 | 1215 |
| HI score.py (ms) | 850 | 850 | 850 | 850 |
| Val AUC | 0.631 | **0.641** ← best | 0.630 | 0.626 |
| Val delay (ms) | 1156 | 1156 | 1156 | **1060** ← best |

Best single-language gain: **−70ms on English** from fixing the dead `filler_score` threshold.
Best held-out val delay: **1060ms** after all 3 iterations combined.
