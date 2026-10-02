"""
extract_features.py
-------------------
Runs the full signal-processing pipeline on every intraoperative ECG recording
and writes one ML-ready row per recording.

Pipeline per recording (.mat, fs = 10 kHz, ~7.5 s):
  1. Zero-phase Butterworth band-pass, 0.5-50 Hz (SOS form, order 4)
  2. R-peak detection with scipy.signal.find_peaks (height 0.5, min distance 200 ms)
     -> NeuroKit2's built-in Pan-Tompkins is bypassed because its refractory
        period halves the beat count at 210+ BPM
  3. P/Q/S/T delineation with NeuroKit2 ecg_delineate (DWT) on the supplied R-peaks
  4. Feature extraction (8 categories) on channels 3 and 4
  5. Per-recording QC plots (raw signals, Poincare) and an Excel feature matrix

Expected data layout (NOT included in this repository - see data/README.md):

    data/
      AVNRT/<session>/<run>/ECG*.mat
      AVRT/<session>/<run>/ECG*.mat

Each .mat file must contain: fs, time, ch1_acq, ch3_acq, ch4_acq.

Usage:
    python src/extract_features.py --data-dir data --out features.xlsx

Author: Anurag Chatterjee
"""
import argparse
import glob
import os
import warnings

import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import find_peaks

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ecg_features import bandpass_filter, run_delineation, extract_all_features

warnings.filterwarnings("ignore")

LABELS = {"AVNRT": 0, "AVRT": 1}

# Full candidate feature list (per channel) exported to the feature matrix.
ML_FEATURES = [
    # Rhythm / HRV
    "RR_mean_ms", "RR_std_ms", "RMSSD_ms",
    "HR_mean_bpm", "HR_std_bpm", "pNN50_pct",
    # QRS morphology
    "QRS_amplitude_mean", "QRS_duration_ms_mean",
    "beat_skewness_mean", "beat_kurtosis_mean",
    "RS_interval_ms_mean",
    # Spectral
    "QRS_band_power", "P_wave_band_power",
    "peak_frequency_Hz", "spectral_flatness",
    # Poincare
    "SD1_ms", "SD2_ms", "SD1_SD2_ratio",
    # Intervals
    "PR_interval_mean_ms", "QT_interval_mean_ms",
    "P_wave_presence_ratio", "P_duration_mean_ms",
    # R-peak
    "R_peak_amplitude_mean", "R_peak_count",
    # AVNRT vs AVRT specific
    "RP_interval_mean_ms",   # short RP (<~90 ms) -> AVNRT; long RP (>=100 ms) -> AVRT
    "VA_interval_mean_ms",   # same clinical meaning as RP interval
    "RP_PR_ratio",           # <1 short-RP (AVNRT); >1 long-RP (AVRT)
    "ST_slope_mean",         # ST depression reported as a predictor of AVRT
    "P_wave_absent_flag",    # buried P -> AVNRT; visible retrograde P -> AVRT
]

# R-peak detection parameters
MIN_HEIGHT = 0.5
MIN_DIST_S = 0.20


def find_recordings(data_dir):
    """Return a list of (path, label) for every ECG*.mat file under data_dir/<label>/."""
    recs = []
    for label in LABELS:
        pattern = os.path.join(data_dir, label, "**", "ECG*.mat")
        for path in sorted(glob.glob(pattern, recursive=True)):
            recs.append((path, label))
    return recs


def process_one_file(mat_path, label, rec_id, plot_dir=None):
    """Full pipeline on one recording. Returns an ML-ready dict or None on failure."""
    print(f"\n{'=' * 65}\n  {rec_id}  |  LABEL: {label}\n{'=' * 65}")

    try:
        data = {k: v for k, v in scipy.io.loadmat(mat_path).items() if not k.startswith("__")}
        fs = float(data["fs"].squeeze())
        time = data["time"].squeeze()
        ch1 = data["ch1_acq"].squeeze().astype(float)
        ch3 = data["ch3_acq"].squeeze().astype(float)
        ch4 = data["ch4_acq"].squeeze().astype(float)
    except Exception as e:  # noqa: BLE001
        print(f"  ERROR loading: {e} - skipping.")
        return None

    print(f"  fs={fs:.0f} Hz | duration={time[-1]:.2f} s | samples={len(time)}")

    # --- QC plot: raw signals --------------------------------------------
    if plot_dir:
        out_folder = os.path.join(plot_dir, rec_id)
        os.makedirs(out_folder, exist_ok=True)
        fig, axes = plt.subplots(3, 1, figsize=(15, 7), sharex=True)
        fig.suptitle(f"Raw signals - {label} - {rec_id}", fontsize=11)
        for ax, ch, name, col in zip(axes, [ch1, ch3, ch4],
                                     ["Channel 1", "Channel 3", "Channel 4"],
                                     ["royalblue", "darkorange", "green"]):
            ax.plot(time, ch, color=col, linewidth=0.6)
            ax.set_title(name); ax.set_ylabel("Amplitude"); ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout(); plt.savefig(os.path.join(out_folder, "raw_signals.png"), dpi=120); plt.close()

    # --- Band-pass filter ---------------------------------------------------
    # Channel 1 was excluded from modelling (noisy / irregular signal).
    try:
        ch3_f = bandpass_filter(ch3, fs)
        ch4_f = bandpass_filter(ch4, fs)
    except Exception as e:  # noqa: BLE001
        print(f"  ERROR filtering: {e} - skipping.")
        return None
    if not (np.all(np.isfinite(ch3_f)) and np.all(np.isfinite(ch4_f))):
        print("  ERROR: filter produced Inf/NaN - skipping.")
        return None

    # --- R-peak detection ---------------------------------------------------
    dist = int(MIN_DIST_S * fs)
    r3, _ = find_peaks(ch3_f, height=MIN_HEIGHT, distance=dist)
    r4, _ = find_peaks(ch4_f, height=MIN_HEIGHT, distance=dist)
    print(f"  R-peaks - Ch3: {len(r3)} | Ch4: {len(r4)}")

    # Data-quality check: a doubled RR usually means a missed beat / baseline wander
    for name, locs in [("Ch3", r3), ("Ch4", r4)]:
        if len(locs) >= 2:
            rr = np.diff(locs) / fs * 1000
            if np.max(rr) > 2 * np.median(rr):
                print(f"  WARNING: {name} doubled RR (max={np.max(rr):.0f} ms vs "
                      f"median={np.median(rr):.0f} ms) - features may be unreliable.")

    # --- Delineation + features --------------------------------------------
    waves3 = run_delineation(ch3_f, r3, fs, "Ch3")
    waves4 = run_delineation(ch4_f, r4, fs, "Ch4")
    f3, rr3 = extract_all_features(ch3_f, ch3, r3, waves3, fs, "Channel_3")
    f4, rr4 = extract_all_features(ch4_f, ch4, r4, waves4, fs, "Channel_4")

    # --- QC plot: Poincare --------------------------------------------------
    if plot_dir:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(f"Poincare - {label} - {rec_id}", fontsize=10)
        for ax, rr, name, col in zip(axes, [rr3, rr4], ["Ch3", "Ch4"], ["darkorange", "green"]):
            if len(rr) >= 2:
                ax.scatter(rr[:-1], rr[1:], alpha=0.7, color=col, s=30)
            ax.set_xlabel("RR_n (ms)"); ax.set_ylabel("RR_{n+1} (ms)")
            ax.set_title(f"Poincare - {name}"); ax.grid(True, alpha=0.3); ax.set_aspect("equal")
        plt.tight_layout(); plt.savefig(os.path.join(out_folder, "poincare.png"), dpi=120); plt.close()

    row = {"recording_id": rec_id, "label": label, "label_num": LABELS[label]}
    row.update({f"ch3_{k}": f3.get(k, np.nan) for k in ML_FEATURES})
    row.update({f"ch4_{k}": f4.get(k, np.nan) for k in ML_FEATURES})
    return row


def print_class_summary(df):
    """Mean +/- std of each channel-3 feature, per class."""
    print(f"\n  {'Feature':<28} {'AVNRT mean':>12} {'AVNRT std':>10} {'AVRT mean':>12} {'AVRT std':>10}")
    print(f"  {'-' * 74}")
    for feat in ML_FEATURES:
        col = f"ch3_{feat}"
        a = df.loc[df.label == "AVNRT", col].dropna()
        b = df.loc[df.label == "AVRT", col].dropna()
        print(f"  {feat:<28} {a.mean():>12.4f} {a.std():>10.4f} {b.mean():>12.4f} {b.std():>10.4f}")


def main():
    ap = argparse.ArgumentParser(description="ECG feature extraction for AVNRT vs AVRT")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out", default="features.xlsx")
    ap.add_argument("--plot-dir", default="outputs", help="QC plots folder ('' to disable)")
    args = ap.parse_args()

    recs = find_recordings(args.data_dir)
    if not recs:
        raise SystemExit(f"No ECG*.mat files found under {args.data_dir}/AVNRT or {args.data_dir}/AVRT")
    print(f"Found {len(recs)} recordings")

    rows, failed = [], []
    for i, (path, label) in enumerate(recs, start=1):
        rec_id = f"REC_{i:03d}"  # anonymised ID - raw file names are never written out
        row = process_one_file(path, label, rec_id, args.plot_dir or None)
        (rows if row else failed).append(row or rec_id)

    print(f"\nProcessed: {len(rows)} | Failed: {len(failed)} {failed if failed else ''}")
    if rows:
        df = pd.DataFrame(rows)
        print_class_summary(df)
        df.to_excel(args.out, index=False)
        print(f"\nSaved feature matrix -> {args.out}  ({df.shape[0]} rows x {df.shape[1]} cols)")


if __name__ == "__main__":
    main()
