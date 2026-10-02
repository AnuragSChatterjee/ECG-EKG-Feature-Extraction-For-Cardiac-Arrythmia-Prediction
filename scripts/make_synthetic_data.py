"""
make_synthetic_data.py
----------------------
Generates SYNTHETIC tachycardia ECG recordings (NeuroKit2 simulator) in the
folder layout expected by src/extract_features.py, so the full pipeline can be
run end-to-end without access to the private clinical dataset.

The synthetic signals are NOT clinically meaningful - they only exercise the code.

Usage:
    python scripts/make_synthetic_data.py --out data_synthetic --n 10
"""
import argparse
import os

import numpy as np
import scipy.io
import neurokit2 as nk

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="data_synthetic")
ap.add_argument("--n", type=int, default=10, help="recordings per class")
args = ap.parse_args()

rng = np.random.default_rng(0)
FS = 10_000
for label, hr in [("AVNRT", 210), ("AVRT", 195)]:
    for i in range(args.n):
        d = os.path.join(args.out, label, f"session_{i:02d}", "run_1")
        os.makedirs(d, exist_ok=True)
        sig = nk.ecg_simulate(duration=7.5, sampling_rate=FS, heart_rate=hr + rng.normal(0, 8),
                              noise=0.02, random_state=int(rng.integers(1_000_000)))
        t = np.arange(len(sig)) / FS
        scipy.io.savemat(os.path.join(d, "ECG_synthetic.mat"), {
            "fs": FS, "time": t,
            "ch1_acq": sig + rng.normal(0, 0.05, len(sig)),
            "ch3_acq": sig + rng.normal(0, 0.01, len(sig)),
            "ch4_acq": 0.9 * sig,
        })
print(f"Wrote {2 * args.n} synthetic recordings to {args.out}/")
