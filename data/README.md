# Data

**No patient data is included in this repository.**

The original dataset consists of intraoperative clinical ECG recordings used
under the lab's research protocol and is not publicly shareable. Raw signals,
extracted feature tables, per-recording predictions, recording dates and file
names have all been intentionally left out.

To run the pipeline on your own data, place `.mat` files in this layout:

```
data/
  AVNRT/<session>/<run>/ECG*.mat
  AVRT/<session>/<run>/ECG*.mat
```

Each `.mat` file must contain the variables:

| Variable  | Description                          |
|-----------|--------------------------------------|
| `fs`      | Sampling rate in Hz (10 000 in the original study) |
| `time`    | Time vector (s)                      |
| `ch1_acq` | ECG channel 1 (excluded from modelling – noisy) |
| `ch3_acq` | ECG channel 3 (primary channel)      |
| `ch4_acq` | ECG channel 4                        |

To try the code without real data, generate synthetic recordings:

```bash
python scripts/make_synthetic_data.py --out data_synthetic
python src/extract_features.py --data-dir data_synthetic --out features.xlsx
```
