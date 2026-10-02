# Feature Reference

All features are computed per channel on the band-passed signal (0.5–50 Hz) after
R-peak detection and NeuroKit2 DWT delineation. Around 90 features per channel are
extracted in total; the tables below cover the ones used in modelling.

## Clinical background: AVNRT vs AVRT

Both are re-entrant supraventricular tachycardias (SVT). They differ in the circuit:

| | AVNRT | AVRT |
|---|---|---|
| Circuit | Small loop inside / around the AV node (slow and fast pathways) | Large loop: AV node + an accessory pathway |
| Retrograde P wave | Buried in, or right after, the QRS (pseudo-R′ in V1) | Visible after the QRS, often in the ST segment |
| RP interval | Short (RP′ < ~90 ms) | Long (RP ≥ ~100 ms) |
| RP/PR ratio | < 1 | Often > 1 |

## Candidate feature set (8 categories)

| Category | Features |
|---|---|
| Temporal / HRV | RR mean, median, std, min, max, IQR, variance · HR mean/std · RMSSD · SDSD · pNN50 |
| Morphology | QRS amplitude & duration · beat area · skewness · kurtosis · S-nadir · RS interval · QS pattern · ST slope · T amplitude/duration · peak-to-peak · signal integral |
| Interval | P duration/amplitude/presence ratio · P-absent flag · PR · QT · QTc (Bazett) · Q onset · T offset · **RP** · **VA** · **RP/PR ratio** · R/T axis proxies |
| QRS alternans | Beat-to-beat R amplitude std + flag |
| Spectral (Welch PSD) | P-band (0.67–5 Hz), T-band (1–7 Hz), QRS-band (10–50 Hz) power · peak & centroid frequency · total energy · spectral flatness · AMSA |
| Nonlinear | Poincaré SD1, SD2, SD1/SD2 · Higuchi fractal dimension · approximate & sample entropy · Lyapunov-exponent proxy |
| Wavelet | DWT (db4) detail energy, levels 1–4 |
| R-peak | Amplitude mean/std/max · R-peak count |

## Final model features (19)

Selected using Mann-Whitney U tests (p < 0.05), effect size, and missing-value rate.
Channel 3 was the primary channel. Only two channel-4 features were significant, and
channel 1 was dropped because it was noisy.

| Tier | Features | Notes |
|---|---|---|
| 1. Strongest separators | `pNN50`, `HR_std`, `beat_skewness`, `beat_kurtosis`, `QRS_duration`, `P_wave_presence_ratio`, `ST_slope`, `R_peak_count` | Low NaN rate, significant |
| 2. Strong effect, partly missing | `RP_interval`, `RP_PR_ratio` | ~28–30% NaN at 200+ BPM, median-imputed (or native NaN in XGBoost) |
| 3. Moderate | `SD1`, `SD2`, `SD1/SD2`, `RR_std`, `RMSSD`, `RR_mean`, `P_wave_absent_flag` | Significant, smaller effect |
| Channel 4 | `SD1/SD2`, `P_wave_absent_flag` | |

## Implementation notes

- **R-peak detection:** NeuroKit2's internal Pan-Tompkins detector halves the beat
  count above ~210 BPM because of its refractory period. R-peaks are found with
  `scipy.signal.find_peaks` (200 ms minimum distance) and then passed to `ecg_delineate`.
- **Statistics:** sample std/variance (`ddof=1`) and bias-corrected skewness/kurtosis
  are used so results match the lab's earlier MATLAB implementation.
- **Short recordings:** standard LF/HF HRV bands (0.04–0.4 Hz) need minutes of data,
  which a 7.5 s recording doesn't have. ECG-morphology spectral bands are used instead.
