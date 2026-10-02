# Literature Review Summary

Before building the pipeline, I ran two literature reviews. The first chose which
features to extract. The second set comparison baselines for the classifier.

## 1. 12-lead ECG arrhythmia prediction: feature survey

I reviewed papers on deep-learning and feature-based arrhythmia prediction
(AF risk, sudden cardiac death risk, multi-class 12-lead classification, PhysioNet/CinC
2020). From them I built a feature list grouped into ten families: temporal, HRV,
frequency-domain, time-frequency (wavelet), morphological, nonlinear/statistical,
spectral, clinical/demographic, and end-to-end deep representations.

Main takeaways:
- Most feature-based work leans on 1-D time-domain features and under-uses
  frequency-domain information, so spectral and wavelet features were added.
- Nonlinear descriptors (entropy, fractal dimension, Poincaré) capture rhythm
  regularity, which is what separates re-entrant SVT from irregular rhythms.
- Demographic and medication features were left out on purpose, because the task is
  signal-only classification during the procedure.

## 2. SVT / VT classification

| Work | Task | Method | Reported result |
|---|---|---|---|
| PubMed 38246906 | VT vs SVT (wide QRS) | Gradient boosting, 40 hand-crafted ECG features | AUC 0.97 |
| arXiv 2507.14196 | VT vs SVT with aberrancy | Per-lead 1-D CNN + LSTM, SHAP | 95.6% accuracy |
| arXiv 2112.12953 | NSR / SVT / VT / VF | Decision tree + autoregressive coefficients | 97% accuracy |
| PMC10532022 | VT / VF detection | 2-D CNN on time-frequency images | 99.1% accuracy |
| arXiv 2509.25804 | Arrhythmia classification | CardioForest ensemble | — |

## 3. AVNRT vs AVRT (the harder intra-SVT problem)

Most published ML work addresses SVT vs VT. Telling AVNRT from AVRT is a narrower,
harder problem within SVT.

- **Primary baseline:** a CNN trained on 12-lead ECG reached about **0.77 AUC** for
  AVNRT vs AVRT (*Heart Rhythm O2*, doi:10.1016/j.hroo.2023.01.001).
- A ResNet-34 model reached AUROC 0.726.
- Clinical algorithm papers agree that the **RP interval** is the main
  electrophysiological discriminator: RP′ < 90 ms points to AVNRT and RP ≥ 100 ms
  points to AVRT, with paediatric cohorts reporting a mean RP′ of about 86 ms for AVNRT.
- Lab context: electromechanical wave imaging (EWI) has been used to tell AVNRT from
  AVRT using atrioventricular activation delays (Proestaki *et al.*, *Heart Rhythm*
  2026, doi:10.1016/j.hrthm.2026.03.829). This project looks at what the
  intraoperative ECG alone can contribute.
