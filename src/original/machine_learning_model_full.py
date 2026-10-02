# =============================================================================
#  Machine Learning — ECG Feature Extraction for All AVNRT & AVRT Datasets And AVNRT vs AVRT Classification Models
# =============================================================================
#  REQUIREMENTS:
#    pip install neurokit2 scipy numpy pandas matplotlib openpyxl PyWavelets scikit-learn
#
#  HOW TO RUN:
#    python "Machine Learning Model.py"
#    Must be run from the directory containing the AVNRT and AVRT session folders.
#
#  OUTPUT:
#    ECG_ALL_ML_features.xlsx — one row per recording, all ML features + label
#    outputs/<filename>/      — raw signal plot and Poincaré plot per recording
# =============================================================================

# =============================================================================
#  IMPORTS
# =============================================================================
import os
import math
import pandas as pd
import numpy as np
from sklearn import model_selection
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, ConfusionMatrixDisplay)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

import scipy.io
from scipy.signal import butter, sosfiltfilt, find_peaks

try:
    import neurokit2 as nk
    NK_AVAILABLE = True
except ImportError:
    NK_AVAILABLE = False
    print("WARNING: neurokit2 not installed. P/T-wave delineation skipped.")
    print("Install: pip install neurokit2")

try:
    import pywt
    PWT_AVAILABLE = True
except ImportError:
    PWT_AVAILABLE = False
    print("WARNING: PyWavelets not installed. DWT features will be NaN.")
    print("Install: pip install PyWavelets")


# =============================================================================
#  SECTION A — LOAD ALL AVAILABLE ECG DATASETS
# =============================================================================

# Session folder names anonymised for the public repository
# (original names contained procedure dates).
AVNRT_FILES = [f'AVNRT_session_{i:02d}' for i in range(1, 9)]   # 8 AVNRT patient sessions
AVRT_FILES  = [f'AVRT_session_{i:02d}'  for i in range(1, 8)]   # 7 AVRT patient sessions
# dataset_paths = AVNRT_FILES + AVRT_FILES

def AVNRT_datasets():
    AVNRT_datasets_gathered = []
    for file in AVNRT_FILES:
        for dataset in os.listdir(file):
            print(f"{file}/{dataset}")
            for ECG_files in os.listdir(f"{file}/{dataset}"):
                if ECG_files.startswith('ECG') and ECG_files.endswith('.mat'):
                    AVNRT_datasets_gathered.append(f"{file}/{dataset}/{ECG_files}")
                    # print(f"  {AVNRT_datasets_gathered}")
    return AVNRT_datasets_gathered

def AVRT_datasets():
    AVRT_datasets_gathered = []
    for file in AVRT_FILES:
        for dataset in os.listdir(file):
            print(f"{file}/{dataset}")
            for ECG_files in os.listdir(f"{file}/{dataset}"):
                if ECG_files.startswith('ECG') and ECG_files.endswith('.mat'):
                    AVRT_datasets_gathered.append(f"{file}/{dataset}/{ECG_files}")
                    # print(f"  {AVRT_datasets_gathered}")
    return AVRT_datasets_gathered

avnrt_datasets = AVNRT_datasets()
avrt_datasets = AVRT_datasets()
print("AVNRT Datasets:", avnrt_datasets)
print("AVRT Datasets:", avrt_datasets)
print(f"\nTotal AVNRT: {len(avnrt_datasets)}  |  Total AVRT: {len(avrt_datasets)}")
print(f"Total combined: {len(avnrt_datasets) + len(avrt_datasets)} recordings\n")

# =============================================================================
#  SECTION B — ML FEATURE LIST
#  Your finalised feature list — both versions combined into one master list.
#  All AVNRT vs AVRT specific features included.
# =============================================================================

ML_FEATURES = [
    # Rhythm / HRV
    'RR_mean_ms', 'RR_std_ms', 'RMSSD_ms',
    'HR_mean_bpm', 'HR_std_bpm', 'pNN50_pct',

    # QRS Morphology
    'QRS_amplitude_mean', 'QRS_duration_ms_mean',
    'beat_skewness_mean', 'beat_kurtosis_mean',
    'RS_interval_ms_mean',

    # Spectral
    'QRS_band_power', 'P_wave_band_power',
    'peak_frequency_Hz', 'spectral_flatness',

    # Poincaré
    'SD1_ms', 'SD2_ms', 'SD1_SD2_ratio',

    # Interval
    'PR_interval_mean_ms', 'QT_interval_mean_ms',
    'P_wave_presence_ratio', 'P_duration_mean_ms',

    # R-peak
    'R_peak_amplitude_mean', 'R_peak_count',

    # *** AVNRT vs AVRT specific features ***
    'RP_interval_mean_ms',   # PRIMARY: <70-90ms = AVNRT; >100ms = AVRT
    'VA_interval_mean_ms',   # Same clinical meaning as RP interval
    'RP_PR_ratio',           # <1.0 = short RP (AVNRT); >1.0 = long RP (AVRT)
    'ST_slope_mean',         # ST depression predictor of AVRT
    'P_wave_absent_flag',    # P buried/absent = AVNRT; visible retrograde P = AVRT
]

# =============================================================================
#  SECTION C — SIGNAL PROCESSING HELPER FUNCTIONS
#  Identical to ECG_Feature_Extraction.py — all 18 feature categories.
# =============================================================================

def bandpass_filter(signal, fs, low=0.5, high=50.0, order=4):
    """Zero-phase Butterworth bandpass filter. Stable at fs=10000 via SOS form."""
    sos = butter(order, [low, high], btype='band', fs=fs, output='sos')
    return sosfiltfilt(sos, signal)


def run_delineation(signal, r_locs, fs, ch_name):
    """
    Supply pre-detected R-peaks to NeuroKit2 for waveform boundary detection only.
    Bypasses NK2's internal Pan-Tompkins which halves beat count at 210+ BPM.
    """
    if not NK_AVAILABLE or len(r_locs) < 2:
        return None
    try:
        _, waves = nk.ecg_delineate(
            signal.astype(float),
            rpeaks=r_locs.astype(int),
            sampling_rate=int(fs),
            method='dwt',
            show=False
        )
        p_found = sum(
            1 for v in waves.get('ECG_P_Peaks', [])
            if v is not None and not (isinstance(v, float) and np.isnan(v))
        )
        print(f"    NK2 OK on {ch_name} — P-peaks found: {p_found}")
        return waves
    except Exception as e:
        print(f"    NK2 failed on {ch_name}: {e}")
        return None


def rr_features(r_locs, fs):
    """RR interval and HRV temporal features. ddof=1 matches MATLAB default."""
    if len(r_locs) < 2:
        return {}, np.array([])
    RR_s  = np.diff(r_locs) / fs
    RR_ms = RR_s * 1000
    HR    = 60.0 / RR_s
    dRR   = np.abs(np.diff(RR_ms))
    feats = {
        'RR_mean_ms'          : float(np.mean(RR_ms)),
        'RR_median_ms'        : float(np.median(RR_ms)),
        'RR_std_ms'           : float(np.std(RR_ms,  ddof=1)),
        'RR_min_ms'           : float(np.min(RR_ms)),
        'RR_max_ms'           : float(np.max(RR_ms)),
        'RR_IQR_ms'           : float(np.percentile(RR_ms,75)-np.percentile(RR_ms,25)),
        'RR_variance'         : float(np.var(RR_ms,  ddof=1)),
        'HR_mean_bpm'         : float(np.mean(HR)),
        'HR_std_bpm'          : float(np.std(HR,     ddof=1)),
        'RMSSD_ms'            : float(np.sqrt(np.mean(np.diff(RR_ms)**2))),
        'SDSD_ms'             : float(np.std(np.diff(RR_ms), ddof=1)),
        'pNN50_pct'           : float(100*np.sum(dRR>50)/len(dRR)) if len(dRR)>0 else np.nan,
        'atrial_rate_bpm'     : float(np.mean(HR)),
        'ventricular_rate_bpm': float(np.mean(HR)),
    }
    return feats, RR_ms


def morph_features(signal, r_locs, fs):
    """
    Per-beat morphological features.
    bias=False on skew/kurt matches MATLAB corrected estimators.
    Windows: 150ms before to 250ms after R-peak (same as MATLAB script).
    """
    from scipy.stats import skew, kurtosis as kurt_fn
    pre   = int(0.15 * fs)
    post  = int(0.25 * fs)
    s_win = int(0.080 * fs)
    st_s  = int(0.060 * fs)
    st_e  = int(0.100 * fs)
    t_s   = int(0.100 * fs)
    t_e   = int(0.250 * fs)
    n    = len(r_locs)
    keys = ['QRS_amplitude','QRS_duration_ms','beat_area',
            'beat_skewness','beat_kurtosis',
            'S_nadir_amplitude','RS_interval_ms','QS_pattern',
            'ST_slope','T_amplitude','T_duration_ms',
            'amplitude_range','peak_to_peak','signal_integral']
    arrs  = {k: np.full(n, np.nan) for k in keys}
    valid = 0
    for i, r in enumerate(r_locs):
        si = r - pre;  ei = r + post
        if si < 0 or ei >= len(signal): continue
        seg = signal[si:ei]
        if not np.all(np.isfinite(seg)): continue
        valid += 1
        r_in = pre
        arrs['QRS_amplitude'][i]   = np.max(seg) - np.min(seg)
        half                        = 0.5 * np.max(seg)
        arrs['QRS_duration_ms'][i] = np.sum(seg > half) / fs * 1000
        arrs['beat_area'][i]       = np.trapz(seg)
        arrs['beat_skewness'][i]   = float(skew(seg,    bias=False))
        arrs['beat_kurtosis'][i]   = float(kurt_fn(seg, bias=False))
        s_seg = seg[r_in : r_in + s_win]
        if len(s_seg) > 0:
            s_rel = int(np.argmin(s_seg))
            arrs['S_nadir_amplitude'][i] = float(s_seg[s_rel])
            arrs['RS_interval_ms'][i]    = s_rel / fs * 1000
        arrs['QS_pattern'][i] = 1.0 if signal[r] <= 0 else 0.0
        st_seg = seg[r_in + st_s : r_in + st_e]
        if len(st_seg) >= 3:
            x = np.arange(len(st_seg)) / fs * 1000
            slope, _ = np.polyfit(x, st_seg, 1)
            arrs['ST_slope'][i] = float(slope)
        t_seg = seg[r_in + t_s : r_in + t_e]
        if len(t_seg) > 0:
            arrs['T_amplitude'][i]   = float(t_seg[np.argmax(np.abs(t_seg))])
            half_t = 0.5 * np.max(np.abs(t_seg))
            arrs['T_duration_ms'][i] = np.sum(np.abs(t_seg) > half_t) / fs * 1000
        arrs['amplitude_range'][i]  = np.max(seg) - np.min(seg)
        arrs['peak_to_peak'][i]     = float(np.ptp(seg))
        arrs['signal_integral'][i]  = float(np.trapz(np.abs(seg)))
    out = {'valid_beats': valid, 'total_beats': n}
    for k, arr in arrs.items():
        v = arr[~np.isnan(arr)]
        out[f'{k}_mean'] = float(np.mean(v))        if len(v)>0 else np.nan
        out[f'{k}_std']  = float(np.std(v, ddof=1)) if len(v)>1 else np.nan
    return out


def interval_features(signal, r_locs, waves, fs):
    """
    P-wave, PR, QT/QTc, RP interval, VA interval, Q onset, T offset,
    R/T axis proxy, RP/PR ratio. ddof=1 on all variance/std.
    """
    nan_keys = [
        'P_duration_mean_ms','P_duration_std_ms',
        'P_amplitude_mean','P_amplitude_std','P_amplitude_variance',
        'P_wave_presence_ratio','P_wave_absent_flag',
        'PR_interval_mean_ms','PR_interval_std_ms',
        'QT_interval_mean_ms','QTc_interval_mean_ms','QT_interval_std_ms',
        'Q_onset_time_ms','T_offset_time_ms',
        'RP_interval_mean_ms','RP_interval_std_ms','VA_interval_mean_ms',
        'R_axis_proxy','T_axis_proxy','RP_PR_ratio'
    ]
    if waves is None:
        return {k: np.nan for k in nan_keys}
    def get_arr(key):
        vals = waves.get(key, [])
        return np.array([
            v if v is not None and not (isinstance(v,float) and np.isnan(v))
            else np.nan for v in vals], dtype=float)
    p_peaks  = get_arr('ECG_P_Peaks')
    p_onsets = get_arr('ECG_P_Onsets')
    p_offsets= get_arr('ECG_P_Offsets')
    q_peaks  = get_arr('ECG_Q_Peaks')
    t_offsets= get_arr('ECG_T_Offsets')
    t_peaks  = get_arr('ECG_T_Peaks')
    n_beats  = len(r_locs)
    feats    = {}
    P_present, P_dur, P_amp = [], [], []
    for i in range(min(n_beats, len(p_peaks))):
        pp = p_peaks[i]
        if np.isnan(pp):
            P_present.append(0); continue
        pp = int(pp)
        P_present.append(1)
        P_amp.append(float(signal[pp]) if pp < len(signal) else np.nan)
        po  = p_onsets[i]  if i < len(p_onsets)  else np.nan
        pof = p_offsets[i] if i < len(p_offsets) else np.nan
        if not np.isnan(po) and not np.isnan(pof):
            P_dur.append((pof - po) / fs * 1000)
    feats['P_wave_presence_ratio'] = float(np.mean(P_present))          if P_present     else np.nan
    feats['P_wave_absent_flag']    = 1.0 if feats['P_wave_presence_ratio'] < 0.5 else 0.0
    feats['P_duration_mean_ms']    = float(np.nanmean(P_dur))            if P_dur         else np.nan
    feats['P_duration_std_ms']     = float(np.nanstd(P_dur,  ddof=1))   if len(P_dur)>1  else np.nan
    feats['P_amplitude_mean']      = float(np.nanmean(P_amp))            if P_amp         else np.nan
    feats['P_amplitude_std']       = float(np.nanstd(P_amp,  ddof=1))   if len(P_amp)>1  else np.nan
    feats['P_amplitude_variance']  = float(np.nanvar(P_amp,  ddof=1))   if len(P_amp)>1  else np.nan
    PR = []
    for i in range(min(n_beats, len(p_onsets))):
        po = p_onsets[i]
        if not np.isnan(po):
            PR.append((r_locs[i] - po) / fs * 1000)
    feats['PR_interval_mean_ms'] = float(np.nanmean(PR))         if PR        else np.nan
    feats['PR_interval_std_ms']  = float(np.nanstd(PR, ddof=1))  if len(PR)>1 else np.nan
    QT, QTc = [], []
    RR_s = np.diff(r_locs) / fs
    for i in range(min(n_beats, len(q_peaks), len(t_offsets))):
        qp = q_peaks[i];  tof = t_offsets[i]
        if np.isnan(qp) or np.isnan(tof): continue
        qt_ms = (int(tof) - int(qp)) / fs * 1000
        if qt_ms > 0:
            QT.append(qt_ms)
            if i < len(RR_s) and RR_s[i] > 0:
                QTc.append(qt_ms / np.sqrt(RR_s[i]))
    feats['QT_interval_mean_ms']  = float(np.nanmean(QT))         if QT        else np.nan
    feats['QT_interval_std_ms']   = float(np.nanstd(QT, ddof=1))  if len(QT)>1 else np.nan
    feats['QTc_interval_mean_ms'] = float(np.nanmean(QTc))         if QTc       else np.nan
    q_valid = q_peaks[~np.isnan(q_peaks)]
    t_valid = t_offsets[~np.isnan(t_offsets)]
    feats['Q_onset_time_ms']  = float(np.mean(q_valid)/fs*1000) if len(q_valid) else np.nan
    feats['T_offset_time_ms'] = float(np.mean(t_valid)/fs*1000) if len(t_valid) else np.nan
    RP, VA = [], []
    rp_win = int(0.20 * fs)
    for r in r_locs:
        p_after = [p_peaks[j] for j in range(len(p_peaks))
                   if not np.isnan(p_peaks[j])
                   and p_peaks[j] > r
                   and p_peaks[j] <= r + rp_win]
        if p_after:
            rp_ms = (p_after[0] - r) / fs * 1000
            RP.append(rp_ms); VA.append(rp_ms)
    feats['RP_interval_mean_ms'] = float(np.nanmean(RP))         if RP        else np.nan
    feats['RP_interval_std_ms']  = float(np.nanstd(RP, ddof=1))  if len(RP)>1 else np.nan
    feats['VA_interval_mean_ms'] = float(np.nanmean(VA))         if VA        else np.nan
    r_amps = [signal[r] for r in r_locs if r < len(signal)]
    t_amps = [signal[int(t)] for t in t_peaks
              if not np.isnan(t) and int(t) < len(signal)]
    feats['R_axis_proxy'] = float(np.nanmean(r_amps)) if r_amps else np.nan
    feats['T_axis_proxy'] = float(np.nanmean(t_amps)) if t_amps else np.nan
    pr = feats.get('PR_interval_mean_ms', np.nan)
    rp = feats.get('RP_interval_mean_ms', np.nan)
    feats['RP_PR_ratio'] = rp / pr if (pr and pr > 0) else np.nan
    return feats


def spectral_features(signal, fs):
    """Welch PSD spectral features on bandpass filtered signal."""
    from scipy.signal import welch
    nfft    = int(2**np.ceil(np.log2(len(signal))))
    nperseg = min(nfft, len(signal))
    f, pxx  = welch(signal, fs=fs, nperseg=nperseg, nfft=nfft)
    def band_power(flo, fhi):
        idx = (f >= flo) & (f <= fhi)
        return float(np.trapz(pxx[idx], f[idx])) if np.sum(idx)>=2 else np.nan
    total  = float(np.trapz(pxx, f))
    pk_idx = int(np.argmax(pxx))
    return {
        'P_wave_band_power'    : band_power(0.67, 5.0),
        'T_wave_band_power'    : band_power(1.0,  7.0),
        'QRS_band_power'       : band_power(10.0, 50.0),
        'total_spectral_energy': total,
        'peak_frequency_Hz'    : float(f[pk_idx]),
        'max_power'            : float(pxx[pk_idx]),
        'centroid_freq_Hz'     : float(np.trapz(f*pxx,f)/total) if total>0 else np.nan,
        'spectral_flatness'    : float(np.exp(np.mean(np.log(pxx+1e-12)))/np.mean(pxx))
                                  if np.mean(pxx)>0 else np.nan,
        'AMSA'                 : band_power(0.5, 150.0),
    }


def qrs_alternans(r_locs, signal):
    """Beat-to-beat QRS amplitude variation. ddof=1 matches MATLAB."""
    if len(r_locs) < 2:
        return {'QRS_alternans_std': np.nan, 'QRS_alternans_flag': np.nan}
    amps = signal[r_locs]
    std  = float(np.std(amps, ddof=1))
    flag = 1.0 if std > 0.1 * np.abs(np.mean(amps)) else 0.0
    return {'QRS_alternans_std': std, 'QRS_alternans_flag': flag}


def nonlinear_features(signal, RR_ms):
    """Poincaré SD1/SD2, Higuchi FD, ApEn, SampEn, Lyapunov exponent."""
    feats = {}
    if len(RR_ms) >= 3:
        dRR     = np.diff(RR_ms)
        SD1     = float(np.std(dRR, ddof=1) / np.sqrt(2))
        var_SD2 = 2*np.var(RR_ms, ddof=1) - np.var(dRR, ddof=1)/2
        SD2     = float(np.sqrt(var_SD2)) if var_SD2 >= 0 else np.nan
        feats['SD1_ms']        = SD1
        feats['SD2_ms']        = SD2
        feats['SD1_SD2_ratio'] = SD1/SD2 if (SD2 and SD2>0) else np.nan
    else:
        feats.update({'SD1_ms':np.nan,'SD2_ms':np.nan,'SD1_SD2_ratio':np.nan})
    def higuchi_fd(x, kmax=50):
        N = len(x); L = []
        for k in range(1, kmax+1):
            Lk = []
            for m in range(1, k+1):
                idxs = np.arange(m-1, N, k)
                if len(idxs)<2: continue
                Lmk = np.sum(np.abs(np.diff(x[idxs])))*(N-1)/(k*len(idxs))
                Lk.append(Lmk)
            if Lk: L.append(np.log(np.mean(Lk)))
        if len(L)<2: return np.nan
        slope, _ = np.polyfit(np.log(np.arange(1,len(L)+1)), L, 1)
        return float(slope)
    feats['fractal_dimension'] = higuchi_fd(signal[:10000])
    def approximate_entropy(U, m=2, r_factor=0.2):
        N = len(U); r = r_factor*np.std(U, ddof=1)
        if r==0 or N<m+2: return np.nan
        def phi(m_):
            x = np.array([U[i:i+m_] for i in range(N-m_+1)])
            C = np.sum(np.max(np.abs(x[:,None]-x[None,:]),axis=2)<=r,axis=0)/(N-m_+1)
            return np.sum(np.log(C+1e-12))/(N-m_+1)
        return float(phi(m)-phi(m+1))
    feats['approximate_entropy'] = approximate_entropy(signal[::20][:500])
    def sample_entropy(U, m=2, r_factor=0.2):
        N = len(U); r = r_factor*np.std(U, ddof=1)
        if r==0 or N<m+2: return np.nan
        def count(m_):
            c = 0
            for i in range(N-m_):
                for j in range(i+1, N-m_):
                    if np.max(np.abs(U[i:i+m_]-U[j:j+m_]))<=r: c+=1
            return c
        B=count(m); A=count(m+1)
        return float(-np.log(A/B)) if B>0 and A>0 else np.nan
    feats['sample_entropy'] = sample_entropy(signal[::20][:300])
    if len(RR_ms) >= 10:
        try:
            x = RR_ms - np.mean(RR_ms)
            N = len(x); emb=3; lag=1
            M = N-(emb-1)*lag
            Y = np.array([x[i:i+emb*lag:lag] for i in range(M)])
            divergence = []
            for i in range(M):
                d = np.max(np.abs(Y-Y[i]),axis=1)
                d[max(0,i-2):min(M,i+3)] = np.inf
                j = np.argmin(d)
                if np.isfinite(d[j]) and d[j]>0:
                    divergence.append(np.log(d[j]))
            feats['lyapunov_exponent'] = float(np.mean(divergence)) if divergence else np.nan
        except Exception:
            feats['lyapunov_exponent'] = np.nan
    else:
        feats['lyapunov_exponent'] = np.nan
    return feats


def wavelet_features_fn(signal_raw, fs):
    """DWT energy L1–L4 using db4 on high-pass filtered signal."""
    if not PWT_AVAILABLE:
        return {f'DWT_energy_L{i}': np.nan for i in range(1, 5)}
    sos_hp    = butter(4, 0.5, btype='high', fs=fs, output='sos')
    signal_hp = sosfiltfilt(sos_hp, signal_raw)
    coeffs = pywt.wavedec(signal_hp, 'db4', level=4)
    feats  = {}
    for lvl in range(1, 5):
        detail = coeffs[5-lvl]
        feats[f'DWT_energy_L{lvl}'] = float(np.sum(detail**2))
    return feats


def extract_all_features(signal_filt, signal_raw, r_locs, waves, fs, ch_name):
    """Master function — all 18 feature categories for one channel."""
    all_feats = {'channel': ch_name}
    rr_feats, RR_ms = rr_features(r_locs, fs)
    all_feats.update(rr_feats)
    all_feats.update(morph_features(signal_filt, r_locs, fs))
    all_feats.update(interval_features(signal_filt, r_locs, waves, fs))
    all_feats.update(qrs_alternans(r_locs, signal_filt))
    all_feats.update(spectral_features(signal_filt, fs))
    if len(RR_ms) >= 3:
        all_feats.update(nonlinear_features(signal_filt, RR_ms))
    all_feats.update(wavelet_features_fn(signal_raw, fs))
    if len(r_locs) > 0:
        r_amps = signal_filt[r_locs]
        all_feats.update({
            'R_peak_amplitude_mean': float(np.mean(r_amps)),
            'R_peak_amplitude_std' : float(np.std(r_amps, ddof=1)),
            'R_peak_amplitude_max' : float(np.max(r_amps)),
            'R_peak_count'         : int(len(r_locs)),
        })
    return all_feats, RR_ms

# =============================================================================
#  SECTION D — SIGNAL PROCESSING AND FEATURE EXTRACTION FOR ALL AVNRT DATASETS
# =============================================================================

def process_one_file(mat_path, label):
    """
    Runs full signal processing pipeline on one .mat file.
    Returns one ML-ready row dict, or None if file cannot be processed.
    """
    print(f"\n{'='*65}")
    print(f"  FILE  : {mat_path}")
    print(f"  LABEL : {label}")
    print(f"{'='*65}")

    # Load
    try:
        mat_data = scipy.io.loadmat(mat_path)
    except Exception as e:
        print(f"  ERROR loading: {e} — skipping.")
        return None

    data = {k: v for k, v in mat_data.items() if not k.startswith('__')}
    try:
        fs      = float(data['fs'].squeeze())
        time    = data['time'].squeeze()
        ch1_acq = data['ch1_acq'].squeeze().astype(float)
        ch3_acq = data['ch3_acq'].squeeze().astype(float)
        ch4_acq = data['ch4_acq'].squeeze().astype(float)
    except KeyError as e:
        print(f"  ERROR: missing variable {e} — skipping.")
        return None

    print(f"  fs={fs:.0f}Hz | duration={time[-1]:.2f}s | samples={len(time)}")

    # Plot raw signals
    base_name  = os.path.splitext(os.path.basename(mat_path))[0]
    out_folder = os.path.join('outputs', base_name)
    os.makedirs(out_folder, exist_ok=True)

    fig, axes = plt.subplots(3, 1, figsize=(15, 7), sharex=True)
    fig.suptitle(f"Raw Signals — {label} — {base_name}", fontsize=11)
    for ax, ch, lbl, col in zip(axes,
            [ch1_acq, ch3_acq, ch4_acq],
            ['Channel 1 (Raw)', 'Channel 3 (Raw)', 'Channel 4 (Raw)'],
            ['royalblue', 'darkorange', 'green']):
        ax.plot(time, ch, color=col, linewidth=0.6)
        ax.set_title(lbl); ax.set_ylabel('Amplitude'); ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, 'raw_signals.png'), dpi=120)
    plt.close()

    # Bandpass filter
    try:
        ch1_filt = bandpass_filter(ch1_acq, fs)
        ch3_filt = bandpass_filter(ch3_acq, fs)
        ch4_filt = bandpass_filter(ch4_acq, fs)
    except Exception as e:
        print(f"  ERROR filtering: {e} — skipping.")
        return None

    if not (np.all(np.isfinite(ch3_filt)) and np.all(np.isfinite(ch4_filt))):
        print("  ERROR: filter produced Inf/NaN — skipping.")
        return None

    # R-peak detection
    MIN_HEIGHT   = 0.5
    MIN_DIST_PTS = int(0.20 * fs)
    r_locs3, _ = find_peaks(ch3_filt, height=MIN_HEIGHT, distance=MIN_DIST_PTS)
    r_locs4, _ = find_peaks(ch4_filt, height=MIN_HEIGHT, distance=MIN_DIST_PTS)
    print(f"  R-peaks — Ch3: {len(r_locs3)}  |  Ch4: {len(r_locs4)}")

    # Data quality warning: doubled RR means false peak from baseline wander
    for ch_name, locs in [('Ch3', r_locs3), ('Ch4', r_locs4)]:
        if len(locs) >= 2:
            rr = np.diff(locs) / fs * 1000
            if np.max(rr) > 2 * np.median(rr):
                print(f"  WARNING: {ch_name} doubled RR detected "
                      f"(max={np.max(rr):.0f}ms vs median={np.median(rr):.0f}ms) "
                      f"— baseline wander likely. Ch{ch_name[-1]} features may be unreliable.")

    # NeuroKit2 delineation
    print("  Running NK2 delineation...")
    waves_ch3 = run_delineation(ch3_filt, r_locs3, fs, 'Ch3')
    waves_ch4 = run_delineation(ch4_filt, r_locs4, fs, 'Ch4')

    # Feature extraction
    print("  Extracting features...")
    feats_ch3, RR_ms3 = extract_all_features(ch3_filt, ch3_acq, r_locs3, waves_ch3, fs, 'Channel_3')
    feats_ch4, RR_ms4 = extract_all_features(ch4_filt, ch4_acq, r_locs4, waves_ch4, fs, 'Channel_4')

    # Per-file feature summary printout
    print(f"\n  {'─'*61}")
    print(f"  {'Feature':<40} {'Ch3':>10} {'Ch4':>10}")
    print(f"  {'─'*61}")
    summary_keys = [
        'RR_mean_ms','RR_std_ms','HR_mean_bpm','RMSSD_ms','pNN50_pct',
        'QRS_amplitude_mean','QRS_duration_ms_mean','RS_interval_ms_mean',
        'beat_skewness_mean','beat_kurtosis_mean',
        'P_wave_presence_ratio','P_wave_absent_flag','P_duration_mean_ms',
        'PR_interval_mean_ms','QT_interval_mean_ms',
        'RP_interval_mean_ms','VA_interval_mean_ms','RP_PR_ratio',
        'ST_slope_mean',
        'QRS_band_power','P_wave_band_power','peak_frequency_Hz','spectral_flatness',
        'SD1_ms','SD2_ms','SD1_SD2_ratio',
        'R_peak_amplitude_mean','R_peak_count',
    ]
    for key in summary_keys:
        v3 = feats_ch3.get(key, np.nan)
        v4 = feats_ch4.get(key, np.nan)
        try:
            print(f"  {key:<40} {float(v3):>10.4f} {float(v4):>10.4f}")
        except (TypeError, ValueError):
            print(f"  {key:<40} {str(v3):>10} {str(v4):>10}")

    # Poincaré plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Poincaré — {label} — {base_name}", fontsize=10)
    for ax, RR_ms, lbl, col in zip(axes, [RR_ms3, RR_ms4],
                                    ['Ch3', 'Ch4'], ['darkorange', 'green']):
        if len(RR_ms) >= 2:
            ax.scatter(RR_ms[:-1], RR_ms[1:], alpha=0.7, color=col, s=30)
        ax.set_xlabel('RR_n (ms)'); ax.set_ylabel('RR_{n+1} (ms)')
        ax.set_title(f'Poincaré — {lbl}')
        ax.grid(True, alpha=0.3); ax.set_aspect('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(out_folder, 'poincare.png'), dpi=120)
    plt.close()

    # Build ML row — Ch3 and Ch4 features with label
    ml_row_ch3 = {f'ch3_{k}': feats_ch3.get(k, np.nan) for k in ML_FEATURES}
    ml_row_ch4 = {f'ch4_{k}': feats_ch4.get(k, np.nan) for k in ML_FEATURES}
    ml_row = {
        'file'     : mat_path,
        'label'    : label,
        'label_num': 0 if label == 'AVNRT' else 1,
        **ml_row_ch3,
        **ml_row_ch4,
    }
    print(f"\n  ML feature vector saved — {len(ML_FEATURES)*2} features total")
    return ml_row


# =============================================================================
#  SIGNAL PROCESSING AND FEATURE EXTRACTION FOR ALL AVNRT DATASETS
# =============================================================================
print("\n" + "="*65)
print("SIGNAL PROCESSING AND FEATURE EXTRACTION — ALL AVNRT DATASETS")
print("="*65)

all_ml_rows = []
failed_files = []

for mat_path in avnrt_datasets: # Process amd iterate over all AVNRT .mat files
    row = process_one_file(mat_path, label='AVNRT')
    if row is not None:
        all_ml_rows.append(row)
    else:
        failed_files.append(mat_path)

# =============================================================================
#  SIGNAL PROCESSING AND FEATURE EXTRACTION FOR ALL AVRT DATASETS
# =============================================================================
print("\n" + "="*65)
print("SIGNAL PROCESSING AND FEATURE EXTRACTION — ALL AVRT DATASETS")
print("="*65)

for mat_path in avrt_datasets: # Process and iterate over all AVRT .mat files
    row = process_one_file(mat_path, label='AVRT')
    if row is not None:
        all_ml_rows.append(row)
    else:
        failed_files.append(mat_path)

# =============================================================================
#  SECTION E — COMBINED SUMMARY AND EXCEL EXPORT
# =============================================================================
print("\n" + "="*65)
print(f"{'PROCESSING COMPLETE':^65}")
print("="*65)
print(f"Successfully processed : {len(all_ml_rows)} recordings")
print(f"Failed / skipped       : {len(failed_files)} recordings")
if failed_files:
    for f in failed_files:
        print(f"  FAILED: {f}")

if all_ml_rows:
    df_ml = pd.DataFrame(all_ml_rows)

    # Combined feature comparison table (Ch3, key features)
    print("\n" + "="*65)
    print("COMBINED FEATURE COMPARISON — Ch3 (mean ± std across recordings)")
    print("="*65)
    # key_features = [
    #     'HR_mean_bpm', 'RR_mean_ms', 'RR_std_ms', 'RMSSD_ms',
    #     'RS_interval_ms_mean', 'QRS_band_power', 'P_wave_presence_ratio',
    #     'SD1_ms', 'SD1_SD2_ratio', 'RP_interval_mean_ms', 'RP_PR_ratio',
    #     'PR_interval_mean_ms', 'QT_interval_mean_ms', 'QRS_duration_ms_mean',
    #     'beat_skewness_mean', 'beat_kurtosis_mean', 'ST_slope_mean',
    #     'P_wave_absent_flag', 'peak_frequency_Hz', 'R_peak_amplitude_mean',
    # ]

    key_features = ML_FEATURES

    print(f"\n  {'Feature':<35} {'AVNRT mean':>12} {'AVNRT std':>10} {'AVRT mean':>12} {'AVRT std':>10}")
    print(f"  {'─'*71}")
    for feat in key_features:
        col = f'ch3_{feat}'
        if col not in df_ml.columns:
            continue
        avnrt_vals = df_ml[df_ml['label']=='AVNRT'][col].dropna()
        avrt_vals  = df_ml[df_ml['label']=='AVRT'][col].dropna()
        avnrt_m = avnrt_vals.mean() if len(avnrt_vals)>0 else float('nan')
        avnrt_s = avnrt_vals.std()  if len(avnrt_vals)>1 else float('nan')
        avrt_m  = avrt_vals.mean()  if len(avrt_vals)>0  else float('nan')
        avrt_s  = avrt_vals.std()   if len(avrt_vals)>1  else float('nan')
        print(f"  {feat:<35} {avnrt_m:>12.4f} {avnrt_s:>10.4f} {avrt_m:>12.4f} {avrt_s:>10.4f}")

    print(f"\n  AVNRT recordings: {len(df_ml[df_ml['label']=='AVNRT'])}")
    print(f"  AVRT  recordings: {len(df_ml[df_ml['label']=='AVRT'])}")

    # Save to Excel
    out_ml = 'ECG_ALL_ML_features.xlsx'
    df_ml.to_excel(out_ml, index=False)
    print(f"\nAll ML features saved to: {out_ml}")
    print(f"Rows (recordings) : {len(df_ml)}")
    print(f"Columns (features): {len(df_ml.columns)}")
    print(f"\nThis file is ready to load into the ML model (logistic regression,")
    print(f"random forest, XGBoost). Each row = one recording. Label column = 'label_num'")
    print(f"  0 = AVNRT  |  1 = AVRT")

# =============================================================================
#  SECTION F — REVISED ML FEATURES
# =============================================================================

ML_FEATURES_FINAL = [
    # Tier 1 — strongest separators
    'pNN50_pct',
    'HR_std_bpm',
    'beat_skewness_mean',
    'beat_kurtosis_mean',
    'QRS_duration_ms_mean',
    'P_wave_presence_ratio',
    'ST_slope_mean',
    'R_peak_count',

    # Tier 2 — strong but handle NaN
    'RP_interval_mean_ms',
    'RP_PR_ratio',

    # Tier 3 — moderate, statistically significant
    'SD2_ms', 'SD1_ms', 'SD1_SD2_ratio',
    'RR_std_ms', 'RMSSD_ms', 'RR_mean_ms',
    'P_wave_absent_flag',
]

# =============================================================================
#  SECTION G — MACHINE LEARNING MODEL — LOGISTIC REGRESSION WHERE AVNRT IS PREDICTED AS 0 AND AVRT AS 1
# =============================================================================
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,
                             recall_score, f1_score, roc_auc_score,
                             ConfusionMatrixDisplay)
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# =============================================================================
# 1. LOAD DATA
# =============================================================================
# data = pd.read_excel('ECG_ALL_ML_features.xlsx')
out_ml = 'ECG_ALL_ML_features.xlsx'
data = pd.read_excel(out_ml)  # Load the combined ML features Excel file
print(f"Dataset loaded: {data.shape[0]} recordings, {data.shape[1]} columns")
print(f"Label distribution:\n{data['label'].value_counts()}\n")

# =============================================================================
# 2. SELECT FEATURES — only the 17 best features from your analysis
#    Drop 'file', 'label', 'label_num' and any non-numeric columns
# =============================================================================
ML_FEATURES_FINAL_CH3 = [
    # Tier 1 — strongest separators (low NaN, significant p-value)
    'ch3_pNN50_pct',
    'ch3_HR_std_bpm',
    'ch3_beat_skewness_mean',
    'ch3_beat_kurtosis_mean',
    'ch3_QRS_duration_ms_mean',
    'ch3_P_wave_presence_ratio',
    'ch3_ST_slope_mean',
    'ch3_R_peak_count',

    # Tier 2 — strong effect but handle NaN with imputation
    'ch3_RP_interval_mean_ms',
    'ch3_RP_PR_ratio',

    # Tier 3 — moderate, statistically significant
    'ch3_SD2_ms',
    'ch3_SD1_ms',
    'ch3_SD1_SD2_ratio',
    'ch3_RR_std_ms',
    'ch3_RMSSD_ms',
    'ch3_RR_mean_ms',
    'ch3_P_wave_absent_flag',
]

ML_FEATURES_FINAL_CH4 = [
    # Tier 1 — strongest separators (low NaN, significant p-value)
    'ch4_pNN50_pct',
    'ch4_HR_std_bpm',
    'ch4_beat_skewness_mean',
    'ch4_beat_kurtosis_mean',
    'ch4_QRS_duration_ms_mean',
    'ch4_P_wave_presence_ratio',
    'ch4_ST_slope_mean',
    'ch4_R_peak_count',

    # Tier 2 — strong effect but handle NaN with imputation
    'ch4_RP_interval_mean_ms',
    'ch4_RP_PR_ratio',

    # Tier 3 — moderate, statistically significant
    'ch4_SD2_ms',
    'ch4_SD1_ms',
    'ch4_SD1_SD2_ratio',
    'ch4_RR_std_ms',
    'ch4_RMSSD_ms',
    'ch4_RR_mean_ms',
    'ch4_P_wave_absent_flag',
]

ML_FEATURES_FINAL_CH4_SIGNIFICANT = [
    'ch4_SD1_SD2_ratio',
    'ch4_P_wave_absent_flag',
]

COMBINED_FEATURES = ML_FEATURES_FINAL_CH3 + ML_FEATURES_FINAL_CH4_SIGNIFICANT

X = data[COMBINED_FEATURES]   # Only numeric feature columns
y = data['label_num']          # 0 = AVNRT, 1 = AVRT

print(f"Features selected: {len(COMBINED_FEATURES)}")
print(f"NaN counts per feature:")
print(X.isna().sum()[X.isna().sum() > 0])

# =============================================================================
# 3. HANDLE MISSING VALUES (NaN imputation)
#    RP_interval and RP_PR_ratio have ~30% NaN — fill with median
#    Median imputation is the correct approach for small medical datasets
# =============================================================================
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(
    imputer.fit_transform(X),
    columns=COMBINED_FEATURES
)
print(f"\nAfter imputation — NaN count: {X_imputed.isna().sum().sum()}")

# =============================================================================
# 4. SCALE FEATURES
#    Logistic Regression requires feature scaling — StandardScaler
#    (zero mean, unit variance). Random Forest and XGBoost do NOT need this
#    but it does not hurt to include it for consistency.
# =============================================================================
scaler = StandardScaler()

# =============================================================================
# 5. TRAIN / TEST SPLIT
#    stratify=y ensures both train and test sets have the same AVNRT/AVRT ratio
#    With only 46 recordings, this is critical to avoid imbalanced splits
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y,
    test_size=0.2,
    random_state=42,
    stratify=y        # preserves AVNRT/AVRT ratio in both sets
)

# Fit scaler on training data only, then apply to both train and test
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f"\nTrain set: {X_train_scaled.shape[0]} recordings")
print(f"Test set : {X_test_scaled.shape[0]} recordings")
print(f"Train label distribution: AVNRT={sum(y_train==0)}, AVRT={sum(y_train==1)}")
print(f"Test  label distribution: AVNRT={sum(y_test==0)},  AVRT={sum(y_test==1)}")

# =============================================================================
# 6. HYPERPARAMETER TUNING WITH STRATIFIED CROSS-VALIDATION
#    StratifiedKFold preserves class balance in each fold
#    With 46 samples, cv=5 gives ~37 train / ~9 test per fold
#    solver='saga' supports both L1 and L2 penalties
# =============================================================================
param_grid = {
    'C'      : [0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver' : ['saga'],             # saga supports both l1 and l2
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # CROSS FOLD VALIDATION WITH STRATIFICATION OF 5

grid_search = GridSearchCV(
    estimator=LogisticRegression(max_iter=2000, random_state=42),
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0,
)
grid_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters : {grid_search.best_params_}")
print(f"Best Cross Validation (CV) accuracy for Logistic Regression : {grid_search.best_score_:.4f}")

# =============================================================================
# 7. TRAIN BEST MODEL AND EVALUATE ON TEST SET
# =============================================================================
best_lr = LogisticRegression(
    **grid_search.best_params_,
    max_iter=2000,
    random_state=42
)
best_lr.fit(X_train_scaled, y_train)

y_pred     = best_lr.predict(X_test_scaled)
y_pred_prob= best_lr.predict_proba(X_test_scaled)[:, 1]  # probability of AVRT

print("\n" + "="*50)
print("LOGISTIC REGRESSION — TEST SET RESULTS")
print("="*50)
print(f"Accuracy  : {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision : {precision_score(y_test, y_pred):.4f}")
print(f"Recall    : {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score  : {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC   : {roc_auc_score(y_test, y_pred_prob):.4f}")
print(f"Test set label distribution: AVNRT={sum(y_test==0)}, AVRT={sum(y_test==1)}")
print("Correct predictions:", sum(y_test == y_pred), "out of", len(y_test))

# =============================================================================
# 8. CONFUSION MATRIX PLOT
# =============================================================================
fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['AVNRT (0)', 'AVRT (1)']
)
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Logistic Regression — Confusion Matrix')
plt.tight_layout()
plt.savefig('LR_confusion_matrix.png', dpi=150)
plt.show()
print("Saved: LR_confusion_matrix.png")

# =============================================================================
# 9. FEATURE IMPORTANCE — LOGISTIC REGRESSION COEFFICIENTS
#    Absolute coefficient value = importance of each feature
# =============================================================================
coef_df = pd.DataFrame({
    'feature'    : COMBINED_FEATURES,
    'coefficient': best_lr.coef_[0],
    'abs_coef'   : np.abs(best_lr.coef_[0])
}).sort_values('abs_coef', ascending=False)

print("\nFeature importance (by absolute coefficient):")
print(coef_df[['feature','coefficient']].to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 6))
colors = ['tomato' if c > 0 else 'steelblue' for c in coef_df['coefficient']]
ax.barh(coef_df['feature'], coef_df['coefficient'], color=colors)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Coefficient (positive = predicts AVRT)')
ax.set_title('Logistic Regression — Feature Coefficients')
ax.invert_yaxis()
plt.tight_layout()
plt.savefig('LR_feature_coefficients.png', dpi=150)
plt.show()
print("Saved: LR_feature_coefficients.png")

# =============================================================================
# 10. LEAVE-ONE-OUT CROSS-VALIDATION
#     With only 46 recordings, LOO-CV gives the most reliable accuracy estimate
#     since each fold uses 45 recordings for training and 1 for testing
# =============================================================================
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.pipeline import Pipeline

loo_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',  StandardScaler()),
    ('lr',      LogisticRegression(
                    **grid_search.best_params_,
                    max_iter=2000,
                    random_state=42))
])

loo = LeaveOneOut()
loo_scores = cross_val_score(loo_pipeline, X, y, cv=loo, scoring='accuracy')

print(f"\nLeave-One-Out CV accuracy for Logistic Regression: {loo_scores.mean():.4f} ± {loo_scores.std():.4f}")
print(f"  ({sum(loo_scores)} correct out of {len(loo_scores)} recordings)")
print("\nNOTE: LOO-CV accuracy is the most reliable metric for n=46 datasets.")
print("      The 80/20 test set result above may vary due to small sample size.")

# -----------------------------------------------------------------------------------------------
# =============================================================================
#  SECTION H — MACHINE LEARNING MODEL — RANDOM FOREST CLASSIFIER WHERE AVNRT IS PREDICTED AS 0 AND AVRT AS 1
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, LeaveOneOut, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# =============================================================================
# 1. LOAD DATA & SELECT FEATURES
# =============================================================================
# data = pd.read_excel('ECG_ALL_ML_features.xlsx')
out_ml = 'ECG_ALL_ML_features.xlsx'
data = pd.read_excel(out_ml)  # Load the combined ML features Excel file

ML_FEATURES_FINAL_CH3 = [
    # Tier 1 — strongest separators (low NaN, significant p-value)
    'ch3_pNN50_pct',
    'ch3_HR_std_bpm',
    'ch3_beat_skewness_mean',
    'ch3_beat_kurtosis_mean',
    'ch3_QRS_duration_ms_mean',
    'ch3_P_wave_presence_ratio',
    'ch3_ST_slope_mean',
    'ch3_R_peak_count',

    # Tier 2 — strong effect but handle NaN with imputation
    'ch3_RP_interval_mean_ms',
    'ch3_RP_PR_ratio',

    # Tier 3 — moderate, statistically significant
    'ch3_SD2_ms',
    'ch3_SD1_ms',
    'ch3_SD1_SD2_ratio',
    'ch3_RR_std_ms',
    'ch3_RMSSD_ms',
    'ch3_RR_mean_ms',
    'ch3_P_wave_absent_flag',
]

ML_FEATURES_FINAL_CH4 = [
    # Tier 1 — strongest separators (low NaN, significant p-value)
    'ch4_pNN50_pct',
    'ch4_HR_std_bpm',
    'ch4_beat_skewness_mean',
    'ch4_beat_kurtosis_mean',
    'ch4_QRS_duration_ms_mean',
    'ch4_P_wave_presence_ratio',
    'ch4_ST_slope_mean',
    'ch4_R_peak_count',

    # Tier 2 — strong effect but handle NaN with imputation
    'ch4_RP_interval_mean_ms',
    'ch4_RP_PR_ratio',

    # Tier 3 — moderate, statistically significant
    'ch4_SD2_ms',
    'ch4_SD1_ms',
    'ch4_SD1_SD2_ratio',
    'ch4_RR_std_ms',
    'ch4_RMSSD_ms',
    'ch4_RR_mean_ms',
    'ch4_P_wave_absent_flag',
]

ML_FEATURES_FINAL_CH4_SIGNIFICANT = [
    'ch4_SD1_SD2_ratio',
    'ch4_P_wave_absent_flag',
]

COMBINED_FEATURES = ML_FEATURES_FINAL_CH3 + ML_FEATURES_FINAL_CH4_SIGNIFICANT

X = data[COMBINED_FEATURES]
y = data['label_num']

# =============================================================================
# 2. TRAIN / TEST SPLIT
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================================================================
# 3. BUILD PIPELINE & TUNING
# =============================================================================
random_forest_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('rf', RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1))
])

random_forest_param_grid = {
    'rf__n_estimators' : [50, 100, 200],
    'rf__max_depth'    : [3, 5, None],
    'rf__min_samples_split': [2, 5],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) # Cross-validation strategy with stratification of 5

random_forest_grid = GridSearchCV(
    random_forest_pipeline,
    random_forest_param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
)

random_forest_grid.fit(X_train, y_train)
print(f"Random Forest (RF) best parameters: {random_forest_grid.best_params_}")
print(f"Best Cross Validation (CV) accuracy for Random Forest : {random_forest_grid.best_score_:.4f}")

# =============================================================================
# 4. EVALUATE ON TEST SET
# =============================================================================
best_random_forest_model = random_forest_grid.best_estimator_
y_pred_random_forest = best_random_forest_model.predict(X_test)

print("\n" + "="*50)
print("RANDOM FOREST — TEST SET RESULTS")
print("="*50)
print(f"Random Forest test accuracy: {accuracy_score(y_test, y_pred_random_forest):.4f}")
print(f"Random Forest precision: {precision_score(y_test, y_pred_random_forest):.4f}")
print(f"Random Forest recall: {recall_score(y_test, y_pred_random_forest):.4f}")
print(f"Random Forest F1 score: {f1_score(y_test, y_pred_random_forest):.4f}")

# Plot Confusion Matrix
confusion_random_forest = confusion_matrix(y_test, y_pred_random_forest)
ConfusionMatrixDisplay(confusion_matrix=confusion_random_forest, display_labels=['AVNRT (0)', 'AVRT (1)']).plot(cmap='Blues')
plt.title('Random Forest — Confusion Matrix')
plt.show()

# =============================================================================
# 5. LEAVE-ONE-OUT CROSS-VALIDATION (The most reliable metric)
# =============================================================================
# We instantiate a fresh pipeline using the best discovered parameters to evaluate via LOO-CV safely
loocv_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('rf', RandomForestClassifier(
        n_estimators=random_forest_grid.best_params_['rf__n_estimators'],
        max_depth=random_forest_grid.best_params_['rf__max_depth'],
        min_samples_split=random_forest_grid.best_params_['rf__min_samples_split'],
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ))
])

rf_loocv_scores = cross_val_score(loocv_pipeline, X, y, cv=LeaveOneOut(), scoring='accuracy')
print(f"\nRandom Forest (RF) LOO-CV accuracy: {rf_loocv_scores.mean():.4f} ({int(sum(rf_loocv_scores))}/46 recordings)")

# =============================================================================
# 6. FEATURE IMPORTANCE (Fixed array name alignment)
# =============================================================================
rf_internal_model = best_random_forest_model.named_steps['rf']
feat_imp = pd.DataFrame({
    'feature'   : COMBINED_FEATURES, # Fixed name alignment
    'importance': rf_internal_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nRandom Forest feature importances:")
print(feat_imp.to_string(index=False))


# # 3. Initialize and train the model
# # n_estimators is the number of trees in the forest
# model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs = -1)
# model.fit(X_train, y_train)

# param_dist = {
#   'n_estimators': randint(100, 500),
#   'max_depth': randint(3, 15),
#   'min_samples_split': randint(2, 10),
#   'min_samples_leaf': randint(1, 5)
# }


# # Create a random forest classifier
# rf = RandomForestClassifier(random_state=42, n_jobs=-1)

# # Use random search to find the best hyperparameters
# rand_search = RandomizedSearchCV(
#   rf, param_distributions=param_dist,
#   n_iter=10, cv=5, scoring='accuracy',
#   n_jobs=-1, random_state=42)

# # Create a random forest classifier
# rf = RandomForestClassifier(random_state=42, n_jobs=-1)

# # Use random search to find the best hyperparameters
# rand_search = RandomizedSearchCV(
#   rf, param_distributions=param_dist,
#   n_iter=10, cv=5, scoring='accuracy',
#   n_jobs=-1, random_state=42)


# # Create a variable for the best model
# best_rf = rand_search.best_estimator_

# # Print the best hyperparameters
# print('Best hyperparameters:',  rand_search.best_params_)


# # 4. Make predictions
# y_pred = model.predict(X_test)

# # Generate predictions with the best model
# y_pred = best_rf.predict(X_test)

# # Create the confusion matrix
# cm = confusion_matrix(y_test, y_pred)

# ConfusionMatrixDisplay(confusion_matrix=cm).plot();

# # 5. Evaluate
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
# print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")


# y_pred = knn.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)

# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)

# mse = mean_squared_error(y_test, y_pred)
# print("Mean Squared Error:", mse)

# r2 = r2_score(y_test, y_pred)
# print("R-squared:", r2)





# # 3. Initialize and train the model
# # n_estimators is the number of trees in the forest
# model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs = -1)
# model.fit(X_train, y_train)

# param_dist = {
#   'n_estimators': randint(100, 500),
#   'max_depth': randint(3, 15),
#   'min_samples_split': randint(2, 10),
#   'min_samples_leaf': randint(1, 5)
# }


# # Create a random forest classifier
# rf = RandomForestClassifier(random_state=42, n_jobs=-1)

# # Use random search to find the best hyperparameters
# rand_search = RandomizedSearchCV(
#   rf, param_distributions=param_dist,
#   n_iter=10, cv=5, scoring='accuracy',
#   n_jobs=-1, random_state=42)

# # Create a random forest classifier
# rf = RandomForestClassifier(random_state=42, n_jobs=-1)

# # Use random search to find the best hyperparameters
# rand_search = RandomizedSearchCV(
#   rf, param_distributions=param_dist,
#   n_iter=10, cv=5, scoring='accuracy',
#   n_jobs=-1, random_state=42)


# # Create a variable for the best model
# best_rf = rand_search.best_estimator_

# # Print the best hyperparameters
# print('Best hyperparameters:',  rand_search.best_params_)


# # 4. Make predictions
# y_pred = model.predict(X_test)

# # Generate predictions with the best model
# y_pred = best_rf.predict(X_test)

# # Create the confusion matrix
# cm = confusion_matrix(y_test, y_pred)

# ConfusionMatrixDisplay(confusion_matrix=cm).plot();

# # 5. Evaluate
# accuracy = accuracy_score(y_test, y_pred)
# print("Accuracy:", accuracy)
# print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")


# y_pred = knn.predict(X_test)

# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred)
# recall = recall_score(y_test, y_pred)

# print("Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)

# mse = mean_squared_error(y_test, y_pred)
# print("Mean Squared Error:", mse)

# r2 = r2_score(y_test, y_pred)
# print("R-squared:", r2)

# -----------------------------------------------------------------------------------------------
# =============================================================================
#  SECTION I — MACHINE LEARNING MODEL — XGBOOST CLASSIFIER WHERE AVNRT IS PREDICTED AS 0 AND AVRT AS 1
# =============================================================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, LeaveOneOut, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb

# =============================================================================
# 1. LOAD DATA & SELECT FEATURES
# =============================================================================
# data = pd.read_excel('ECG_ALL_ML_features.xlsx')
out_ml = 'ECG_ALL_ML_features.xlsx'
data = pd.read_excel(out_ml)  # Load the combined ML features Excel file

ML_FEATURES_FINAL_CH3 = [
    # Tier 1 — strongest separators (low NaN, significant p-value)
    'ch3_pNN50_pct',
    'ch3_HR_std_bpm',
    'ch3_beat_skewness_mean',
    'ch3_beat_kurtosis_mean',
    'ch3_QRS_duration_ms_mean',
    'ch3_P_wave_presence_ratio',
    'ch3_ST_slope_mean',
    'ch3_R_peak_count',

    # Tier 2 — strong effect but handle NaN with imputation
    'ch3_RP_interval_mean_ms',
    'ch3_RP_PR_ratio',

    # Tier 3 — moderate, statistically significant
    'ch3_SD2_ms',
    'ch3_SD1_ms',
    'ch3_SD1_SD2_ratio',
    'ch3_RR_std_ms',
    'ch3_RMSSD_ms',
    'ch3_RR_mean_ms',
    'ch3_P_wave_absent_flag',
]

ML_FEATURES_FINAL_CH4 = [
    # Tier 1 — strongest separators (low NaN, significant p-value)
    'ch4_pNN50_pct',
    'ch4_HR_std_bpm',
    'ch4_beat_skewness_mean',
    'ch4_beat_kurtosis_mean',
    'ch4_QRS_duration_ms_mean',
    'ch4_P_wave_presence_ratio',
    'ch4_ST_slope_mean',
    'ch4_R_peak_count',

    # Tier 2 — strong effect but handle NaN with imputation
    'ch4_RP_interval_mean_ms',
    'ch4_RP_PR_ratio',

    # Tier 3 — moderate, statistically significant
    'ch4_SD2_ms',
    'ch4_SD1_ms',
    'ch4_SD1_SD2_ratio',
    'ch4_RR_std_ms',
    'ch4_RMSSD_ms',
    'ch4_RR_mean_ms',
    'ch4_P_wave_absent_flag',
]

ML_FEATURES_FINAL_CH4_SIGNIFICANT = [
    'ch4_SD1_SD2_ratio',
    'ch4_P_wave_absent_flag',
]

COMBINED_FEATURES = ML_FEATURES_FINAL_CH3 + ML_FEATURES_FINAL_CH4_SIGNIFICANT

X = data[COMBINED_FEATURES]
y = data['label_num']

# =============================================================================
# 2. TRAIN / TEST SPLIT
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# =============================================================================
# 3. BUILD PIPELINE & TUNING FOR XGBOOST
# =============================================================================
# We explicitly tell XGBoost to look for missing values (np.nan) natively
xgb_pipeline = Pipeline([
    ('xgb', xgb.XGBClassifier(
        objective='binary:logistic',
        missing=np.nan,
        random_state=42,
        eval_metric='logloss'
    ))
])

# Define hyperparameter grid tailored to XGBoost on small data
xgb_param_grid = {
    'xgb__n_estimators': [50, 100],
    'xgb__max_depth': [2, 3, 4],
    'xgb__learning_rate': [0.01, 0.1, 0.2],
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

xgb_grid_search = GridSearchCV(
    xgb_pipeline,
    xgb_param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1
)

# Fit on training data only to optimize hyperparameters safely
xgb_grid_search.fit(X_train, y_train)
print(f"XGBoost best parameters: {xgb_grid_search.best_params_}")
print(f"Best Cross Validation (CV) accuracy for XGBoost : {xgb_grid_search.best_score_:.4f}")

# =============================================================================
# 4. EVALUATE ON TEST SET
# =============================================================================
best_xgb_model = xgb_grid_search.best_estimator_
y_pred_xgb = best_xgb_model.predict(X_test)

print("\n" + "="*50)
print("XGBOOST — TEST SET RESULTS")
print("="*50)
print(f"XGBoost test accuracy : {accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"XGBoost precision     : {precision_score(y_test, y_pred_xgb):.4f}")
print(f"XGBoost recall        : {recall_score(y_test, y_pred_xgb):.4f}")
print(f"XGBoost F1 score      : {f1_score(y_test, y_pred_xgb):.4f}")

# Plot Confusion Matrix
confusion_xgb = confusion_matrix(y_test, y_pred_xgb)
ConfusionMatrixDisplay(confusion_matrix=confusion_xgb, display_labels=['AVNRT (0)', 'AVRT (1)']).plot(cmap='Blues')
plt.title('XGBoost — Confusion Matrix')
plt.show()

# =============================================================================
# 5. LEAVE-ONE-OUT CROSS-VALIDATION
# =============================================================================
# Extract structural best parameters dynamically to plug into LOO evaluation loop
loo_xgb_pipeline = Pipeline([
    ('xgb', xgb.XGBClassifier(
        objective='binary:logistic',
        missing=np.nan,
        n_estimators=xgb_grid_search.best_params_['xgb__n_estimators'],
        max_depth=xgb_grid_search.best_params_['xgb__max_depth'],
        learning_rate=xgb_grid_search.best_params_['xgb__learning_rate'],
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    ))
])

xgb_loo_scores = cross_val_score(loo_xgb_pipeline, X, y, cv=LeaveOneOut(), scoring='accuracy')
print(f"\nXGBoost LOO-CV accuracy: {xgb_loo_scores.mean():.4f} ({int(sum(xgb_loo_scores))}/46 recordings)")

# =============================================================================
# 6. FEATURE IMPORTANCE
# =============================================================================
xgb_internal_model = best_xgb_model.named_steps['xgb']
feat_imp_xgb = pd.DataFrame({
    'feature'   : COMBINED_FEATURES,
    'importance': xgb_internal_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nXGBoost feature importances:")
print(feat_imp_xgb.to_string(index=False))


# =============================================================================
#  SECTION J — OVERALL DATASET AUDIT AND MAPPING
# =============================================================================
from sklearn.model_selection import LeaveOneOut

# 1. Initialize lists to hold out-of-sample predictions
all_predictions = np.zeros(len(X))
all_probabilities = np.zeros(len(X))

# 2. Re-instantiate the winning XGBoost pipeline layout
winning_pipeline = Pipeline([
    ('xgb', xgb.XGBClassifier(
        objective='binary:logistic',
        missing=np.nan,
        n_estimators=xgb_grid_search.best_params_['xgb__n_estimators'],
        max_depth=xgb_grid_search.best_params_['xgb__max_depth'],
        learning_rate=xgb_grid_search.best_params_['xgb__learning_rate'],
        random_state=42,
        eval_metric='logloss',
        n_jobs=-1
    ))
])

# 3. Step through the entire dataset sample-by-sample using LOO
loo = LeaveOneOut()
for train_idx, test_idx in loo.split(X):
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]

    winning_pipeline.fit(X_tr, y_tr)

    all_predictions[test_idx] = winning_pipeline.predict(X_te)
    all_probabilities[test_idx] = winning_pipeline.predict_proba(X_te)[:, 1]

# 4. Construct a comprehensive summary DataFrame
audit_df = pd.DataFrame({
    'Row_Index': data.index,
    # Try to grab a 'file' or identification column if it exists in your raw data
    'Recording_ID': data['file'] if 'file' in data.columns else data.index,
    'True_Label_Num': y,
    'Predicted_Label_Num': all_predictions.astype(int),
    'AVRT_Probability': all_probabilities
})

# Map numeric classifications back to clinical strings
label_map = {0: 'AVNRT', 1: 'AVRT'}
audit_df['True_Label'] = audit_df['True_Label_Num'].map(label_map)
audit_df['Predicted_Label'] = audit_df['Predicted_Label_Num'].map(label_map)

# Classify the machine learning outcome category
def get_outcome(row):
    if row['True_Label_Num'] == 1 and row['Predicted_Label_Num'] == 1:
        return 'True Positive (Correct AVRT)'
    elif row['True_Label_Num'] == 0 and row['Predicted_Label_Num'] == 0:
        return 'True Negative (Correct AVNRT)'
    elif row['True_Label_Num'] == 0 and row['Predicted_Label_Num'] == 1:
        return 'False Positive (AVNRT misclassified as AVRT)'
    else:
        return 'False Negative (AVRT misclassified as AVNRT)'

audit_df['Classification_Outcome'] = audit_df.apply(get_outcome, axis=1)

# 5. Print a clean, formatted diagnostic summary to the console
print("\n" + "="*60)
print("             XGBOOST FULL DATASET DIAGNOSTIC AUDIT          ")
print("="*60)
print(f"Total Recordings Evaluated: {len(audit_df)}")
print(f"XGBoost LOO-CV accuracy: {xgb_loo_scores.mean():.4f} ({int(sum(xgb_loo_scores))}/46 recordings)")
print(f"XGBoost Best CV accuracy: {xgb_grid_search.best_score_:.4f}")
print("\n--- Breakdown of Diagnostic Outcomes ---")
print(audit_df['Classification_Outcome'].value_counts().to_string())

print("\n--- List of Misclassified Clinical Cases ---")
failures = audit_df[audit_df['True_Label_Num'] != audit_df['Predicted_Label_Num']]
if not failures.empty:
    print(failures[['Recording_ID', 'True_Label', 'Predicted_Label', 'AVRT_Probability']].to_string(index=False))
else:
    print("Incredible! No misclassifications found.")

# 6. Save back to a spreadsheet for clinical presentation
audit_df.to_excel('XGBoost_Full_Dataset_Audit.xlsx', index=False)
print("\nSaved full audit trace to: 'XGBoost_Full_Dataset_Audit.xlsx'")
