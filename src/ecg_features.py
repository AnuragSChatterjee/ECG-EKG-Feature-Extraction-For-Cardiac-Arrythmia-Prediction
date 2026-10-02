"""
ecg_features.py
---------------
Signal-processing helpers and hand-crafted ECG feature extractors used for
AVNRT vs AVRT classification.

Feature categories (per channel):
  1. Temporal / HRV           (RR stats, HR, RMSSD, SDSD, pNN50)
  2. Morphological            (QRS amplitude/duration, skewness, kurtosis, RS, ST slope, T wave)
  3. Interval                 (P wave, PR, QT/QTc, RP, VA, RP/PR ratio)
  4. QRS alternans
  5. Spectral                 (Welch PSD band powers, peak/centroid freq, spectral flatness)
  6. Nonlinear / Poincare     (SD1, SD2, Higuchi FD, ApEn, SampEn, Lyapunov proxy)
  7. Wavelet                  (DWT db4 energy, levels 1-4)
  8. R-peak amplitude stats

Author: Anurag Chatterjee
"""
import numpy as np
from scipy.signal import butter, sosfiltfilt

# np.trapz was renamed np.trapezoid in NumPy 2.x
_trapz = getattr(np, "trapezoid", None) or np.trapz

try:
    import neurokit2 as nk
    NK_AVAILABLE = True
except ImportError:
    NK_AVAILABLE = False
    print("WARNING: neurokit2 not installed - P/T-wave delineation skipped (pip install neurokit2).")

try:
    import pywt
    PWT_AVAILABLE = True
except ImportError:
    PWT_AVAILABLE = False
    print("WARNING: PyWavelets not installed - DWT features will be NaN (pip install PyWavelets).")


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
        arrs['beat_area'][i]       = _trapz(seg)
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
        arrs['signal_integral'][i]  = float(_trapz(np.abs(seg)))
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
        return float(_trapz(pxx[idx], f[idx])) if np.sum(idx)>=2 else np.nan
    total  = float(_trapz(pxx, f))
    pk_idx = int(np.argmax(pxx))
    return {
        'P_wave_band_power'    : band_power(0.67, 5.0),
        'T_wave_band_power'    : band_power(1.0,  7.0),
        'QRS_band_power'       : band_power(10.0, 50.0),
        'total_spectral_energy': total,
        'peak_frequency_Hz'    : float(f[pk_idx]),
        'max_power'            : float(pxx[pk_idx]),
        'centroid_freq_Hz'     : float(_trapz(f*pxx,f)/total) if total>0 else np.nan,
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
