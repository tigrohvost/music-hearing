"""Optional numpy FFT spectral features: rich centroid/rolloff/flatness, a
12-bin chroma + Krumhansl key estimate, MFCC timbre, an onset-autocorrelation
tempo, harmonic/percussive split, stereo width, and a loudness/centroid
timeline.

The base :mod:`music_hearing.dsp` profiler runs at 8 kHz with a 15-probe
Goertzel band sketch — it loses treble and can't name key or texture. This
module adds the detailed view. It requires numpy (``pip install
music-hearing[rich]``) and is opt-in via ``--rich`` / ``rich=True``.
"""
from __future__ import annotations

from typing import Any

import numpy as np

_KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
# Krumhansl-Kessler key profiles.
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_EPS = 1e-10


def _stft_mag(x: np.ndarray, win: int = 2048, hop: int = 1024) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size < win:
        x = np.pad(x, (0, win - x.size))
    w = np.hanning(win)
    n = 1 + (x.size - win) // hop
    frames = np.stack([x[i * hop:i * hop + win] * w for i in range(n)])
    return np.abs(np.fft.rfft(frames, axis=1))


def _freqs(rate: int, win: int = 2048) -> np.ndarray:
    return np.fft.rfftfreq(win, 1.0 / rate)


def _centroid(mag_avg: np.ndarray, freqs: np.ndarray) -> float:
    total = mag_avg.sum() + _EPS
    return float((freqs * mag_avg).sum() / total)


def rich_features(samples: Any, rate: int) -> dict:
    """True FFT spectral summary: centroid, rolloff(85%), flatness, bandwidth,
    mean spectral flux, and an onset rate (flux peaks per second)."""
    mag = _stft_mag(samples)
    freqs = _freqs(rate)
    avg = mag.mean(axis=0)
    total = avg.sum() + _EPS
    centroid = _centroid(avg, freqs)
    cum = np.cumsum(avg)
    roll_idx = int(np.searchsorted(cum, 0.85 * cum[-1]))
    rolloff = float(freqs[min(roll_idx, len(freqs) - 1)])
    flatness = float(np.exp(np.mean(np.log(avg + _EPS))) / (avg.mean() + _EPS))
    bandwidth = float(np.sqrt((avg * (freqs - centroid) ** 2).sum() / total))
    if mag.shape[0] >= 2:
        flux = np.maximum(np.diff(mag, axis=0), 0.0).sum(axis=1)
        flux_mean = float(flux.mean() / mag.shape[1])
        thr = flux.mean() + flux.std()
        body = flux[1:-1]
        peaks = (body > thr) & (body > flux[:-2]) & (body > flux[2:])
        onset_rate = float(int(peaks.sum()) / (np.asarray(samples).size / rate))
    else:
        flux_mean = 0.0
        onset_rate = 0.0
    return {
        "spectral_centroid_hz": round(centroid, 1),
        "spectral_rolloff_hz": round(rolloff, 1),
        "spectral_flatness": round(min(1.0, max(0.0, flatness)), 4),
        "spectral_bandwidth_hz": round(bandwidth, 1),
        "spectral_flux_mean": round(flux_mean, 6),
        "onset_rate_hz": round(onset_rate, 2),
        "sample_rate": int(rate),
    }


def chroma_vector(samples: Any, rate: int) -> list:
    """12-bin pitch-class energy (C..B), normalized to sum 1."""
    mag = _stft_mag(samples)
    avg = mag.mean(axis=0)
    freqs = _freqs(rate)
    mask = freqs > 20.0
    with np.errstate(divide="ignore", invalid="ignore"):
        midi = 69.0 + 12.0 * np.log2(np.where(mask, freqs, 1.0) / 440.0)
    pc = np.mod(np.round(midi).astype(int), 12)
    chroma = np.zeros(12)
    idx = np.where(mask)[0]
    np.add.at(chroma, pc[idx], avg[idx])
    s = chroma.sum()
    if s > 0:
        chroma = chroma / s
    return [float(v) for v in chroma]


def estimate_key(chroma: Any) -> dict:
    """Krumhansl key estimate from a 12-bin chroma -> {key, mode, confidence}."""
    c = np.asarray(chroma, dtype=np.float64)
    cn = c - c.mean()
    best = (-2.0, 0, "major")
    for t in range(12):
        for prof, mode in ((_MAJOR, "major"), (_MINOR, "minor")):
            p = np.roll(prof, t)
            p = p - p.mean()
            denom = np.sqrt((cn ** 2).sum() * (p ** 2).sum()) + _EPS
            corr = float((cn * p).sum() / denom)
            if corr > best[0]:
                best = (corr, t, mode)
    return {
        "key": _KEYS[best[1]],
        "mode": best[2],
        "confidence": round(max(0.0, min(1.0, (best[0] + 1.0) / 2.0)), 4),
    }


def timeline(samples: Any, rate: int, window_sec: float = 5.0) -> list:
    """Per-window loudness (dBFS) + spectral centroid — the arrangement shape."""
    x = np.asarray(samples, dtype=np.float64)
    w = max(1, int(window_sec * rate))
    out = []
    for i in range(0, x.size, w):
        seg = x[i:i + w]
        if seg.size < w // 2:
            break
        rms = float(np.sqrt(np.mean(seg ** 2)))
        dbfs = 20.0 * np.log10(max(rms, 1e-9))
        mag = _stft_mag(seg).mean(axis=0)
        out.append({
            "t": round(i / rate, 2),
            "rms_dbfs": round(dbfs, 2),
            "centroid_hz": round(_centroid(mag, _freqs(rate)), 1),
        })
    return out


def stereo_width(left: Any, right: Any) -> dict:
    """Mid/side width + L/R correlation. width = rms(side)/rms(mid)."""
    l = np.asarray(left, dtype=np.float64)
    r = np.asarray(right, dtype=np.float64)
    n = min(l.size, r.size)
    l, r = l[:n], r[:n]
    mid = (l + r) / 2.0
    side = (l - r) / 2.0
    rms_mid = float(np.sqrt(np.mean(mid ** 2))) + _EPS
    rms_side = float(np.sqrt(np.mean(side ** 2)))
    ln, rn = l - l.mean(), r - r.mean()
    denom = np.sqrt((ln ** 2).sum() * (rn ** 2).sum()) + _EPS
    corr = float((ln * rn).sum() / denom)
    return {
        "correlation": round(max(-1.0, min(1.0, corr)), 4),
        "width": round(rms_side / rms_mid, 4),
    }


def _k_weight_power(freqs: np.ndarray, rate: int) -> np.ndarray:
    """ITU-R BS.1770 K-weighting magnitude-squared response (48k coeffs,
    evaluated at the actual rate — an approximation good enough for relative
    loudness)."""
    b1 = (1.53512485958697, -2.69169618940638, 1.19839281085285)
    a1 = (1.0, -1.69065929318241, 0.73248077421585)
    b2 = (1.0, -2.0, 1.0)
    a2 = (1.0, -1.99004745483398, 0.99007225036621)
    z = np.exp(-1j * 2.0 * np.pi * freqs / rate)

    def _h(b, a):
        return (b[0] + b[1] * z + b[2] * z ** 2) / (a[0] + a[1] * z + a[2] * z ** 2)

    return np.abs(_h(b1, a1) * _h(b2, a2)) ** 2


def loudness_lufs(samples: Any, rate: int) -> float:
    """Approximate integrated loudness (LUFS): K-weighted mean square, ungated.
    Absolute scale is uncalibrated; differences/ordering are meaningful."""
    mag = _stft_mag(samples)
    w = _k_weight_power(_freqs(rate), rate)
    power = (mag ** 2).mean(axis=0)
    ms = float((power * w).sum() / (mag.shape[1] * 2048))
    return round(-0.691 + 10.0 * np.log10(ms + 1e-12), 2)


def _mel_filterbank(n_filters: int, n_fft: int, rate: int) -> np.ndarray:
    def hz2mel(f):
        return 2595.0 * np.log10(1.0 + f / 700.0)

    def mel2hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    lo, hi = hz2mel(0.0), hz2mel(rate / 2.0)
    points = mel2hz(np.linspace(lo, hi, n_filters + 2))
    bins = np.floor((n_fft + 1) * points / rate).astype(int)
    fb = np.zeros((n_filters, n_fft // 2 + 1))
    for i in range(1, n_filters + 1):
        l, c, r = bins[i - 1], bins[i], bins[i + 1]
        for k in range(l, c):
            if c > l:
                fb[i - 1, k] = (k - l) / (c - l)
        for k in range(c, r):
            if r > c:
                fb[i - 1, k] = (r - k) / (r - c)
    return fb


def mfcc(samples: Any, rate: int, n_mfcc: int = 13, n_filters: int = 26,
         win: int = 2048) -> list:
    """Mean MFCC vector (mel filterbank -> log -> DCT-II), a timbre fingerprint."""
    mag = _stft_mag(samples, win=win)
    fb = _mel_filterbank(n_filters, win, rate)
    mel = mag @ fb.T
    log_mel = np.log(mel + _EPS)
    k = np.arange(n_filters)
    dct = np.cos(np.pi / n_filters * (k[:, None] + 0.5) * np.arange(n_mfcc)[None, :])
    coeffs = (log_mel @ dct).mean(axis=0)
    return [round(float(v), 4) for v in coeffs]


def _onset_env(samples: Any, rate: int, hop: int = 512):
    mag = _stft_mag(samples, win=1024, hop=hop)
    flux = np.maximum(np.diff(mag, axis=0), 0.0).sum(axis=1)
    # Scale-free onset strength: rise energy vs total spectral energy. A steady
    # tone -> ~0 (no onsets) regardless of how loud it is; a click train -> high.
    strength = float(flux.sum() / (mag.sum() + _EPS))
    return flux - flux.mean(), rate / hop, strength


def estimate_tempo(samples: Any, rate: int, bpm_min: float = 50.0,
                   bpm_max: float = 200.0) -> dict:
    """BPM from the onset-envelope autocorrelation, with an octave check and a
    confidence (autocorr peak relative to lag-0)."""
    env, frame_rate, strength = _onset_env(samples, rate)
    if env.size < 4 or not np.any(env):
        return {"bpm": None, "confidence": 0.0, "candidates": []}
    ac = np.correlate(env, env, mode="full")[env.size - 1:]
    lag_min = max(1, int(frame_rate * 60.0 / bpm_max))
    lag_max = min(ac.size - 1, int(frame_rate * 60.0 / bpm_min))
    if lag_max <= lag_min:
        return {"bpm": None, "confidence": 0.0, "candidates": []}
    seg = ac[lag_min:lag_max]
    lags = np.arange(lag_min, lag_max)
    bpms = 60.0 * frame_rate / lags
    # Log-normal tempo prior (~120 BPM) disambiguates octave errors: a click
    # train autocorrelates at both its period and double it; the prior picks the
    # musical reading instead of the slowest peak.
    prior = np.exp(-0.5 * (np.log2(bpms / 120.0) / 0.9) ** 2)
    peak_i = int(np.argmax(seg * prior))
    peak = peak_i + lag_min
    bpm = 60.0 * frame_rate / peak
    # Confidence = peak prominence scaled by onset activity. A steady tone has
    # almost no onsets (a degenerate envelope can still look "peaky"), so gate on
    # how many real onset events exist — tempo needs a repeating pulse.
    prominence = (seg[peak_i] - seg.mean()) / (abs(seg[peak_i]) + _EPS)
    activity = min(1.0, strength / 0.05)
    conf = float(max(0.0, min(1.0, prominence)) * activity)
    cands = [round(bpm, 1)]
    for factor in (0.5, 2.0):
        alt = bpm * factor
        if bpm_min <= alt <= bpm_max:
            cands.append(round(alt, 1))
    return {"bpm": round(bpm, 1), "confidence": round(conf, 4), "candidates": cands}


def _median_axis(m: np.ndarray, axis: int, k: int = 9) -> np.ndarray:
    from numpy.lib.stride_tricks import sliding_window_view
    pad = [(0, 0), (0, 0)]
    pad[axis] = (k // 2, k // 2)
    mp = np.pad(m, pad, mode="edge")
    return np.median(sliding_window_view(mp, k, axis=axis), axis=-1)


def hpss_ratio(samples: Any, rate: int, k: int = 9) -> dict:
    """Harmonic/percussive energy split (Fitzgerald median-filter HPSS):
    harmonic structures are smooth over time, percussive over frequency."""
    mag = _stft_mag(samples)
    if mag.shape[0] < 2:
        return {"harmonic_ratio": 0.5, "percussive_ratio": 0.5}
    h = _median_axis(mag, axis=0, k=k)   # smooth across time
    p = _median_axis(mag, axis=1, k=k)   # smooth across frequency
    denom = h + p + _EPS
    he = float((mag * (h / denom)).sum())
    pe = float((mag * (p / denom)).sum())
    tot = he + pe + _EPS
    return {
        "harmonic_ratio": round(he / tot, 4),
        "percussive_ratio": round(pe / tot, 4),
    }


def load_stereo_samples(path: Any, target_rate: int = 22050, max_seconds: float = 45.0):
    """Decode (left, right) float arrays via ffmpeg; mono files duplicate."""
    import shutil
    import subprocess
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg required for stereo decode")
    cmd = [ffmpeg, "-v", "error", "-t", str(max_seconds), "-i", str(path),
           "-ac", "2", "-ar", str(target_rate), "-f", "s16le", "-"]
    raw = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).stdout
    data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    if data.size % 2:
        data = data[:-1]
    inter = data.reshape(-1, 2)
    return inter[:, 0], inter[:, 1]


def rich_profile_from_samples(samples: Any, rate: int, window_sec: float = 5.0) -> dict:
    chroma = chroma_vector(samples, rate)
    out = dict(rich_features(samples, rate))
    out["chroma"] = chroma
    out["key"] = estimate_key(chroma)
    out["timeline"] = timeline(samples, rate, window_sec=window_sec)
    out["lufs_approx"] = loudness_lufs(samples, rate)
    out["mfcc"] = mfcc(samples, rate)
    out["tempo"] = estimate_tempo(samples, rate)
    out["hpss"] = hpss_ratio(samples, rate)
    return out


def rich_profile(path: Any, target_rate: int = 22050, max_seconds: float = 45.0,
                 window_sec: float = 5.0) -> dict:
    """Decode a file (reusing the base loader) then compute the rich profile,
    including stereo width when the source has two channels."""
    from . import dsp
    samples, rate = dsp.load_mono_samples(path, target_rate=target_rate,
                                          max_seconds=max_seconds)
    out = rich_profile_from_samples(np.asarray(samples, dtype=np.float64), rate,
                                    window_sec=window_sec)
    try:
        left, right = load_stereo_samples(path, target_rate=target_rate, max_seconds=max_seconds)
        out["stereo"] = stereo_width(left, right)
    except Exception:
        out["stereo"] = None
    return out
