"""Rhythm and groove proxies for music-hearing v2."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

_EPS = 1e-10


def _clip01(x: float) -> float:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(f):
        return 0.0
    return float(max(0.0, min(1.0, f)))


def _stft_mag(x: np.ndarray, win: int = 1024, hop: int = 512) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.size < win:
        x = np.pad(x, (0, win - x.size))
    n = 1 + max(0, (x.size - win) // hop)
    window = np.hanning(win)
    frames = np.stack([x[i * hop:i * hop + win] * window for i in range(n)])
    return np.abs(np.fft.rfft(frames, axis=1))


def _freqs(rate: int, win: int = 1024) -> np.ndarray:
    return np.fft.rfftfreq(win, 1.0 / rate)


def onset_envelope(samples: Any, rate: int, hop: int = 512) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    mag = _stft_mag(np.asarray(samples, dtype=np.float64), hop=hop)
    flux_bins = np.maximum(np.diff(mag, axis=0), 0.0)
    flux = flux_bins.sum(axis=1)
    if flux.size:
        flux = flux / (np.percentile(flux, 95) + _EPS)
    return flux, rate / hop, flux_bins, _freqs(rate)


def _peaks(env: np.ndarray) -> np.ndarray:
    if env.size < 3 or not np.any(env):
        return np.array([], dtype=int)
    threshold = float(env.mean() + 0.75 * env.std())
    body = env[1:-1]
    return np.where((body > threshold) & (body >= env[:-2]) & (body >= env[2:]))[0] + 1


def _tempo_candidates(env: np.ndarray, frame_rate: float,
                      bpm_min: float = 45.0, bpm_max: float = 180.0) -> tuple[list[dict], float | None, float]:
    if env.size < 8 or not np.any(env):
        return [], None, 0.0
    y = env - env.mean()
    ac = np.correlate(y, y, mode="full")[y.size - 1:]
    if ac[0] <= _EPS:
        return [], None, 0.0
    lag_min = max(1, int(frame_rate * 60.0 / bpm_max))
    lag_max = min(ac.size - 1, int(frame_rate * 60.0 / bpm_min))
    if lag_max <= lag_min:
        return [], None, 0.0
    lags = np.arange(lag_min, lag_max + 1)
    bpms = 60.0 * frame_rate / lags
    seg = ac[lags] / (ac[0] + _EPS)
    prior = np.exp(-0.5 * (np.log2(bpms / 95.0) / 1.0) ** 2)
    scores = np.maximum(0.0, seg) * prior
    order = np.argsort(scores)[::-1]
    out = []
    used: list[float] = []
    for idx in order[:12]:
        bpm = float(bpms[idx])
        if any(abs(np.log2(bpm / u)) < 0.045 for u in used):
            continue
        used.append(bpm)
        out.append({"bpm": round(bpm, 1), "score": round(_clip01(float(scores[idx])), 4)})
        if len(out) >= 3:
            break
    if out:
        best_bpm = float(out[0]["bpm"])
        best_conf = _clip01(float(out[0]["score"]))
        for factor in (0.5, 2.0):
            alt = best_bpm * factor
            if bpm_min <= alt <= bpm_max and all(abs(alt - c["bpm"]) > 2.0 for c in out):
                out.append({"bpm": round(alt, 1), "score": round(best_conf * 0.62, 4)})
    else:
        best_bpm, best_conf = None, 0.0
    return out[:3], best_bpm, best_conf


def _phase_and_grid(env: np.ndarray, frame_rate: float, bpm: float | None, peaks: np.ndarray) -> dict:
    if bpm is None or env.size == 0:
        return {"period_sec": None, "phase_sec": None, "stability": 0.0,
                "local_tempo_std_bpm": None, "pulse_clarity": 0.0, "swing_ratio": None}
    period_frames = max(1.0, frame_rate * 60.0 / bpm)
    bins = int(max(1, round(period_frames)))
    phase_scores = []
    for phase in range(bins):
        idx = np.arange(phase, env.size, bins).astype(int)
        phase_scores.append(float(env[idx].mean()) if idx.size else 0.0)
    phase = int(np.argmax(phase_scores)) if phase_scores else 0
    best = max(phase_scores) if phase_scores else 0.0
    avg = float(np.mean(env) + _EPS)
    pulse = _clip01((best - avg) / (best + avg + _EPS) * 2.0)
    stability = pulse
    local_std = None
    if peaks.size >= 4:
        intervals = np.diff(peaks) / frame_rate
        if intervals.size:
            bpms = 60.0 / np.maximum(intervals, 1e-3)
            plausible = bpms[(bpms >= 45.0) & (bpms <= 180.0)]
            if plausible.size >= 2:
                local_std = round(float(np.std(plausible)), 3)
                stability = _clip01(stability * (1.0 - min(1.0, local_std / 20.0)))
    swing_ratio = None
    half = max(1, int(round(period_frames / 2.0)))
    if bins >= 4 and env.size > bins * 2:
        straight = []
        swung = []
        for start in np.arange(phase, env.size - bins, bins).astype(int):
            straight.append(float(env[min(env.size - 1, start + half)]))
            swung.append(float(env[min(env.size - 1, start + int(round(period_frames * 2.0 / 3.0)))]))
        denom = float(np.mean(straight) + np.mean(swung) + _EPS)
        swing_ratio = round(float(np.mean(swung) / denom), 4) if denom > 0 else None
    return {
        "period_sec": round(60.0 / bpm, 4),
        "phase_sec": round(phase / frame_rate, 4),
        "stability": round(stability, 4),
        "local_tempo_std_bpm": local_std,
        "pulse_clarity": round(pulse, 4),
        "swing_ratio": swing_ratio,
    }


def _band_rates(flux_bins: np.ndarray, freqs: np.ndarray, duration: float) -> dict:
    if flux_bins.size == 0 or duration <= 0:
        return {"low_band_onset_rate": 0.0, "mid_band_onset_rate": 0.0, "high_band_onset_rate": 0.0}
    bands = {
        "low_band_onset_rate": freqs < 250.0,
        "mid_band_onset_rate": (freqs >= 250.0) & (freqs < 2000.0),
        "high_band_onset_rate": freqs >= 2000.0,
    }
    out = {}
    for name, mask in bands.items():
        if not mask.any():
            out[name] = 0.0
            continue
        env = flux_bins[:, mask].sum(axis=1)
        peaks = _peaks(env / (np.percentile(env, 95) + _EPS) if np.any(env) else env)
        out[name] = round(float(peaks.size) / duration, 4)
    return out


def _syncopation(env: np.ndarray, frame_rate: float, bpm: float | None, phase_sec: float | None) -> dict:
    if bpm is None or phase_sec is None or env.size == 0:
        return {"offbeat_energy_ratio": 0.0, "syncopation_index": 0.0,
                "backbeat_strength": 0.0, "downbeat_certainty": 0.0}
    period = frame_rate * 60.0 / bpm
    phase = int(round(phase_sec * frame_rate))
    on, off, back = [], [], []
    width = max(1, int(period * 0.12))
    for start in np.arange(phase, env.size, period):
        s = int(round(start))
        on.extend(env[max(0, s - width):min(env.size, s + width + 1)])
        off_i = int(round(start + period / 2.0))
        off.extend(env[max(0, off_i - width):min(env.size, off_i + width + 1)])
        back_i = int(round(start + period))
        back.extend(env[max(0, back_i - width):min(env.size, back_i + width + 1)])
    on_m = float(np.mean(on)) if on else 0.0
    off_m = float(np.mean(off)) if off else 0.0
    back_m = float(np.mean(back)) if back else 0.0
    total = on_m + off_m + _EPS
    off_ratio = _clip01(off_m / total)
    return {
        "offbeat_energy_ratio": round(off_ratio, 4),
        "syncopation_index": round(_clip01(max(0.0, off_m - on_m * 0.55) / (max(off_m, on_m) + _EPS)), 4),
        "backbeat_strength": round(_clip01(back_m / (on_m + back_m + _EPS)), 4),
        "downbeat_certainty": round(_clip01((on_m - off_m) / (on_m + off_m + _EPS)), 4),
    }


def analyze(samples: Any, rate: int, rich: Mapping[str, Any] | None = None) -> dict:
    x = np.asarray(samples, dtype=np.float64)
    duration = max(float(x.size) / float(rate or 1), 1e-6)
    env, frame_rate, flux_bins, freqs = onset_envelope(x, rate)
    peaks = _peaks(env)
    candidates, bpm, conf = _tempo_candidates(env, frame_rate)
    if rich and isinstance(rich.get("tempo"), Mapping) and rich["tempo"].get("bpm") is not None:
        if conf < float(rich["tempo"].get("confidence") or 0.0):
            bpm = float(rich["tempo"]["bpm"])
            conf = _clip01(float(rich["tempo"].get("confidence") or conf))
            if not candidates or abs(candidates[0]["bpm"] - bpm) > 2.0:
                candidates = [{"bpm": round(bpm, 1), "score": round(conf, 4)}] + candidates[:2]
    grid = _phase_and_grid(env, frame_rate, bpm, peaks)
    band_rates = _band_rates(flux_bins, freqs, duration)
    onset_rate = round(float(peaks.size) / duration, 4)
    bars = duration / max(60.0 / float(bpm or 60.0) * 4.0, 1e-6)
    density = {
        "onset_rate_per_sec": onset_rate,
        "onsets_per_bar_est": round(float(peaks.size) / max(bars, 1e-6), 4),
        **band_rates,
    }
    return {
        "bpm": round(float(bpm), 3) if bpm is not None else None,
        "bpm_confidence": round(_clip01(conf), 4),
        "tempo_candidates": candidates[:3],
        "beat_grid": grid,
        "density": density,
        "syncopation": _syncopation(env, frame_rate, bpm, grid.get("phase_sec")),
    }
