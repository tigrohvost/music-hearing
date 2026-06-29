"""Mixture-level timbre families and lo-fi artifact proxies."""
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


def _spectrum(x: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray]:
    n = min(65536, max(2048, int(2 ** np.ceil(np.log2(max(32, x.size))))))
    if x.size < n:
        x = np.pad(x, (0, n - x.size))
    mag = np.abs(np.fft.rfft(x[:n] * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / rate)
    return mag, freqs


def _band_energy(mag: np.ndarray, freqs: np.ndarray, lo: float, hi: float) -> float:
    mask = (freqs >= lo) & (freqs < hi)
    return float(mag[mask].sum())


def _rms_frames(x: np.ndarray, rate: int, frame_sec: float = 0.05) -> np.ndarray:
    n = max(1, int(rate * frame_sec))
    frames = [x[i:i + n] for i in range(0, x.size, n) if x[i:i + n].size >= n // 2]
    if not frames:
        return np.zeros(1)
    return np.asarray([float(np.sqrt(np.mean(f ** 2))) for f in frames])


def _attack_profile(x: np.ndarray, rate: int, rhythm: Mapping[str, Any] | None) -> dict:
    rms = _rms_frames(x, rate)
    diff = np.maximum(np.diff(rms), 0.0)
    sharp = float(np.percentile(diff, 95) / (np.percentile(rms, 95) + _EPS)) if rms.size else 0.0
    sustain = float(np.mean(rms > np.percentile(rms, 40))) if rms.size else 0.0
    onset_rate = ((rhythm or {}).get("density") or {}).get("onset_rate_per_sec", 0.0)
    soft = 1.0 - _clip01(float(onset_rate or 0.0) / 8.0 + sharp * 0.3)
    return {
        "soft_attack_ratio": round(_clip01(soft), 4),
        "transient_sharpness": round(_clip01(sharp), 4),
        "sustain_ratio": round(_clip01(sustain), 4),
    }


def _lofi(x: np.ndarray, rate: int, mag: np.ndarray, freqs: np.ndarray,
          rich: Mapping[str, Any] | None) -> dict:
    total = float(mag.sum() + _EPS)
    high = _band_energy(mag, freqs, 5000.0, min(rate / 2, 10500.0)) / total
    flatness = float((rich or {}).get("spectral_flatness") or 0.0)
    hiss = _clip01(high * 3.0 + flatness * 0.25)
    hum = 0.0
    for base in (50.0, 60.0):
        e = sum(_band_energy(mag, freqs, base * k - 1.5, base * k + 1.5) for k in (1, 2, 3))
        hum = max(hum, e / total * 40.0)
    rms = _rms_frames(x, rate, frame_sec=0.02)
    diff = np.abs(np.diff(rms))
    click_thr = float(diff.mean() + 4.0 * diff.std()) if diff.size else 1.0
    clicks = int(np.sum(diff > click_thr)) if diff.size and click_thr > 0 else 0
    duration = max(x.size / float(rate or 1), 1e-6)
    absx = np.abs(x)
    peak = float(absx.max()) if absx.size else 0.0
    near_peak = float(np.mean(absx > peak * 0.985)) if peak > 0 else 0.0
    soft_clip = _clip01(near_peak * 20.0)
    saturation = _clip01(soft_clip * 0.6 + max(0.0, high - 0.08) * 2.0)
    alias = _clip01(flatness * high * 3.0)
    med = float(np.median(rms)) if rms.size else 0.0
    dropout = _clip01(float(np.mean(rms < med * 0.22)) * 2.0) if med > 0 else 0.0
    slow = _rms_frames(x, rate, frame_sec=0.5)
    wow = _clip01(float(np.std(slow) / (np.mean(slow) + _EPS)) * 0.5) if slow.size else 0.0
    noise_floor = float(np.percentile(rms, 10)) if rms.size else 0.0
    noise_db = 20.0 * np.log10(max(noise_floor, 1e-9))
    return {
        "hiss_level": round(hiss, 4),
        "hum_50_60_level": round(_clip01(hum), 4),
        "click_pop_rate_per_sec": round(clicks / duration, 4),
        "saturation_proxy": round(saturation, 4),
        "soft_clip_proxy": round(soft_clip, 4),
        "bitcrush_alias_proxy": round(alias, 4),
        "dropout_proxy": round(dropout, 4),
        "wow_flutter_proxy": round(wow, 4),
        "noise_floor_dbfs_est": round(noise_db, 2),
    }


def _family(label: str, confidence: float) -> dict:
    return {"label": label, "confidence": round(_clip01(confidence), 4)}


def analyze(samples: Any, rate: int, rich: Mapping[str, Any] | None = None,
            rhythm: Mapping[str, Any] | None = None) -> tuple[dict, dict]:
    x = np.asarray(samples, dtype=np.float64)
    mag, freqs = _spectrum(x, rate)
    total = float(mag.sum() + _EPS)
    low = _band_energy(mag, freqs, 20.0, 180.0) / total
    low_mid = _band_energy(mag, freqs, 180.0, 900.0) / total
    mid = _band_energy(mag, freqs, 900.0, 2800.0) / total
    high = _band_energy(mag, freqs, 2800.0, rate / 2.0) / total
    flat = float((rich or {}).get("spectral_flatness") or 0.0)
    centroid = float((rich or {}).get("spectral_centroid_hz") or 0.0)
    hpss = (rich or {}).get("hpss") or {}
    harmonic = float(hpss.get("harmonic_ratio") or 0.5)
    percussive = float(hpss.get("percussive_ratio") or 0.5)
    attack = _attack_profile(x, rate, rhythm)
    lofi = _lofi(x, rate, mag, freqs, rich)
    texture = {
        "airiness": round(_clip01(high * 2.4 + flat * 0.2), 4),
        "muddiness": round(_clip01(low_mid * 1.8 + low * 0.5), 4),
        "harshness": round(_clip01(max(0.0, centroid - 2500.0) / 4500.0 + high * 0.8), 4),
        "warmth_proxy": round(_clip01(low_mid * 1.5 + low * 0.9 - high * 0.4), 4),
    }
    families = [
        _family("warm_pad", harmonic * attack["sustain_ratio"] * (low_mid + 0.35)),
        _family("soft_bell_keys", harmonic * mid * (1.0 - attack["transient_sharpness"] * 0.45) * 1.8),
        _family("plucky_keys", harmonic * attack["transient_sharpness"] * mid * 2.2),
        _family("sub_bass", low * 4.0),
        _family("round_bass", (low + low_mid) * 1.4 * (1.0 - texture["harshness"])),
        _family("airy_noise", texture["airiness"] * flat),
        _family("vinyl_air", lofi["hiss_level"] * (1.0 - lofi["bitcrush_alias_proxy"] * 0.5)),
        _family("percussive_ticks", percussive * attack["transient_sharpness"]),
        _family("glitch_sparks", lofi["click_pop_rate_per_sec"] / 6.0 + lofi["bitcrush_alias_proxy"] * 0.5),
        _family("drone", harmonic * attack["sustain_ratio"] * (1.0 - float(((rhythm or {}).get("density") or {}).get("onset_rate_per_sec") or 0.0) / 6.0)),
        _family("fm_console_tone", harmonic * mid * (1.0 - flat) * 1.1),
        _family("bitcrushed_texture", lofi["bitcrush_alias_proxy"]),
    ]
    families = sorted((f for f in families if f["confidence"] >= 0.05),
                      key=lambda f: f["confidence"], reverse=True)[:6]
    return {
        "families": families,
        "attack_profile": attack,
        "spectral_texture": texture,
    }, lofi
