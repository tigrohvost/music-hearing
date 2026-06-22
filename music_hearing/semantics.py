"""Turn a numeric ``AcousticProfile`` into words, and compare two profiles.

``dsp`` emits floats (centroid, bands, bpm, crest...). This maps them to plain
language ("warm, slow, sub-heavy, dynamic") and gives a band-distance /
similarity so a caller can ask "is this track like that reference?". Pure
stdlib; thresholds are deliberate heuristics tuned for the 8 kHz profiler.
"""
from __future__ import annotations

from typing import Any, Mapping

_BANDS = ("sub_bass", "bass", "low_mid", "mid", "high")


def _f(profile: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    try:
        v = profile.get(key)
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _brightness(centroid: float) -> str:
    if centroid < 500:
        return "dark"
    if centroid < 1000:
        return "warm"
    if centroid < 1800:
        return "neutral"
    return "bright"


def _weight(bands: Mapping[str, float]) -> str:
    sub = float(bands.get("sub_bass", 0.0))
    bass = float(bands.get("bass", 0.0))
    low = sub + bass
    mid = float(bands.get("low_mid", 0.0)) + float(bands.get("mid", 0.0))
    hi = float(bands.get("high", 0.0))
    if low >= 0.5:
        return "sub-heavy" if sub >= bass else "bass-heavy"
    if hi >= mid and hi >= low:
        return "thin"
    if mid >= low:
        return "mid-forward"
    return "balanced"


def _tempo_feel(bpm) -> str:
    if bpm is None:
        return "unknown"
    bpm = float(bpm)
    if bpm < 60:
        return "still"
    if bpm < 90:
        return "slow"
    if bpm < 120:
        return "mid-tempo"
    return "up-tempo"


def _dynamics(crest: float, dr: float) -> str:
    if dr >= 18 or crest >= 6:
        return "dynamic"
    if dr >= 9:
        return "moderate"
    return "flat"


def _texture(zcr: float) -> str:
    if zcr >= 1500:
        return "noisy"
    if zcr >= 600:
        return "textured"
    return "tonal"


def describe(profile: Mapping[str, Any]) -> dict:
    """Map an AcousticProfile dict to semantic labels + a one-line summary."""
    bands = profile.get("bands") or {}
    brightness = _brightness(_f(profile, "spectral_centroid_hz"))
    weight = _weight(bands)
    tempo_feel = _tempo_feel(profile.get("estimated_bpm"))
    dynamics = _dynamics(_f(profile, "crest_factor"), _f(profile, "dynamic_range_db"))
    texture = _texture(_f(profile, "zero_crossing_rate"))
    parts = [p for p in (tempo_feel if tempo_feel != "unknown" else None,
                         brightness, weight, dynamics, texture) if p]
    summary = ", ".join(parts)
    bpm = profile.get("estimated_bpm")
    if bpm is not None:
        summary += f" (~{round(float(bpm))} BPM)"
    return {
        "brightness": brightness,
        "weight": weight,
        "tempo_feel": tempo_feel,
        "dynamics": dynamics,
        "texture": texture,
        "tempo_bpm": (float(bpm) if bpm is not None else None),
        "summary": summary,
    }


def compare(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict:
    """Distance between two profiles. ``band_distance`` is L1/2 over the band
    ratios (0..1); ``similarity`` = 1 - band_distance. bpm/centroid/rms diffs
    are absolute; bpm_diff is None when either bpm is unknown."""
    ba, bb = a.get("bands") or {}, b.get("bands") or {}
    band_distance = sum(abs(float(ba.get(k, 0.0)) - float(bb.get(k, 0.0))) for k in _BANDS) / 2.0
    band_distance = round(max(0.0, min(1.0, band_distance)), 4)
    bpm_a, bpm_b = a.get("estimated_bpm"), b.get("estimated_bpm")
    bpm_diff = abs(float(bpm_a) - float(bpm_b)) if (bpm_a is not None and bpm_b is not None) else None
    return {
        "band_distance": band_distance,
        "similarity": round(1.0 - band_distance, 4),
        "centroid_hz_diff": round(abs(_f(a, "spectral_centroid_hz") - _f(b, "spectral_centroid_hz")), 1),
        "bpm_diff": bpm_diff,
        "rms_db_diff": round(abs(_f(a, "rms_dbfs") - _f(b, "rms_dbfs")), 2),
    }
