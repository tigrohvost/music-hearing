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


def describe_music_v2(music_v2: Mapping[str, Any]) -> dict:
    """Plain-language musical labels from the additive ``music_v2`` profile."""
    rhythm = music_v2.get("rhythm") if isinstance(music_v2.get("rhythm"), Mapping) else {}
    structure = music_v2.get("structure") if isinstance(music_v2.get("structure"), Mapping) else {}
    harmony = music_v2.get("harmony") if isinstance(music_v2.get("harmony"), Mapping) else {}
    timbre = music_v2.get("timbre") if isinstance(music_v2.get("timbre"), Mapping) else {}
    lofi = music_v2.get("lofi") if isinstance(music_v2.get("lofi"), Mapping) else {}
    density = rhythm.get("density") if isinstance(rhythm.get("density"), Mapping) else {}
    grid = rhythm.get("beat_grid") if isinstance(rhythm.get("beat_grid"), Mapping) else {}
    arc = structure.get("arc") if isinstance(structure.get("arc"), Mapping) else {}
    onset = _f(density, "onset_rate_per_sec")
    pulse = _f(grid, "pulse_clarity")
    motion = _f(arc, "arrangement_motion")
    families = [f.get("label") for f in timbre.get("families", []) if isinstance(f, Mapping)][:3]
    groove = "stable" if pulse >= 0.65 else "loose" if pulse >= 0.3 else "blurred"
    rhythm_density = "dense" if onset >= 3.0 else "active" if onset >= 1.2 else "sparse"
    arrangement = "evolving" if motion >= 0.45 else "gently moving" if motion >= 0.18 else "loop-like"
    key = " ".join(str(x) for x in (harmony.get("key"), harmony.get("mode")) if x) or "ambiguous"
    artifacts = []
    if _f(lofi, "hiss_level") >= 0.35:
        artifacts.append("hiss")
    if _f(lofi, "click_pop_rate_per_sec") >= 0.2:
        artifacts.append("clicks")
    if _f(lofi, "wow_flutter_proxy") >= 0.25:
        artifacts.append("wow/flutter")
    parts = [groove + " pulse", rhythm_density + " rhythm", arrangement, key]
    if families:
        parts.append("/".join(str(f) for f in families))
    if artifacts:
        parts.append("lofi " + "+".join(artifacts))
    return {
        "groove": groove,
        "rhythm_density": rhythm_density,
        "arrangement": arrangement,
        "key_hint": key,
        "timbre_families": families,
        "lofi_artifacts": artifacts,
        "summary": ", ".join(parts),
    }
