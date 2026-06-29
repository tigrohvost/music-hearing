"""Versioned additive music-hearing v2 contract builder.

The legacy top-level profile remains unchanged. This module emits a bounded
``music_v2`` block only when a caller explicitly requests it or enables the
compatibility environment flag.
"""
from __future__ import annotations

import math
import os
import threading
import time
from pathlib import Path
from typing import Any, Mapping

MUSIC_V2_SCHEMA = "music_hearing_profile.v2"
DEFAULT_RATE = 22050
MAX_SECONDS = 90.0
MAX_ARRAY = 128
MAX_CHORD_HINTS = 32

_ANALYSIS_LOCK = threading.Lock()


def env_enabled(name: str | None = None) -> bool:
    names = (name,) if name else ("MH_MUSIC_V2",)
    return any((os.environ.get(n) or "").strip().lower() in {"1", "true", "yes", "on"} for n in names)


def requested(value: Any = None) -> bool:
    if value is None:
        return env_enabled()
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "music", "v2"}
    return bool(value)


def _clip01(value: Any, default: float = 0.0) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(f):
        return default
    return 0.0 if f < 0.0 else 1.0 if f > 1.0 else f


def _sanitize(obj: Any, *, max_array: int = MAX_ARRAY) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _sanitize(v, max_array=max_array) for k, v in obj.items()}
    if isinstance(obj, tuple):
        obj = list(obj)
    if isinstance(obj, list):
        return [_sanitize(v, max_array=max_array) for v in obj[:max_array]]
    if isinstance(obj, float):
        return round(obj, 6) if math.isfinite(obj) else None
    if isinstance(obj, int) or obj is None or isinstance(obj, bool) or isinstance(obj, str):
        return obj
    try:
        f = float(obj)
    except (TypeError, ValueError):
        return str(obj)
    return round(f, 6) if math.isfinite(f) else None


def empty_music_v2(*, duration_sec: float = 0.0, sample_rate_hz: int = DEFAULT_RATE,
                   warnings: list[str] | None = None, runtime_sec: float = 0.0) -> dict:
    return {
        "schema": MUSIC_V2_SCHEMA,
        "enabled": True,
        "excerpt": {
            "duration_sec": round(float(duration_sec or 0.0), 3),
            "sample_rate_hz": int(sample_rate_hz or DEFAULT_RATE),
            "channels_analyzed": "mono",
            "start_sec": 0.0,
            "source_kind": "unknown",
        },
        "rhythm": None,
        "structure": None,
        "harmony": None,
        "timbre": None,
        "lofi": None,
        "embedding": None,
        "similarity": None,
        "quality_flags": {
            "analysis_confidence": 0.0,
            "warnings": list(warnings or ["music_v2_submodules_not_enabled"]),
            "limitations": ["mixture_level_timbre_only"],
        },
        "runtime": {"music_v2_sec": round(float(runtime_sec or 0.0), 3)},
    }


def _analysis_confidence(parts: Mapping[str, Any]) -> float:
    vals = []
    rhythm = parts.get("rhythm") or {}
    if isinstance(rhythm, Mapping):
        vals.append(_clip01(rhythm.get("bpm_confidence"), 0.0))
        vals.append(_clip01((rhythm.get("beat_grid") or {}).get("pulse_clarity"), 0.0))
    harmony = parts.get("harmony") or {}
    if isinstance(harmony, Mapping):
        vals.append(_clip01(harmony.get("key_confidence"), 0.0))
        vals.append(_clip01(harmony.get("tonal_stability"), 0.0))
    structure = parts.get("structure") or {}
    if isinstance(structure, Mapping):
        vals.append(_clip01((structure.get("arc") or {}).get("arrangement_motion"), 0.2))
    timbre = parts.get("timbre") or {}
    if isinstance(timbre, Mapping):
        fam = timbre.get("families") or []
        if fam:
            vals.append(max(_clip01((x or {}).get("confidence"), 0.0) for x in fam if isinstance(x, Mapping)))
    return round(sum(vals) / len(vals), 4) if vals else 0.0


def _quality_warnings(parts: Mapping[str, Any]) -> list[str]:
    warnings: list[str] = []
    harmony = parts.get("harmony") or {}
    if isinstance(harmony, Mapping):
        if _clip01(harmony.get("key_confidence"), 0.0) < 0.35:
            warnings.append("key_low_confidence")
        if harmony.get("chord_hints") and _clip01((harmony.get("progression_hint") or {}).get("confidence"), 0.0) < 0.45:
            warnings.append("chord_hints_low_confidence")
    rhythm = parts.get("rhythm") or {}
    if isinstance(rhythm, Mapping) and _clip01(rhythm.get("bpm_confidence"), 0.0) < 0.25:
        warnings.append("rhythm_low_confidence")
    return sorted(set(warnings))


def build_music_v2(path: str | Path, *, base_profile: Mapping[str, Any] | None = None,
                   rich_profile: Mapping[str, Any] | None = None,
                   seconds: float = 45.0, source_kind: str = "local",
                   sample_rate: int = DEFAULT_RATE) -> dict:
    """Compute the full v2 block from a bounded audio excerpt.

    The lock prevents accidental concurrent heavy analyses in one tool process.
    On failure, callers receive an empty v2 envelope with a warning instead of
    an exception.
    """
    t0 = time.monotonic()
    bounded_seconds = max(5.0, min(float(seconds or 45.0), MAX_SECONDS))
    if not _ANALYSIS_LOCK.acquire(blocking=False):
        return empty_music_v2(duration_sec=bounded_seconds, sample_rate_hz=sample_rate,
                              warnings=["music_v2_analysis_already_running"])
    try:
        import numpy as np

        from . import dsp, harmony, rhythm, similarity, structure, timbre

        samples, rate = dsp.load_mono_samples(path, target_rate=sample_rate, max_seconds=bounded_seconds)
        x = np.asarray(samples, dtype=np.float64)
        duration = round(float(x.size) / float(rate or sample_rate), 3)
        rich = dict(rich_profile or {})
        rhythm_data = rhythm.analyze(x, rate, rich=rich)
        structure_data = structure.analyze(x, rate, rich=rich, rhythm=rhythm_data)
        harmony_data = harmony.analyze(x, rate, rich=rich, max_chord_hints=MAX_CHORD_HINTS)
        timbre_data, lofi_data = timbre.analyze(x, rate, rich=rich, rhythm=rhythm_data)
        partial = {
            "rhythm": rhythm_data,
            "structure": structure_data,
            "harmony": harmony_data,
            "timbre": timbre_data,
            "lofi": lofi_data,
        }
        embedding = similarity.embedding_from_parts(
            base_profile or {}, rich, rhythm_data, structure_data, harmony_data, timbre_data, lofi_data
        )
        partial["embedding"] = embedding
        warnings = _quality_warnings(partial)
        out = {
            "schema": MUSIC_V2_SCHEMA,
            "enabled": True,
            "excerpt": {
                "duration_sec": duration,
                "sample_rate_hz": int(rate),
                "channels_analyzed": "mono_with_optional_stereo_width" if rich.get("stereo") else "mono",
                "start_sec": 0.0,
                "source_kind": source_kind,
            },
            "rhythm": rhythm_data,
            "structure": structure_data,
            "harmony": harmony_data,
            "timbre": timbre_data,
            "lofi": lofi_data,
            "embedding": embedding,
            "similarity": None,
            "quality_flags": {
                "analysis_confidence": _analysis_confidence(partial),
                "warnings": warnings,
                "limitations": ["mixture_level_timbre_only", "chord_hints_are_low_confidence"],
            },
            "runtime": {
                "music_v2_sec": round(time.monotonic() - t0, 3),
                "peak_memory_mb_est": None,
            },
        }
        return _sanitize(out)
    except Exception as exc:  # noqa: BLE001 - hearing should degrade, not crash callers
        return empty_music_v2(
            duration_sec=bounded_seconds,
            sample_rate_hz=sample_rate,
            warnings=[f"music_v2_error:{type(exc).__name__}"],
            runtime_sec=time.monotonic() - t0,
        )
    finally:
        _ANALYSIS_LOCK.release()
