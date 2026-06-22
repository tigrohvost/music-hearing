"""Dependency-light acoustic profiler.

Reads an audio file and emits a small numeric ``AcousticProfile``: loudness,
crest/dynamic range, zero-crossing rate, a 5-band Goertzel sketch, a coarse
spectral centroid and an autocorrelation BPM. Runs at 8 kHz mono. MP3/other
formats use the local ``ffmpeg`` binary; WAV fixtures use only the standard
library. Observational only — it decodes and measures, nothing else.
"""
from __future__ import annotations

import math
import pathlib
import shutil
import struct
import subprocess
import wave
from dataclasses import dataclass
from typing import Iterable

DEFAULT_RATE = 8000
FRAME_SECONDS = 0.10
MAX_SECONDS = 45.0
BAND_FREQS = {
    "sub_bass": (55.0, 70.0, 90.0),
    "bass": (120.0, 170.0, 230.0),
    "low_mid": (320.0, 440.0, 560.0),
    "mid": (850.0, 1100.0, 1400.0),
    "high": (2200.0, 2800.0, 3400.0),
}


@dataclass(frozen=True)
class AcousticProfile:
    path: str
    sample_rate: int
    seconds_analyzed: float
    rms_dbfs: float
    peak: float
    crest_factor: float
    dynamic_range_db: float
    zero_crossing_rate: float
    spectral_centroid_hz: float
    bands: dict[str, float]
    estimated_bpm: float | None


def _clamp_unit(x: float) -> float:
    return max(0.0, min(1.0, x))


def _dbfs(x: float) -> float:
    if x <= 0.0:
        return -120.0
    return 20.0 * math.log10(min(1.0, x))


def _read_wav_mono(path: str | pathlib.Path, target_rate: int) -> tuple[list[float], int]:
    with wave.open(str(path), "rb") as w:
        channels = w.getnchannels()
        width = w.getsampwidth()
        rate = w.getframerate()
        raw = w.readframes(w.getnframes())
    if width != 2:
        raise ValueError("only 16-bit PCM WAV is supported without ffmpeg")
    vals = struct.unpack("<" + "h" * (len(raw) // 2), raw)
    mono = []
    if channels == 1:
        mono = [v / 32768.0 for v in vals]
    else:
        for i in range(0, len(vals), channels):
            mono.append(sum(vals[i:i + channels]) / (32768.0 * channels))
    if rate == target_rate:
        return mono, rate
    return _resample_nearest(mono, rate, target_rate), target_rate


def _decode_with_ffmpeg(path: str | pathlib.Path, target_rate: int, max_seconds: float) -> tuple[list[float], int]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to profile non-WAV audio")
    cmd = [
        ffmpeg, "-v", "error", "-t", str(max_seconds), "-i", str(path),
        "-ac", "1", "-ar", str(target_rate), "-f", "s16le", "-",
    ]
    proc = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    vals = struct.unpack("<" + "h" * (len(proc.stdout) // 2), proc.stdout)
    return [v / 32768.0 for v in vals], target_rate


def _resample_nearest(samples: list[float], src_rate: int, dst_rate: int) -> list[float]:
    if not samples:
        return []
    n = max(1, int(len(samples) * dst_rate / src_rate))
    return [samples[min(len(samples) - 1, int(i * src_rate / dst_rate))] for i in range(n)]


def load_mono_samples(path: str | pathlib.Path, target_rate: int = DEFAULT_RATE, max_seconds: float = MAX_SECONDS) -> tuple[list[float], int]:
    p = pathlib.Path(path)
    if p.suffix.lower() == ".wav":
        samples, rate = _read_wav_mono(p, target_rate)
        return samples[: int(rate * max_seconds)], rate
    return _decode_with_ffmpeg(p, target_rate, max_seconds)


def _frames(samples: list[float], rate: int, frame_seconds: float = FRAME_SECONDS) -> list[list[float]]:
    n = max(1, int(rate * frame_seconds))
    return [samples[i:i + n] for i in range(0, len(samples), n) if len(samples[i:i + n]) >= n // 2]


def _rms(xs: Iterable[float]) -> float:
    vals = list(xs)
    if not vals:
        return 0.0
    return math.sqrt(sum(x * x for x in vals) / len(vals))


def _zero_crossing_rate(samples: list[float], rate: int) -> float:
    if len(samples) < 2:
        return 0.0
    crossings = sum(1 for a, b in zip(samples, samples[1:]) if (a < 0 <= b) or (a >= 0 > b))
    return crossings / (len(samples) / rate)


def _goertzel_power(samples: list[float], rate: int, freq: float) -> float:
    if not samples:
        return 0.0
    w = 2.0 * math.pi * freq / rate
    coeff = 2.0 * math.cos(w)
    s0 = s1 = s2 = 0.0
    for x in samples:
        s0 = x + coeff * s1 - s2
        s2 = s1
        s1 = s0
    return max(0.0, s1 * s1 + s2 * s2 - coeff * s1 * s2)


def _band_profile(samples: list[float], rate: int) -> tuple[dict[str, float], float]:
    energies = {}
    weighted = total = 0.0
    for band, freqs in BAND_FREQS.items():
        e = sum(_goertzel_power(samples, rate, f) for f in freqs)
        energies[band] = e
        center = sum(freqs) / len(freqs)
        weighted += center * e
        total += e
    if total <= 0.0:
        return {k: 0.0 for k in BAND_FREQS}, 0.0
    return {k: round(v / total, 4) for k, v in energies.items()}, round(weighted / total, 1)


def _estimate_bpm(frames: list[list[float]]) -> float | None:
    if len(frames) < 20:
        return None
    env = [_rms(f) for f in frames]
    mean = sum(env) / len(env)
    env = [max(0.0, x - mean) for x in env]
    if not any(env):
        return None
    best_lag = None
    best_score = 0.0
    frame_hz = 1.0 / FRAME_SECONDS
    min_lag = max(1, int(frame_hz * 60.0 / 160.0))
    max_lag = max(min_lag + 1, int(frame_hz * 60.0 / 55.0))
    for lag in range(min_lag, min(max_lag, len(env) // 2)):
        score = sum(a * b for a, b in zip(env, env[lag:]))
        if score > best_score:
            best_score = score
            best_lag = lag
    if best_lag is None or best_score <= 0.0:
        return None
    return round(60.0 / (best_lag * FRAME_SECONDS), 1)


def acoustic_profile(path: str | pathlib.Path, target_rate: int = DEFAULT_RATE, max_seconds: float = MAX_SECONDS) -> AcousticProfile:
    samples, rate = load_mono_samples(path, target_rate=target_rate, max_seconds=max_seconds)
    fr = _frames(samples, rate)
    frame_rms = [_rms(f) for f in fr]
    rms = _rms(samples)
    peak = max((abs(x) for x in samples), default=0.0)
    quiet = sorted(frame_rms)[max(0, int(len(frame_rms) * 0.1) - 1)] if frame_rms else 0.0
    loud = sorted(frame_rms)[min(len(frame_rms) - 1, int(len(frame_rms) * 0.9))] if frame_rms else 0.0
    bands, centroid = _band_profile(samples, rate)
    return AcousticProfile(
        path=str(path),
        sample_rate=rate,
        seconds_analyzed=round(len(samples) / rate, 2),
        rms_dbfs=round(_dbfs(rms), 2),
        peak=round(_clamp_unit(peak), 4),
        crest_factor=round((peak / rms) if rms else 0.0, 3),
        dynamic_range_db=round(_dbfs(loud) - _dbfs(quiet), 2),
        zero_crossing_rate=round(_zero_crossing_rate(samples, rate), 2),
        spectral_centroid_hz=centroid,
        bands=bands,
        estimated_bpm=_estimate_bpm(fr),
    )
