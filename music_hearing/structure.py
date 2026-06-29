"""Arrangement and section-shape proxies for music-hearing v2."""
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


def _centroid(seg: np.ndarray, rate: int) -> float:
    if seg.size == 0:
        return 0.0
    mag = np.abs(np.fft.rfft(seg * np.hanning(seg.size)))
    freqs = np.fft.rfftfreq(seg.size, 1.0 / rate)
    return float((freqs * mag).sum() / (mag.sum() + _EPS))


def _frame_features(samples: np.ndarray, rate: int, hop_sec: float = 1.0) -> tuple[np.ndarray, list[dict]]:
    hop = max(1, int(rate * hop_sec))
    rows = []
    dicts = []
    prev_mag = None
    for start in range(0, samples.size, hop):
        seg = samples[start:start + hop]
        if seg.size < hop // 2:
            break
        rms = float(np.sqrt(np.mean(seg ** 2)))
        centroid = _centroid(seg, rate)
        mag = np.abs(np.fft.rfft(seg * np.hanning(seg.size)))
        freqs = np.fft.rfftfreq(seg.size, 1.0 / rate)
        low = float(mag[freqs < 250].sum())
        mid = float(mag[(freqs >= 250) & (freqs < 2000)].sum())
        high = float(mag[freqs >= 2000].sum())
        total = low + mid + high + _EPS
        flux = float(np.maximum(mag - prev_mag, 0.0).sum() / (mag.sum() + _EPS)) if prev_mag is not None and prev_mag.size == mag.size else 0.0
        prev_mag = mag
        row = [rms, centroid / 5000.0, low / total, mid / total, high / total, flux]
        rows.append(row)
        dicts.append({
            "t": start / rate,
            "energy": _clip01((20.0 * np.log10(max(rms, 1e-9)) + 42.0) / 36.0),
            "brightness": _clip01(centroid / 5000.0),
            "density": _clip01(flux * 4.0),
        })
    return np.asarray(rows, dtype=np.float64), dicts


def _novelty(features: np.ndarray) -> np.ndarray:
    if features.shape[0] < 2:
        return np.zeros(features.shape[0], dtype=np.float64)
    raw_std = np.std(features, axis=0)
    if float(np.max(raw_std)) < 1e-5:
        return np.zeros(features.shape[0], dtype=np.float64)
    norm = (features - np.mean(features, axis=0)) / (raw_std + _EPS)
    diff = np.linalg.norm(np.diff(norm, axis=0), axis=1)
    out = np.concatenate([[0.0], diff])
    if out.max() > 0:
        out = out / out.max()
    if out.size >= 5:
        out = np.convolve(out, np.ones(3) / 3.0, mode="same")
    return out


def _boundaries(novelty: np.ndarray, duration: float, min_section_sec: float = 6.0) -> list[float]:
    if novelty.size < 4:
        return [0.0, duration]
    threshold = float(novelty.mean() + 0.75 * novelty.std())
    peaks = []
    for i in range(1, novelty.size - 1):
        if novelty[i] >= threshold and novelty[i] >= novelty[i - 1] and novelty[i] >= novelty[i + 1]:
            t = float(i)
            if not peaks or t - peaks[-1] >= min_section_sec:
                peaks.append(t)
    points = [0.0] + [p for p in peaks if min_section_sec <= p <= duration - min_section_sec / 2.0] + [duration]
    merged = [points[0]]
    for p in points[1:]:
        if p - merged[-1] < min_section_sec and p != duration:
            continue
        merged.append(p)
    if merged[-1] != duration:
        merged[-1] = duration
    return merged[:17]


def _section_label(section_vec: np.ndarray, prior: list[np.ndarray]) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for idx, prev in enumerate(prior):
        denom = np.linalg.norm(section_vec) * np.linalg.norm(prev) + _EPS
        sim = float((section_vec * prev).sum() / denom)
        if sim > 0.92:
            return letters[idx] + "'"
        if sim > 0.82:
            return letters[idx]
    return letters[min(len(prior), len(letters) - 1)]


def analyze(samples: Any, rate: int, rich: Mapping[str, Any] | None = None,
            rhythm: Mapping[str, Any] | None = None) -> dict:
    x = np.asarray(samples, dtype=np.float64)
    duration = max(float(x.size) / float(rate or 1), 0.0)
    features, frames = _frame_features(x, rate)
    novelty = _novelty(features)
    points = _boundaries(novelty, duration)
    sections = []
    priors: list[np.ndarray] = []
    for a, b in zip(points, points[1:]):
        ia, ib = int(max(0, np.floor(a))), int(max(1, np.ceil(b)))
        rows = features[ia:min(ib, features.shape[0])] if features.size else np.zeros((1, 6))
        vec = rows.mean(axis=0) if rows.size else np.zeros(6)
        label = _section_label(vec, priors)
        if not label.endswith("'"):
            priors.append(vec)
        fdicts = frames[ia:min(ib, len(frames))] or [{"energy": 0.0, "brightness": 0.0, "density": 0.0}]
        sections.append({
            "id": label,
            "start_sec": round(a, 3),
            "end_sec": round(b, 3),
            "energy": round(float(np.mean([f["energy"] for f in fdicts])), 4),
            "brightness": round(float(np.mean([f["brightness"] for f in fdicts])), 4),
            "density": round(float(np.mean([f["density"] for f in fdicts])), 4),
        })
        if len(sections) >= 16:
            break
    if features.shape[0] >= 2:
        t = np.arange(features.shape[0], dtype=np.float64)
        denom = float(max(1, features.shape[0] - 1))
        energy_slope = float(np.polyfit(t / denom, [f["energy"] for f in frames], 1)[0])
        brightness_slope = float(np.polyfit(t / denom, [f["brightness"] for f in frames], 1)[0])
        density_slope = float(np.polyfit(t / denom, [f["density"] for f in frames], 1)[0])
    else:
        energy_slope = brightness_slope = density_slope = 0.0
    first = features[:max(1, features.shape[0] // 5)].mean(axis=0) if features.size else np.zeros(6)
    last = features[-max(1, features.shape[0] // 5):].mean(axis=0) if features.size else np.zeros(6)
    sim = float((first * last).sum() / (np.linalg.norm(first) * np.linalg.norm(last) + _EPS))
    motion = float(np.mean(novelty)) if novelty.size else 0.0
    values = [round(float(v), 4) for v in novelty[:128]]
    return {
        "section_count": len(sections),
        "sections": sections,
        "novelty": {"hop_sec": 1.0, "values_max_128": values},
        "arc": {
            "energy_slope": round(energy_slope, 4),
            "brightness_slope": round(brightness_slope, 4),
            "density_slope": round(density_slope, 4),
            "has_intro": bool(sections and sections[0]["energy"] < np.mean([s["energy"] for s in sections]) - 0.05),
            "has_outro": bool(sections and sections[-1]["energy"] < np.mean([s["energy"] for s in sections]) - 0.05),
            "loopiness": round(_clip01((sim + 1.0) / 2.0), 4),
            "arrangement_motion": round(_clip01(motion), 4),
        },
    }
