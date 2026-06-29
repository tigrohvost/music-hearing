"""Harmony ambiguity and low-confidence chord hints for music-hearing v2."""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np

_EPS = 1e-10
_KEYS = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
_QUALITIES = {
    "major": (0, 4, 7),
    "minor": (0, 3, 7),
    "sus2": (0, 2, 7),
    "sus4": (0, 5, 7),
    "dim": (0, 3, 6),
    "maj7": (0, 4, 7, 11),
    "min7": (0, 3, 7, 10),
    "dom7": (0, 4, 7, 10),
    "add9": (0, 4, 7, 2),
    "minor_add9": (0, 3, 7, 2),
}


def _clip01(x: float) -> float:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return 0.0
    if not np.isfinite(f):
        return 0.0
    return float(max(0.0, min(1.0, f)))


def _stft_avg(seg: np.ndarray, rate: int) -> tuple[np.ndarray, np.ndarray]:
    n = max(2048, int(2 ** np.ceil(np.log2(max(32, seg.size)))))
    n = min(n, 32768)
    if seg.size < n:
        seg = np.pad(seg, (0, n - seg.size))
    mag = np.abs(np.fft.rfft(seg[:n] * np.hanning(n)))
    freqs = np.fft.rfftfreq(n, 1.0 / rate)
    return mag, freqs


def chroma(samples: Any, rate: int) -> np.ndarray:
    x = np.asarray(samples, dtype=np.float64)
    mag, freqs = _stft_avg(x, rate)
    mask = (freqs > 45.0) & (freqs < 5000.0)
    midi = 69.0 + 12.0 * np.log2(np.maximum(freqs[mask], 1.0) / 440.0)
    pc = np.mod(np.round(midi).astype(int), 12)
    out = np.zeros(12, dtype=np.float64)
    np.add.at(out, pc, mag[mask])
    if out.sum() > 0:
        out /= out.sum()
    return out


def _key_scores(c: np.ndarray) -> list[dict]:
    cn = c - c.mean()
    raw = []
    for tonic in range(12):
        for prof, mode in ((_MAJOR, "major"), (_MINOR, "minor")):
            p = np.roll(prof, tonic)
            p = p - p.mean()
            corr = float((cn * p).sum() / (np.sqrt((cn ** 2).sum() * (p ** 2).sum()) + _EPS))
            raw.append((corr, tonic, mode))
    raw.sort(reverse=True, key=lambda x: x[0])
    return [
        {"key": _KEYS[t], "mode": m, "score": round(_clip01((corr + 1.0) / 2.0), 4)}
        for corr, t, m in raw[:5]
    ]


def _entropy(c: np.ndarray) -> float:
    p = c / (c.sum() + _EPS)
    h = -float((p * np.log2(p + _EPS)).sum())
    return _clip01(h / np.log2(12.0))


def _window_chromas(samples: np.ndarray, rate: int, window_sec: float = 4.0) -> list[tuple[float, float, np.ndarray]]:
    n = max(1, int(rate * window_sec))
    out = []
    for start in range(0, samples.size, n):
        seg = samples[start:start + n]
        if seg.size < n // 2:
            break
        out.append((start / rate, min(samples.size, start + n) / rate, chroma(seg, rate)))
    return out


def _template(root: int, quality: str) -> np.ndarray:
    vec = np.zeros(12, dtype=np.float64)
    for pc in _QUALITIES[quality]:
        vec[(root + pc) % 12] = 1.0
    vec /= vec.sum()
    return vec


def _best_chord(c: np.ndarray) -> tuple[int, str, float]:
    best = (0, "major", -1.0)
    cn = c / (c.sum() + _EPS)
    for root in range(12):
        for quality in _QUALITIES:
            tmpl = _template(root, quality)
            score = float((cn * tmpl).sum() / (np.linalg.norm(cn) * np.linalg.norm(tmpl) + _EPS))
            if score > best[2]:
                best = (root, quality, score)
    return best[0], best[1], _clip01((best[2] - 0.45) / 0.45)


def _roman(root: int, quality: str, key_root: int, mode: str) -> str:
    rel = (root - key_root) % 12
    table_major = {0: "I", 1: "bII", 2: "II", 3: "bIII", 4: "III", 5: "IV",
                   6: "bV", 7: "V", 8: "bVI", 9: "VI", 10: "bVII", 11: "VII"}
    table_minor = {0: "i", 1: "bII", 2: "ii", 3: "bIII", 4: "III", 5: "iv",
                   6: "bV", 7: "v", 8: "bVI", 9: "VI", 10: "bVII", 11: "VII"}
    base = (table_major if mode == "major" else table_minor).get(rel, "?")
    suffix = {
        "minor": "" if base.islower() else "m",
        "major": "" if base.isupper() else "maj",
        "maj7": "maj7",
        "min7": "m7",
        "dom7": "7",
        "add9": "(add9)",
        "minor_add9": "m(add9)",
        "sus2": "sus2",
        "sus4": "sus4",
        "dim": "dim",
    }.get(quality, "")
    return base + suffix


def _progression(chords: list[dict]) -> dict:
    if not chords:
        return {"family": None, "confidence": 0.0}
    romans = [c["roman_hint"] for c in chords[:8] if c.get("confidence", 0.0) >= 0.25]
    if not romans:
        return {"family": None, "confidence": 0.0}
    comp = []
    for r in romans:
        if not comp or comp[-1] != r:
            comp.append(r)
    conf = sum(float(c.get("confidence", 0.0)) for c in chords[:len(romans)]) / max(1, len(romans))
    return {"family": "-".join(comp[:8]), "confidence": round(_clip01(conf), 4)}


def analyze(samples: Any, rate: int, rich: Mapping[str, Any] | None = None,
            max_chord_hints: int = 32) -> dict:
    x = np.asarray(samples, dtype=np.float64)
    c = np.asarray((rich or {}).get("chroma") or chroma(x, rate), dtype=np.float64)
    if c.size != 12:
        c = chroma(x, rate)
    keys = _key_scores(c)
    best = keys[0] if keys else {"key": None, "mode": None, "score": 0.0}
    second = keys[1]["score"] if len(keys) > 1 else 0.0
    confidence = float(best.get("score") or 0.0)
    entropy = _entropy(c)
    tonal_stability = _clip01(confidence * (1.0 - entropy * 0.45))
    key_root = _KEYS.index(best["key"]) if best.get("key") in _KEYS else 0
    chord_hints = []
    harmonic_ratio = ((rich or {}).get("hpss") or {}).get("harmonic_ratio", 0.5) if isinstance(rich, Mapping) else 0.5
    if confidence >= 0.42 and float(harmonic_ratio or 0.5) >= 0.35:
        for start, end, wc in _window_chromas(x, rate):
            root, quality, score = _best_chord(wc)
            if score < 0.22:
                continue
            chord_hints.append({
                "start_sec": round(start, 3),
                "end_sec": round(end, 3),
                "root": _KEYS[root],
                "quality": quality,
                "confidence": round(score, 4),
                "roman_hint": _roman(root, quality, key_root, str(best.get("mode") or "major")),
            })
            if len(chord_hints) >= max(1, int(max_chord_hints)):
                break
    return {
        "key": best.get("key"),
        "mode": best.get("mode"),
        "key_confidence": round(_clip01(confidence), 4),
        "key_ambiguity": round(_clip01(1.0 - max(0.0, confidence - second)), 4),
        "chroma_entropy": round(entropy, 4),
        "tonal_stability": round(tonal_stability, 4),
        "top_keys": keys[:5],
        "chord_hints": chord_hints[:max_chord_hints],
        "progression_hint": _progression(chord_hints),
    }
