"""Compact handcrafted music embeddings and profile comparison."""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping

EMBEDDING_SCHEMA = "music_embedding.handcrafted.v1"
SIMILARITY_SCHEMA = "music_similarity.v1"
DIM = 64
_FAMILIES = (
    "warm_pad", "soft_bell_keys", "plucky_keys", "sub_bass", "round_bass",
    "airy_noise", "vinyl_air", "percussive_ticks", "glitch_sparks", "drone",
    "fm_console_tone", "bitcrushed_texture",
)


def _f(obj: Mapping[str, Any] | None, key: str, default: float = 0.0) -> float:
    try:
        value = (obj or {}).get(key)
        f = float(value) if value is not None else default
    except (TypeError, ValueError):
        return default
    return f if math.isfinite(f) else default


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, float(x))) if math.isfinite(float(x)) else 0.0


def _norm(v: float, center: float, spread: float) -> float:
    return max(-1.0, min(1.0, (float(v) - center) / max(spread, 1e-6)))


def _pad(values: list[float]) -> list[float]:
    values = [round(float(v), 6) if math.isfinite(float(v)) else 0.0 for v in values]
    if len(values) < DIM:
        values.extend([0.0] * (DIM - len(values)))
    return values[:DIM]


def embedding_from_parts(base: Mapping[str, Any], rich: Mapping[str, Any] | None,
                         rhythm: Mapping[str, Any] | None,
                         structure: Mapping[str, Any] | None,
                         harmony: Mapping[str, Any] | None,
                         timbre: Mapping[str, Any] | None,
                         lofi: Mapping[str, Any] | None) -> dict:
    rhythm = rhythm or {}
    structure = structure or {}
    harmony = harmony or {}
    timbre = timbre or {}
    lofi = lofi or {}
    density = rhythm.get("density") or {}
    sync = rhythm.get("syncopation") or {}
    grid = rhythm.get("beat_grid") or {}
    arc = structure.get("arc") or {}
    texture = timbre.get("spectral_texture") or {}
    attack = timbre.get("attack_profile") or {}
    values = [
        _norm(_f(rhythm, "bpm", 80.0), 90.0, 70.0),
        _f(rhythm, "bpm_confidence"),
        _f(grid, "pulse_clarity"),
        _f(grid, "stability"),
        _norm(_f(grid, "local_tempo_std_bpm"), 5.0, 20.0),
        _f(grid, "swing_ratio", 0.5) * 2.0 - 1.0,
        _norm(_f(density, "onset_rate_per_sec"), 2.0, 5.0),
        _norm(_f(density, "low_band_onset_rate"), 0.4, 2.0),
        _norm(_f(density, "mid_band_onset_rate"), 0.8, 3.0),
        _norm(_f(density, "high_band_onset_rate"), 0.8, 3.0),
        _f(sync, "offbeat_energy_ratio") * 2.0 - 1.0,
        _f(sync, "syncopation_index"),
        _norm(_f(base, "spectral_centroid_hz", _f(rich or {}, "spectral_centroid_hz")), 1600.0, 2400.0),
        _norm(_f(base, "rms_dbfs", -18.0), -18.0, 18.0),
        _norm(_f(base, "crest_factor", 4.0), 4.0, 6.0),
        _norm(_f(base, "dynamic_range_db", 10.0), 10.0, 18.0),
    ]
    bands = base.get("bands") if isinstance(base.get("bands"), Mapping) else {}
    for name in ("sub_bass", "bass", "low_mid", "mid", "high"):
        values.append(float((bands or {}).get(name, 0.0)) * 2.0 - 1.0)
    values.extend([
        _f(arc, "energy_slope"),
        _f(arc, "brightness_slope"),
        _f(arc, "density_slope"),
        _f(arc, "loopiness") * 2.0 - 1.0,
        _f(arc, "arrangement_motion") * 2.0 - 1.0,
        _f(harmony, "key_confidence") * 2.0 - 1.0,
        _f(harmony, "key_ambiguity") * 2.0 - 1.0,
        _f(harmony, "chroma_entropy") * 2.0 - 1.0,
        _f(harmony, "tonal_stability") * 2.0 - 1.0,
    ])
    chroma = (rich or {}).get("chroma") if isinstance(rich, Mapping) else None
    if isinstance(chroma, list) and len(chroma) == 12:
        values.extend([float(v) * 2.0 - 1.0 for v in sorted(chroma, reverse=True)[:8]])
    else:
        values.extend([0.0] * 8)
    fam = {str(f.get("label")): float(f.get("confidence") or 0.0)
           for f in timbre.get("families", []) if isinstance(f, Mapping)}
    values.extend([fam.get(name, 0.0) * 2.0 - 1.0 for name in _FAMILIES])
    values.extend([
        _f(attack, "soft_attack_ratio") * 2.0 - 1.0,
        _f(attack, "transient_sharpness") * 2.0 - 1.0,
        _f(attack, "sustain_ratio") * 2.0 - 1.0,
        _f(texture, "airiness") * 2.0 - 1.0,
        _f(texture, "muddiness") * 2.0 - 1.0,
        _f(texture, "harshness") * 2.0 - 1.0,
        _f(texture, "warmth_proxy") * 2.0 - 1.0,
        _f(lofi, "hiss_level") * 2.0 - 1.0,
        _f(lofi, "hum_50_60_level") * 2.0 - 1.0,
        _norm(_f(lofi, "click_pop_rate_per_sec"), 0.2, 3.0),
        _f(lofi, "soft_clip_proxy") * 2.0 - 1.0,
        _f(lofi, "bitcrush_alias_proxy") * 2.0 - 1.0,
        _f(lofi, "dropout_proxy") * 2.0 - 1.0,
        _f(lofi, "wow_flutter_proxy") * 2.0 - 1.0,
    ])
    return {
        "schema": EMBEDDING_SCHEMA,
        "dim": DIM,
        "distance_space": "zscored_cosine_l2_hybrid",
        "values": _pad(values),
    }


def embedding_from_music_v2(music_v2: Mapping[str, Any] | None) -> dict | None:
    emb = (music_v2 or {}).get("embedding") if isinstance(music_v2, Mapping) else None
    if isinstance(emb, Mapping) and emb.get("schema") == EMBEDDING_SCHEMA and isinstance(emb.get("values"), list):
        return dict(emb)
    return None


def distance(a: Mapping[str, Any] | None, b: Mapping[str, Any] | None) -> dict:
    ea, eb = embedding_from_music_v2(a) or a, embedding_from_music_v2(b) or b
    if not isinstance(ea, Mapping) or not isinstance(eb, Mapping):
        return {"distance": None, "similarity": 0.0, "reason": "missing_embedding"}
    va, vb = ea.get("values"), eb.get("values")
    if ea.get("schema") != EMBEDDING_SCHEMA or eb.get("schema") != EMBEDDING_SCHEMA:
        return {"distance": None, "similarity": 0.0, "reason": "schema_mismatch"}
    if not isinstance(va, list) or not isinstance(vb, list) or not va or not vb:
        return {"distance": None, "similarity": 0.0, "reason": "missing_values"}
    n = min(len(va), len(vb), DIM)
    xa = [float(x or 0.0) for x in va[:n]]
    xb = [float(x or 0.0) for x in vb[:n]]
    l2 = math.sqrt(sum((x - y) ** 2 for x, y in zip(xa, xb)) / max(1, n))
    dot = sum(x * y for x, y in zip(xa, xb))
    na = math.sqrt(sum(x * x for x in xa))
    nb = math.sqrt(sum(x * x for x in xb))
    cosine_distance = 1.0 - (dot / (na * nb)) if na > 0 and nb > 0 else 1.0
    d = max(0.0, min(2.0, 0.55 * l2 + 0.45 * cosine_distance))
    return {"distance": round(d, 4), "similarity": round(max(0.0, 1.0 - d / 2.0), 4)}


def reference_id(label: str, embedding: Mapping[str, Any]) -> str:
    body = json.dumps({"label": label, "values": embedding.get("values", [])[:DIM]},
                      sort_keys=True, separators=(",", ":"))
    return "ref_" + hashlib.sha1(body.encode("utf-8")).hexdigest()[:16]


def make_reference_profile(music_v2: Mapping[str, Any], *, user_label: str = "",
                           source_kind: str = "unknown", created_at: str = "") -> dict:
    emb = embedding_from_music_v2(music_v2)
    if emb is None:
        raise ValueError("music_v2 embedding missing")
    harmony = music_v2.get("harmony") or {}
    timbre = music_v2.get("timbre") or {}
    rhythm = music_v2.get("rhythm") or {}
    families = [f.get("label") for f in timbre.get("families", []) if isinstance(f, Mapping)][:5]
    rid = reference_id(user_label or source_kind, emb)
    return {
        "schema": "music_reference_profile.v1",
        "reference_id": rid,
        "created_at": created_at,
        "source_kind": source_kind,
        "user_label": user_label,
        "excerpt_sec": (music_v2.get("excerpt") or {}).get("duration_sec"),
        "music_embedding": emb,
        "summary": {
            "bpm": rhythm.get("bpm"),
            "key": " ".join(str(x) for x in (harmony.get("key"), harmony.get("mode")) if x),
            "families": families,
        },
    }


def compare_to_references(music_v2: Mapping[str, Any], references: list[Mapping[str, Any]],
                          limit: int = 5) -> dict:
    emb = embedding_from_music_v2(music_v2)
    if emb is None:
        return {"schema": SIMILARITY_SCHEMA, "nearest": [], "novelty_against_recent": 1.0}
    nearest = []
    for ref in references:
        ref_emb = ref.get("music_embedding") if isinstance(ref, Mapping) else None
        d = distance(emb, ref_emb)
        if d.get("distance") is None:
            continue
        nearest.append({
            "reference_id": ref.get("reference_id") or "",
            "distance": d["distance"],
            "similarity": d["similarity"],
            "matched_axes": [],
        })
    nearest.sort(key=lambda x: x["distance"])
    best_sim = nearest[0]["similarity"] if nearest else 0.0
    return {
        "schema": SIMILARITY_SCHEMA,
        "nearest": nearest[:limit],
        "novelty_against_recent": round(max(0.0, 1.0 - best_sim), 4),
    }
