"""Music-critic layer: turn objective acoustics + metadata into the inputs an
LLM needs to name genre, similar artists, and an impression.

Genre/artist/impression are world-knowledge judgments, so this module does not
pretend to derive them from numbers. Instead it (a) computes coarse genre
*hints* from the acoustics, (b) assembles a compact evidence *brief*, and (c)
builds a ready *prompt*. An agent feeds that to its own model; a standalone
caller can use :func:`llm_verdict` against any OpenAI-compatible endpoint.
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Mapping


def _uniq(seq):
    seen, out = set(), []
    for x in seq:
        x = x.strip().lower()
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def genre_hints(description: Mapping[str, Any], rich: Mapping[str, Any] | None = None,
                tags: list[str] | None = None, max_hints: int = 6) -> list[str]:
    """Coarse genre/style guesses from the acoustic description (+ rich
    features), with any metadata ``tags`` taking precedence. Hints, not verdicts
    — they ground an LLM, they don't replace it."""
    hints: list[str] = list(tags or [])
    tempo = (description.get("tempo_feel") or "").lower()
    weight = (description.get("weight") or "").lower()
    brightness = (description.get("brightness") or "").lower()
    texture = (description.get("texture") or "").lower()
    dynamics = (description.get("dynamics") or "").lower()

    perc = 0.5
    onset = 0.0
    if rich:
        hp = rich.get("hpss") or {}
        perc = float(hp.get("percussive_ratio", 0.5) or 0.5)
        onset = float(rich.get("onset_rate_hz", 0.0) or 0.0)

    slow = tempo in ("still", "slow")
    fast = tempo == "up-tempo"
    heavy_low = weight in ("sub-heavy", "bass-heavy")
    calm = perc < 0.45 or onset < 1.0
    busy = perc > 0.55 or onset > 2.0

    if slow and heavy_low and texture in ("tonal", "textured") and calm:
        hints += ["ambient", "downtempo"]
        if onset < 0.5:
            hints.append("drone")
    if tempo in ("slow", "mid-tempo") and brightness in ("dark", "warm") and texture != "noisy":
        hints += ["downtempo", "chillout", "lo-fi"]
    if fast and busy:
        hints += ["electronic", "dance"]
        hints.append("techno" if brightness in ("dark", "neutral") else "house")
    if texture == "noisy" and dynamics == "dynamic":
        hints += ["rock", "electronic"]
    if brightness == "bright" and weight in ("thin", "mid-forward") and texture == "tonal":
        hints += ["pop", "acoustic"]

    if not hints:
        hints += ["instrumental", "electronic" if busy else "ambient"]
    return _uniq(hints)[:max_hints]


def build_brief(description: Mapping[str, Any], rich: Mapping[str, Any] | None = None,
                metadata: Mapping[str, Any] | None = None) -> str:
    """Compact, model-readable evidence brief: acoustics + (rich) + metadata."""
    d = description or {}
    parts = [f"Acoustics: {d.get('summary', '')}".rstrip(". ") + "."]
    parts.append(
        f"Bands {d.get('weight', '?')}, brightness {d.get('brightness', '?')}, "
        f"texture {d.get('texture', '?')}, dynamics {d.get('dynamics', '?')}.")
    if rich:
        key = rich.get("key") or {}
        tempo = rich.get("tempo") or {}
        hp = rich.get("hpss") or {}
        bits = []
        if key.get("key"):
            bits.append(f"key ~{key.get('key')} {key.get('mode', '')}"
                        f" (conf {key.get('confidence')})")
        if tempo.get("bpm"):
            bits.append(f"tempo ~{tempo.get('bpm')} BPM")
        if hp:
            bits.append(f"harmonic/percussive {hp.get('harmonic_ratio')}/{hp.get('percussive_ratio')}")
        if rich.get("onset_rate_hz") is not None:
            bits.append(f"onset {rich.get('onset_rate_hz')}/s")
        if bits:
            parts.append("Spectral: " + ", ".join(bits) + ".")
    m = metadata or {}
    if any(m.get(k) for k in ("title", "artist", "album", "tags")):
        meta_bits = []
        if m.get("title"):
            meta_bits.append(f"title \"{m['title']}\"")
        if m.get("artist"):
            meta_bits.append(f"artist {m['artist']}")
        if m.get("album"):
            meta_bits.append(f"album {m['album']}")
        if m.get("tags"):
            meta_bits.append("tags " + ", ".join(m["tags"][:8]))
        parts.append("Metadata: " + "; ".join(meta_bits) + ".")
    return " ".join(parts)


def build_prompt(brief: str, metadata: Mapping[str, Any] | None = None) -> str:
    """A critic prompt for any LLM, embedding the evidence brief."""
    return (
        "You are a seasoned, honest music critic. Using ONLY the acoustic "
        "analysis and metadata below, respond with:\n"
        "1) the most likely genre and subgenre;\n"
        "2) 2-4 similar artists you are genuinely confident about — if unsure, "
        "say so and do not invent names;\n"
        "3) a short subjective impression (2-3 sentences): how it likely sounds, "
        "its mood and what a listener might feel.\n"
        "Ground the genre in the audio evidence; weight metadata tags and the "
        "named artist heavily when present. Be concrete, avoid hedging filler.\n"
        'Return JSON: {"genre": "...", "similar_artists": ["..."], "impression": "..."}.\n\n'
        "--- EVIDENCE ---\n" + brief
    )


def critique(description: Mapping[str, Any], rich: Mapping[str, Any] | None = None,
             metadata: Mapping[str, Any] | None = None) -> dict:
    """Assemble the critic block: metadata, genre_hints, brief, and prompt."""
    meta = dict(metadata or {})
    hints = genre_hints(description, rich=rich, tags=meta.get("tags"))
    brief = build_brief(description, rich=rich, metadata=meta)
    return {
        "metadata": meta,
        "genre_hints": hints,
        "brief": brief,
        "prompt": build_prompt(brief, metadata=meta),
    }


# --- optional standalone LLM verdict (OpenAI-compatible) ------------------

def _http_post_json(url: str, headers: dict, payload: dict, timeout: float = 60.0) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def _strip_fences(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t.split("\n", 1)[-1] if "\n" in t else t
        t = t.rsplit("```", 1)[0]
        if t.lower().startswith("json"):
            t = t[4:]
    return t.strip()


def llm_verdict(prompt: str, *, base_url: str | None = None, api_key: str | None = None,
                model: str | None = None, temperature: float = 0.7,
                timeout: float = 60.0) -> dict:
    """Send the critic prompt to an OpenAI-compatible chat endpoint and parse a
    ``{genre, similar_artists, impression}`` verdict (``{raw: ...}`` if the model
    doesn't return JSON). Config from args or ``MH_LLM_BASE_URL`` /
    ``MH_LLM_API_KEY`` / ``MH_LLM_MODEL``."""
    base_url = base_url or os.environ.get("MH_LLM_BASE_URL", "").strip()
    api_key = api_key or os.environ.get("MH_LLM_API_KEY", "").strip()
    model = model or os.environ.get("MH_LLM_MODEL", "").strip()
    if not base_url or not model:
        raise ValueError("LLM verdict needs a base_url and model "
                         "(args or MH_LLM_BASE_URL / MH_LLM_MODEL)")
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Content-Type": "application/json", "User-Agent": "music-hearing/0.2"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    resp = _http_post_json(url, headers, payload, timeout=timeout)
    content = (resp.get("choices") or [{}])[0].get("message", {}).get("content", "")
    try:
        parsed = json.loads(_strip_fences(content))
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {"raw": content}
