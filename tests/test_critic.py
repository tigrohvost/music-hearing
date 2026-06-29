from music_hearing import critic


def _desc(tempo_feel, brightness, weight, texture, summary=None):
    return {
        "tempo_feel": tempo_feel, "brightness": brightness, "weight": weight,
        "texture": texture, "dynamics": "moderate",
        "summary": summary or f"{tempo_feel}, {brightness}, {weight}, {texture}",
    }


def test_genre_hints_ambient_for_slow_subheavy_tonal():
    desc = _desc("slow", "dark", "sub-heavy", "tonal")
    rich = {"hpss": {"harmonic_ratio": 0.9, "percussive_ratio": 0.1}, "onset_rate_hz": 0.2}
    hints = critic.genre_hints(desc, rich=rich)
    assert any(g in hints for g in ("ambient", "drone", "downtempo"))


def test_genre_hints_electronic_for_fast_percussive():
    desc = _desc("up-tempo", "bright", "balanced", "textured")
    rich = {"hpss": {"harmonic_ratio": 0.3, "percussive_ratio": 0.7}, "onset_rate_hz": 3.5}
    hints = critic.genre_hints(desc, rich=rich)
    assert any(g in hints for g in ("electronic", "dance", "techno", "house"))


def test_genre_hints_prepends_metadata_tags():
    desc = _desc("slow", "dark", "sub-heavy", "tonal")
    hints = critic.genre_hints(desc, tags=["Psybient", "Chillout"])
    assert hints[0] == "psybient"
    assert "chillout" in hints


def test_build_brief_includes_evidence_and_metadata():
    desc = _desc("slow", "dark", "sub-heavy", "tonal", summary="slow, dark, sub-heavy (~67 BPM)")
    meta = {"artist": "Meg Bowles", "album": "Blue Cosmos", "title": "Organic Lullaby", "tags": ["ambient"]}
    rich = {"key": {"key": "D", "mode": "minor", "confidence": 0.9}, "tempo": {"bpm": 67.0}}
    brief = critic.build_brief(desc, rich=rich, metadata=meta)
    assert "Meg Bowles" in brief
    assert "slow, dark, sub-heavy" in brief
    assert "D" in brief and "minor" in brief


def test_build_prompt_asks_for_genre_artists_impression():
    prompt = critic.build_prompt("EVIDENCE-BRIEF-HERE", metadata={"artist": "X"})
    low = prompt.lower()
    assert "genre" in low
    assert "artist" in low
    assert "impression" in low
    assert "EVIDENCE-BRIEF-HERE" in prompt


def test_build_prompt_uses_skeptical_critic_persona():
    prompt = critic.build_prompt("EVIDENCE-BRIEF-HERE").lower()
    assert "skeptical" in prompt
    # meticulous/erudite voice: must still refuse to invent and must flag thin evidence
    assert "do not invent" in prompt
    assert "thin" in prompt or "overclaim" in prompt


def test_critique_assembles_block():
    desc = _desc("slow", "dark", "sub-heavy", "tonal")
    block = critic.critique(desc, rich=None, metadata={"artist": "X", "tags": ["ambient"]})
    assert set(block) >= {"metadata", "genre_hints", "brief", "prompt"}
    assert "ambient" in block["genre_hints"]


def test_llm_verdict_parses_json(monkeypatch):
    def fake_post(url, headers, payload, timeout=60):
        return {"choices": [{"message": {"content":
                '{"genre": "ambient", "similar_artists": ["Stars of the Lid"], '
                '"impression": "spacious and still"}'}}]}
    monkeypatch.setattr(critic, "_http_post_json", fake_post)
    v = critic.llm_verdict("prompt", base_url="http://x/v1", api_key="k", model="m")
    assert v["genre"] == "ambient"
    assert "Stars of the Lid" in v["similar_artists"]
    assert "impression" in v


def test_llm_verdict_falls_back_to_raw_on_non_json(monkeypatch):
    def fake_post(url, headers, payload, timeout=60):
        return {"choices": [{"message": {"content": "Just some prose, not JSON."}}]}
    monkeypatch.setattr(critic, "_http_post_json", fake_post)
    v = critic.llm_verdict("prompt", base_url="http://x/v1", api_key="k", model="m")
    assert v["raw"] == "Just some prose, not JSON."


def test_llm_verdict_requires_config():
    import pytest
    with pytest.raises(ValueError):
        critic.llm_verdict("prompt", base_url=None, api_key=None, model=None)
