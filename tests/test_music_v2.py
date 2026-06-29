import math
import struct
import wave
from dataclasses import asdict
from importlib.util import find_spec

import pytest

from music_hearing import dsp, fetch, music_v2, semantics


def _write_tone(path, freq=440.0, seconds=2.0, rate=22050, amp=0.4):
    n = int(seconds * rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        data = bytearray()
        for i in range(n):
            v = int(amp * 32767 * math.sin(2 * math.pi * freq * i / rate))
            data.extend(struct.pack("<h", v))
        w.writeframes(bytes(data))


@pytest.mark.skipif(find_spec("numpy") is None, reason="music_v2 full analysis needs numpy")
def test_build_music_v2_contract_is_bounded_and_serializable(tmp_path):
    wav = tmp_path / "tone.wav"
    _write_tone(wav, seconds=2.0)
    base = asdict(dsp.acoustic_profile(wav))
    out = music_v2.build_music_v2(wav, base_profile=base, seconds=5.0, source_kind="test")
    assert out["schema"] == "music_hearing_profile.v2"
    assert out["excerpt"]["source_kind"] == "test"
    assert out["rhythm"]["density"]["onset_rate_per_sec"] >= 0.0
    assert out["structure"]["section_count"] >= 1
    assert out["embedding"]["schema"] == "music_embedding.handcrafted.v1"
    assert out["embedding"]["dim"] == 64
    assert len(out["embedding"]["values"]) == 64
    assert len(out["structure"]["novelty"]["values_max_128"]) <= 128
    assert semantics.describe_music_v2(out)["summary"]


def test_profile_music_env_flag_adds_music_v2_without_rich_output(tmp_path, monkeypatch):
    wav = tmp_path / "tone.wav"
    _write_tone(wav, seconds=2.0)
    monkeypatch.setenv("MH_MUSIC_V2", "1")
    monkeypatch.setattr(fetch, "_download_excerpt", lambda *a, **k: wav)
    monkeypatch.setattr(fetch, "ytdlp_version", lambda *a, **k: "2026.06.09")

    prof = fetch.profile_music("Meg Bowles Organic Lullaby", seconds=5.0)

    assert prof.rich is None
    assert prof.music_v2["schema"] == "music_hearing_profile.v2"
    assert prof.music_description["summary"]
    if prof.music_v2.get("embedding") is not None:
        assert prof.music_v2["embedding"]["dim"] == 64
    else:
        assert prof.music_v2["quality_flags"]["warnings"]


def test_music_v2_requested_accepts_explicit_strings(monkeypatch):
    monkeypatch.delenv("MH_MUSIC_V2", raising=False)
    assert music_v2.requested("v2") is True
    assert music_v2.requested("false") is False
    monkeypatch.setenv("MH_MUSIC_V2", "1")
    assert music_v2.env_enabled() is True
