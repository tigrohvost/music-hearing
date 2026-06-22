from music_hearing import semantics


def _profile(centroid, bpm, bands, crest=8.0, dr=20.0, zcr=700.0, rms=-15.0):
    return {
        "spectral_centroid_hz": centroid,
        "estimated_bpm": bpm,
        "bands": bands,
        "crest_factor": crest,
        "dynamic_range_db": dr,
        "zero_crossing_rate": zcr,
        "rms_dbfs": rms,
    }


def test_describe_dark_subheavy_slow():
    p = _profile(200.0, 67.0,
                 {"sub_bass": 0.5, "bass": 0.3, "low_mid": 0.15, "mid": 0.04, "high": 0.01})
    d = semantics.describe(p)
    assert d["brightness"] == "dark"
    assert d["weight"] == "sub-heavy"
    assert d["tempo_feel"] == "slow"
    assert "67 BPM" in d["summary"]


def test_describe_handles_unknown_bpm():
    p = _profile(1500.0, None,
                 {"sub_bass": 0.1, "bass": 0.1, "low_mid": 0.3, "mid": 0.3, "high": 0.2})
    d = semantics.describe(p)
    assert d["tempo_feel"] == "unknown"
    assert d["tempo_bpm"] is None
    assert "BPM" not in d["summary"]


def test_compare_identical_profiles_are_similar():
    bands = {"sub_bass": 0.4, "bass": 0.3, "low_mid": 0.2, "mid": 0.07, "high": 0.03}
    a = _profile(300.0, 90.0, bands)
    c = semantics.compare(a, a)
    assert c["similarity"] == 1.0
    assert c["band_distance"] == 0.0
    assert c["bpm_diff"] == 0.0


def test_compare_bpm_diff_none_when_unknown():
    bands = {"sub_bass": 0.4, "bass": 0.3, "low_mid": 0.2, "mid": 0.07, "high": 0.03}
    a = _profile(300.0, None, bands)
    b = _profile(300.0, 120.0, bands)
    assert semantics.compare(a, b)["bpm_diff"] is None
