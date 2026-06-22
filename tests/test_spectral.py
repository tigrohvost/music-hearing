import pytest

np = pytest.importorskip("numpy")

from music_hearing import spectral


def _sine(freq, rate=22050, seconds=2.0):
    t = np.arange(int(rate * seconds)) / rate
    return 0.5 * np.sin(2 * np.pi * freq * t)


def test_rich_features_centroid_tracks_frequency():
    rate = 22050
    low = spectral.rich_features(_sine(220, rate), rate)["spectral_centroid_hz"]
    high = spectral.rich_features(_sine(2000, rate), rate)["spectral_centroid_hz"]
    assert high > low
    assert 100 < low < 600


def test_estimate_key_on_c_major_triad():
    rate = 22050
    # C4 + E4 + G4
    mix = _sine(261.63, rate) + _sine(329.63, rate) + _sine(392.0, rate)
    chroma = spectral.chroma_vector(mix, rate)
    key = spectral.estimate_key(chroma)
    assert key["key"] == "C"
    assert key["mode"] == "major"
    assert key["confidence"] > 0.7


def test_estimate_tempo_on_click_train_is_musical():
    rate = 22050
    x = np.zeros(int(rate * 4))
    # clicks every 0.5s -> 120 BPM
    for i in range(0, x.size, rate // 2):
        x[i:i + 50] = 1.0
    tempo = spectral.estimate_tempo(x, rate)
    assert tempo["bpm"] is not None
    assert 100 <= tempo["bpm"] <= 130
    assert tempo["confidence"] > 0.0


def test_rich_profile_from_samples_has_expected_keys():
    rate = 22050
    out = spectral.rich_profile_from_samples(_sine(440, rate), rate)
    for k in ("spectral_centroid_hz", "chroma", "key", "timeline", "mfcc", "tempo", "hpss"):
        assert k in out
