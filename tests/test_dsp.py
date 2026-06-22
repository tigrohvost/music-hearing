import math
import struct
import wave

from music_hearing import dsp


def _write_tone(path, freq=440.0, seconds=1.5, rate=8000, amp=0.4):
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


def test_acoustic_profile_of_a_wav_tone(tmp_path):
    wav = tmp_path / "tone.wav"
    _write_tone(wav, freq=440.0, seconds=1.5)
    p = dsp.acoustic_profile(wav)
    assert p.sample_rate == 8000
    assert p.seconds_analyzed > 1.0
    assert -60.0 < p.rms_dbfs < 0.0
    assert 0.0 <= p.peak <= 1.0
    assert set(p.bands) == {"sub_bass", "bass", "low_mid", "mid", "high"}
    # a 440 Hz tone lands in the low_mid band, not sub_bass
    assert p.bands["low_mid"] > p.bands["sub_bass"]


def test_silence_has_no_band_energy(tmp_path):
    wav = tmp_path / "silence.wav"
    _write_tone(wav, freq=440.0, seconds=1.0, amp=0.0)
    p = dsp.acoustic_profile(wav)
    assert all(v == 0.0 for v in p.bands.values())
    assert p.spectral_centroid_hz == 0.0
