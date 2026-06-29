import json
import math
import shutil
import struct
import wave
from dataclasses import asdict

import pytest

from music_hearing import fetch


def _write_tone(path, freq=440.0, seconds=1.0, rate=8000, amp=0.4):
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


# --- resolve_source ---

def test_resolve_accepts_youtube_music_url():
    resolved, extractor = fetch.resolve_source("https://music.youtube.com/watch?v=abc123")
    assert resolved == "https://music.youtube.com/watch?v=abc123"
    assert extractor == "youtube-url"


def test_resolve_turns_plain_track_name_into_bounded_search():
    resolved, extractor = fetch.resolve_source("Meg Bowles - Organic lullaby")
    assert resolved == "ytsearch1:Meg Bowles - Organic lullaby"
    assert extractor == "ytsearch1"


def test_resolve_rejects_non_youtube_url():
    with pytest.raises(ValueError, match="only YouTube"):
        fetch.resolve_source("https://example.com/file.mp3")


def test_resolve_extra_host_accepts_allowlisted_non_youtube():
    resolved, extractor = fetch.resolve_source(
        "https://archive.org/details/some-ambient", extra_hosts=("archive.org",))
    assert resolved == "https://archive.org/details/some-ambient"
    assert extractor == "url"


def test_resolve_extra_host_accepts_subdomain():
    url = "https://ia800.us.archive.org/12/items/x/track.mp3"
    resolved, extractor = fetch.resolve_source(url, extra_hosts=("archive.org",))
    assert resolved == url
    assert extractor == "url"


def test_resolve_extra_host_still_rejects_unlisted():
    with pytest.raises(ValueError, match="only YouTube"):
        fetch.resolve_source("https://example.com/file.mp3", extra_hosts=("archive.org",))


# --- build_ytdlp_cmd ---

def test_build_cmd_default_matches_mp3_pipeline():
    cmd = fetch.build_ytdlp_cmd("ytsearch1:x", "/tmp/out.%(ext)s", 45.0,
                                yt_dlp_bin="/usr/bin/yt-dlp")
    assert cmd[0] == "/usr/bin/yt-dlp"
    assert "--extract-audio" in cmd
    assert cmd[cmd.index("--audio-format") + 1] == "mp3"
    assert "bestaudio/best" in cmd
    assert "*0-45.000" in cmd
    assert cmd[-1] == "ytsearch1:x"
    assert "--cookies-from-browser" not in cmd


def test_build_cmd_native_audio_skips_reencode():
    cmd = fetch.build_ytdlp_cmd("u", "/tmp/o.%(ext)s", 30.0,
                                yt_dlp_bin="yt-dlp", native_audio=True)
    assert "--extract-audio" not in cmd
    assert "--audio-format" not in cmd
    assert "bestaudio/best" in cmd


def test_build_cmd_adds_cookies_from_browser():
    cmd = fetch.build_ytdlp_cmd("u", "/tmp/o.%(ext)s", 30.0,
                                yt_dlp_bin="yt-dlp", cookies_from_browser="chromium")
    assert cmd[cmd.index("--cookies-from-browser") + 1] == "chromium"


def test_build_cmd_adds_extractor_args():
    cmd = fetch.build_ytdlp_cmd("u", "/tmp/o.%(ext)s", 30.0, yt_dlp_bin="yt-dlp",
                                extractor_args="youtube:player_client=web_safari")
    assert cmd[cmd.index("--extractor-args") + 1] == "youtube:player_client=web_safari"


def test_build_cmd_adds_cookies_file():
    cmd = fetch.build_ytdlp_cmd("u", "/tmp/o.%(ext)s", 30.0,
                                yt_dlp_bin="yt-dlp", cookies_file="/secure/yt-cookies.txt")
    assert cmd[cmd.index("--cookies") + 1] == "/secure/yt-cookies.txt"
    assert "--cookies-from-browser" not in cmd


def test_build_cmd_forces_direct_no_proxy():
    # media fetch must bypass any inherited HTTPS_PROXY meant for app/LLM traffic
    cmd = fetch.build_ytdlp_cmd("u", "/tmp/o.%(ext)s", 30.0, yt_dlp_bin="yt-dlp")
    assert cmd[cmd.index("--proxy") + 1] == ""


def test_build_cmd_write_info_json_opt_in():
    assert "--write-info-json" not in fetch.build_ytdlp_cmd(
        "u", "/tmp/o.%(ext)s", 30.0, yt_dlp_bin="yt-dlp")
    assert "--write-info-json" in fetch.build_ytdlp_cmd(
        "u", "/tmp/o.%(ext)s", 30.0, yt_dlp_bin="yt-dlp", write_info_json=True)


def test_build_cmd_remote_components_opt_in():
    # default off: older yt-dlp builds reject an unknown --remote-components flag
    assert "--remote-components" not in fetch.build_ytdlp_cmd(
        "u", "/tmp/o.%(ext)s", 30.0, yt_dlp_bin="yt-dlp")
    cmd = fetch.build_ytdlp_cmd("u", "/tmp/o.%(ext)s", 30.0, yt_dlp_bin="yt-dlp",
                                remote_components="ejs:github")
    assert cmd[cmd.index("--remote-components") + 1] == "ejs:github"


# --- version_is_stale ---

def test_version_is_stale_flags_old_builds():
    assert fetch.version_is_stale("2024.04.09", min_date="2025.06.01") is True


def test_version_is_stale_passes_recent_builds():
    assert fetch.version_is_stale("2026.06.09", min_date="2025.06.01") is False


def test_version_is_stale_unparseable_does_not_warn():
    assert fetch.version_is_stale("", min_date="2025.06.01") is False
    assert fetch.version_is_stale("weird", min_date="2025.06.01") is False


# --- _download_excerpt env + overrides + error surfacing ---

def test_download_excerpt_reads_env_for_native_and_cookies(tmp_path, monkeypatch):
    monkeypatch.setenv("MH_NATIVE_AUDIO", "1")
    monkeypatch.setenv("MH_COOKIES_FROM_BROWSER", "chromium")
    monkeypatch.setattr(fetch.shutil, "which", lambda _: "/usr/bin/yt-dlp")
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        (tmp_path / "source.webm").write_bytes(b"\x00\x00")
        class R: pass
        return R()

    monkeypatch.setattr(fetch.subprocess, "run", fake_run)
    out = fetch._download_excerpt("anything", tmp_path, 20.0)
    assert out.name == "source.webm"
    assert "--extract-audio" not in seen["cmd"]
    assert seen["cmd"][seen["cmd"].index("--cookies-from-browser") + 1] == "chromium"


def test_download_excerpt_reads_cookies_file_env(tmp_path, monkeypatch):
    cookie = tmp_path / "yt-cookies.txt"
    cookie.write_text("# Netscape HTTP Cookie File\n")
    monkeypatch.setenv("MH_COOKIES_FILE", str(cookie))
    monkeypatch.setattr(fetch.shutil, "which", lambda _: "/usr/bin/yt-dlp")
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        (tmp_path / "source.webm").write_bytes(b"\x00\x00")
        class R: pass
        return R()

    monkeypatch.setattr(fetch.subprocess, "run", fake_run)
    fetch._download_excerpt("anything", tmp_path, 20.0)
    assert seen["cmd"][seen["cmd"].index("--cookies") + 1] == str(cookie)


def test_download_excerpt_skips_missing_cookies_file(tmp_path, monkeypatch):
    monkeypatch.setenv("MH_COOKIES_FILE", str(tmp_path / "does-not-exist.txt"))
    monkeypatch.setattr(fetch.shutil, "which", lambda _: "/usr/bin/yt-dlp")
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        (tmp_path / "source.webm").write_bytes(b"\x00\x00")
        class R: pass
        return R()

    monkeypatch.setattr(fetch.subprocess, "run", fake_run)
    fetch._download_excerpt("anything", tmp_path, 20.0)
    assert "--cookies" not in seen["cmd"]


def test_download_excerpt_explicit_cookies_arg_overrides_env(tmp_path, monkeypatch):
    arg_cookie = tmp_path / "explicit.txt"
    arg_cookie.write_text("# Netscape HTTP Cookie File\n")
    monkeypatch.setenv("MH_COOKIES_FILE", "/env/ignored.txt")
    monkeypatch.setattr(fetch.shutil, "which", lambda _: "/usr/bin/yt-dlp")
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        (tmp_path / "source.webm").write_bytes(b"\x00\x00")
        class R: pass
        return R()

    monkeypatch.setattr(fetch.subprocess, "run", fake_run)
    fetch._download_excerpt("anything", tmp_path, 20.0, cookies_file=str(arg_cookie))
    assert seen["cmd"][seen["cmd"].index("--cookies") + 1] == str(arg_cookie)


def test_download_excerpt_reads_remote_components_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MH_REMOTE_COMPONENTS", "ejs:github")
    monkeypatch.setattr(fetch.shutil, "which", lambda _: "/usr/bin/yt-dlp")
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        (tmp_path / "source.webm").write_bytes(b"\x00\x00")
        class R: pass
        return R()

    monkeypatch.setattr(fetch.subprocess, "run", fake_run)
    fetch._download_excerpt("anything", tmp_path, 20.0)
    assert seen["cmd"][seen["cmd"].index("--remote-components") + 1] == "ejs:github"


def test_download_excerpt_remote_components_arg_overrides_env(tmp_path, monkeypatch):
    monkeypatch.setenv("MH_REMOTE_COMPONENTS", "ejs:npm")
    monkeypatch.setattr(fetch.shutil, "which", lambda _: "/usr/bin/yt-dlp")
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        (tmp_path / "source.webm").write_bytes(b"\x00\x00")
        class R: pass
        return R()

    monkeypatch.setattr(fetch.subprocess, "run", fake_run)
    fetch._download_excerpt("anything", tmp_path, 20.0, remote_components="ejs:github")
    assert seen["cmd"][seen["cmd"].index("--remote-components") + 1] == "ejs:github"


def test_download_excerpt_surfaces_ytdlp_stderr_on_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch.shutil, "which", lambda _: "/usr/bin/yt-dlp")

    def fake_run(cmd, **kw):
        raise fetch.subprocess.CalledProcessError(
            1, cmd, output=b"",
            stderr=b"ERROR: [youtube] abc: Sign in to confirm you are not a bot.")

    monkeypatch.setattr(fetch.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError) as ei:
        fetch._download_excerpt("anything", tmp_path, 20.0)
    msg = str(ei.value)
    assert "Sign in to confirm" in msg
    assert "exit 1" in msg


# --- profile_music end-to-end with download mocked (no network) ---

def test_profile_music_uses_downloaded_excerpt(tmp_path, monkeypatch):
    wav = tmp_path / "excerpt.wav"
    _write_tone(wav, freq=440.0, seconds=1.0)

    def fake_download(source, outdir, seconds, **kwargs):
        assert source == "Meg Bowles - Organic lullaby"
        assert seconds == 5.0
        return wav

    monkeypatch.setattr(fetch, "_download_excerpt", fake_download)
    monkeypatch.setattr(fetch, "ytdlp_version", lambda *a, **k: "2026.06.09")
    result = fetch.profile_music("Meg Bowles - Organic lullaby", seconds=1.0)
    data = asdict(result)
    assert data["resolved_input"] == "ytsearch1:Meg Bowles - Organic lullaby"
    assert data["seconds_requested"] == 5.0
    assert data["profile"]["bands"]["low_mid"] > data["profile"]["bands"]["sub_bass"]
    assert data["description"]["summary"]
    assert "brightness" in data["description"]
    assert data["rich"] is None


def test_profile_music_critic_attaches_block(tmp_path, monkeypatch):
    pytest.importorskip("numpy")
    src = tmp_path / "tone.wav"
    _write_tone(src, freq=300.0, seconds=2.0)

    def fake_download(source, outdir, seconds, **kwargs):
        assert kwargs.get("write_info_json") is True
        (outdir / "source.info.json").write_text(json.dumps({
            "title": "Organic Lullaby", "artist": "Meg Bowles",
            "album": "Blue Cosmos", "tags": ["ambient"]}))
        dst = outdir / "source.wav"
        shutil.copy(src, dst)
        return dst

    monkeypatch.setattr(fetch, "_download_excerpt", fake_download)
    monkeypatch.setattr(fetch, "ytdlp_version", lambda *a, **k: "2026.06.09")
    res = fetch.profile_music("Meg Bowles Organic Lullaby", seconds=2.0, critic=True)
    c = res.critic
    assert c is not None
    assert c["metadata"]["artist"] == "Meg Bowles"
    assert "ambient" in c["genre_hints"]
    assert "Meg Bowles" in c["brief"]
    assert "genre" in c["prompt"].lower()
    assert "similar_artists" in c["prompt"] or "artist" in c["prompt"].lower()
    # critic does not force the top-level rich field
    assert res.rich is None
