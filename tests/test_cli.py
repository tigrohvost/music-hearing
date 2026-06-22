import json

from music_hearing import cli, fetch


def _fake_profile():
    return fetch.MusicHearingProfile(
        source="Meg Bowles Organic Lullaby",
        resolved_input="ytsearch1:Meg Bowles Organic Lullaby",
        extractor="ytsearch1",
        seconds_requested=45.0,
        profile={"bands": {}, "estimated_bpm": 67.0},
        yt_dlp_version="2026.06.09",
        description={"summary": "slow, dark, sub-heavy (~67 BPM)", "brightness": "dark"},
    )


def test_cli_passes_flags_through_and_prints_json(capsys, monkeypatch):
    seen = {}

    def fake_profile_music(source, seconds, **kwargs):
        seen["source"] = source
        seen["seconds"] = seconds
        seen.update(kwargs)
        return _fake_profile()

    monkeypatch.setattr(cli, "profile_music", fake_profile_music)
    rc = cli.main(["Meg Bowles Organic Lullaby", "--seconds", "30",
                   "--cookies", "/tmp/c.txt", "--rich"])
    assert rc == 0
    assert seen["source"] == "Meg Bowles Organic Lullaby"
    assert seen["seconds"] == 30.0
    assert seen["cookies_file"] == "/tmp/c.txt"
    assert seen["rich"] is True
    out = json.loads(capsys.readouterr().out)
    assert out["description"]["summary"].startswith("slow")


def test_cli_summary_prints_one_line(capsys, monkeypatch):
    monkeypatch.setattr(cli, "profile_music", lambda *a, **k: _fake_profile())
    rc = cli.main(["whatever", "--summary"])
    assert rc == 0
    assert capsys.readouterr().out.strip() == "slow, dark, sub-heavy (~67 BPM)"


def test_cli_reports_failure_nonzero(capsys, monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("yt-dlp failed (exit 1): ERROR: Sign in to confirm")

    monkeypatch.setattr(cli, "profile_music", boom)
    rc = cli.main(["whatever"])
    assert rc == 1
    assert "Sign in to confirm" in capsys.readouterr().err
