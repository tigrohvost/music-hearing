"""Fetch a short audio excerpt from YouTube / YouTube Music (or an allowlisted
open host) with yt-dlp, then profile it with :mod:`music_hearing.dsp`.

Bounded and observational: it downloads only a short excerpt to a temp file,
measures it, and returns a profile. It never publishes, caches, or stores the
audio. YouTube needs a logged-in cookies file and a JavaScript runtime (deno)
on PATH for yt-dlp's signature challenge — see the README.

Configuration is via explicit arguments or ``MH_*`` environment variables
(arguments win): ``MH_YTDLP_BIN``, ``MH_COOKIES_FILE``,
``MH_COOKIES_FROM_BROWSER``, ``MH_EXTRA_HOSTS``, ``MH_NATIVE_AUDIO``,
``MH_EXTRACTOR_ARGS``, ``MH_MUSIC_V2``.
"""
from __future__ import annotations

import os
import pathlib
import re
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlparse

from . import dsp, semantics

DEFAULT_SECONDS = 45.0
MAX_SECONDS = 90.0
# yt-dlp builds older than this are routinely bot-gated by YouTube; warn so the
# failure cause is visible instead of an opaque "Sign in to confirm" error.
STALE_BEFORE = "2025.06.01"
_ALLOWED_HOST_SUFFIXES = (
    "youtube.com",
    "music.youtube.com",
    "youtu.be",
    "www.youtube.com",
    "m.youtube.com",
)


def _env(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def _env_flag(name: str) -> bool:
    return _env(name).lower() in {"1", "true", "yes", "on"}


def _cookies_file_if_present(override: str | None = None) -> str | None:
    """Cookies path (argument wins over ``MH_COOKIES_FILE``), but only if the
    file exists. Guards the non-YouTube path: a stale or removed cookies file
    must not inject ``--cookies <missing>`` into every fetch and break it."""
    p = (override or _env("MH_COOKIES_FILE") or "").strip()
    return p if p and os.path.exists(p) else None


def _stderr_tail(raw, max_chars: int = 600) -> str:
    """Last chunk of yt-dlp stderr (it runs --quiet, so this is the ERROR line)."""
    text = raw or b""
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    text = text.strip()
    return text[-max_chars:] if text else "(no stderr)"


def _extra_hosts(override: str | None = None) -> tuple[str, ...]:
    """Allowlisted non-YouTube hosts (argument wins over ``MH_EXTRA_HOSTS``).

    YouTube increasingly bot-gates anonymous fetches; broadening to open hosts
    such as archive.org keeps the path alive. Default empty -> YouTube-only.
    """
    raw = _env("MH_EXTRA_HOSTS") if override is None else override
    return tuple(h.strip().lower() for h in str(raw).replace(",", " ").split() if h.strip())


@dataclass(frozen=True)
class MusicHearingProfile:
    source: str
    resolved_input: str
    extractor: str
    seconds_requested: float
    profile: dict[str, Any]
    yt_dlp_version: str = ""
    stale_warning: str = ""
    description: dict[str, Any] = field(default_factory=dict)
    rich: dict[str, Any] | None = None
    music_v2: dict[str, Any] | None = None
    music_description: dict[str, Any] | None = None
    critic: dict[str, Any] | None = None


def _require_yt_dlp(override: str | None = None) -> str:
    exe = override or _env("MH_YTDLP_BIN") or shutil.which("yt-dlp")
    if not exe:
        raise RuntimeError("yt-dlp is required for music hearing, but it is not installed")
    return exe


def ytdlp_version(yt_dlp_bin: str | None = None) -> str:
    """Best-effort ``yt-dlp --version`` (e.g. '2026.06.09'); '' on any failure."""
    exe = yt_dlp_bin or _env("MH_YTDLP_BIN") or shutil.which("yt-dlp")
    if not exe:
        return ""
    try:
        out = subprocess.run([exe, "--version"], check=True, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, timeout=15)
        return out.stdout.decode("utf-8", "replace").strip().splitlines()[0].strip()
    except Exception:
        return ""


def version_is_stale(version: str, min_date: str = STALE_BEFORE) -> bool:
    """True if a ``YYYY.MM.DD`` yt-dlp version is older than ``min_date``.

    Unparseable input never warns (returns False) to avoid false alarms."""
    def _key(v: str):
        m = re.match(r"^\s*(\d{4})\.(\d{2})\.(\d{2})", str(v or ""))
        return (int(m.group(1)), int(m.group(2)), int(m.group(3))) if m else None
    cur, floor = _key(version), _key(min_date)
    if cur is None or floor is None:
        return False
    return cur < floor


def _is_youtube_url(value: str) -> bool:
    try:
        parsed = urlparse(value)
    except ValueError:
        return False
    host = (parsed.hostname or "").lower()
    return parsed.scheme in {"http", "https"} and any(
        host == suffix or host.endswith("." + suffix) for suffix in _ALLOWED_HOST_SUFFIXES
    )


def _host_allowed(value: str, extra_hosts: tuple[str, ...]) -> bool:
    try:
        host = (urlparse(value).hostname or "").lower()
    except ValueError:
        return False
    return any(host == h or host.endswith("." + h) for h in extra_hosts)


def resolve_source(source: str, extra_hosts: tuple[str, ...] = ()) -> tuple[str, str]:
    """Return (yt-dlp input, extractor label) for a URL or search phrase.

    YouTube URLs -> ('youtube-url'); URLs on an allowlisted ``extra_hosts``
    (e.g. archive.org) -> ('url'); other URLs are rejected; bare text becomes a
    bounded ``ytsearch1`` query. ``extra_hosts`` defaults empty (YouTube-only)."""
    clean = " ".join(str(source or "").strip().split())
    if not clean:
        raise ValueError("source is empty")
    if _is_youtube_url(clean):
        return clean, "youtube-url"
    if extra_hosts and _host_allowed(clean, extra_hosts):
        return clean, "url"
    if re.search(r"\b(?:https?://|www\.)", clean, flags=re.I):
        raise ValueError("only YouTube/YouTube Music URLs or allowlisted hosts are accepted")
    return f"ytsearch1:{clean}", "ytsearch1"


def build_ytdlp_cmd(resolved: str, output: str, seconds: float, *, yt_dlp_bin: str,
                    cookies_from_browser: str | None = None, native_audio: bool = False,
                    extractor_args: str | None = None,
                    cookies_file: str | None = None,
                    write_info_json: bool = False) -> list[str]:
    """Assemble the yt-dlp argv. Defaults reproduce the mp3 excerpt pipeline;
    ``native_audio`` skips the lossy re-encode, ``cookies_*`` / ``extractor_args``
    help pass YouTube's bot gate."""
    cmd = [
        yt_dlp_bin,
        "--no-playlist", "--no-progress", "--quiet", "--no-warnings",
        # Fetch public media directly; bypass an inherited HTTPS_PROXY that may
        # only route an app's LLM/API traffic, not YouTube.
        "--proxy", "",
        "--format", "bestaudio/best",
        "--download-sections", f"*0-{seconds:.3f}",
        "--force-keyframes-at-cuts",
    ]
    if not native_audio:
        cmd += ["--extract-audio", "--audio-format", "mp3"]
    if cookies_from_browser:
        cmd += ["--cookies-from-browser", cookies_from_browser]
    if cookies_file:
        cmd += ["--cookies", cookies_file]
    if extractor_args:
        cmd += ["--extractor-args", extractor_args]
    if write_info_json:
        cmd += ["--write-info-json"]
    cmd += ["--output", str(output), resolved]
    return cmd


def _download_excerpt(source: str, outdir: pathlib.Path, seconds: float, *,
                      ytdlp_bin: str | None = None, cookies_file: str | None = None,
                      cookies_from_browser: str | None = None,
                      native_audio: bool | None = None, extractor_args: str | None = None,
                      extra_hosts: str | None = None,
                      write_info_json: bool = False) -> pathlib.Path:
    yt_dlp = _require_yt_dlp(ytdlp_bin)
    resolved, _ = resolve_source(source, _extra_hosts(extra_hosts))
    output = outdir / "source.%(ext)s"
    cmd = build_ytdlp_cmd(
        resolved, str(output), seconds, yt_dlp_bin=yt_dlp,
        cookies_from_browser=(cookies_from_browser or _env("MH_COOKIES_FROM_BROWSER") or None),
        native_audio=(native_audio if native_audio is not None else _env_flag("MH_NATIVE_AUDIO")),
        extractor_args=(extractor_args or _env("MH_EXTRACTOR_ARGS") or None),
        cookies_file=_cookies_file_if_present(cookies_file),
        write_info_json=write_info_json,
    )
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       timeout=max(45, int(seconds) + 90))
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"yt-dlp failed (exit {e.returncode}): {_stderr_tail(e.stderr)}") from e
    # exclude yt-dlp sidecars (.info.json) when picking the audio file
    matches = [m for m in sorted(outdir.glob("source.*")) if not m.name.endswith(".info.json")]
    if not matches:
        raise RuntimeError("yt-dlp completed but produced no audio file")
    return matches[0]


def profile_music(source: str, seconds: float = DEFAULT_SECONDS, *,
                  cookies_file: str | None = None, ytdlp_bin: str | None = None,
                  extra_hosts: str | None = None, cookies_from_browser: str | None = None,
                  native_audio: bool | None = None, extractor_args: str | None = None,
                  rich: bool = False, music: bool = False, critic: bool = False, llm: bool = False,
                  llm_base_url: str | None = None, llm_api_key: str | None = None,
                  llm_model: str | None = None) -> MusicHearingProfile:
    """Fetch a bounded excerpt for ``source`` (URL or search phrase) and return
    its acoustic profile + semantic description.

    ``rich=True`` attaches the numpy spectral profile (needs the ``rich`` extra).
    ``music=True`` or ``RAIN_HEARING_MUSIC_V2=1`` / ``MH_MUSIC_V2=1`` attaches
    the additive ``music_v2`` profile.
    ``critic=True`` attaches a ``critic`` block (metadata + genre hints + an
    evidence brief + a ready prompt) so a model can name genre / similar artists
    / impression. ``llm=True`` additionally calls an OpenAI-compatible endpoint
    to fill in that verdict."""
    bounded_seconds = max(5.0, min(float(seconds), MAX_SECONDS))
    resolved, extractor = resolve_source(source, _extra_hosts(extra_hosts))
    from . import music_v2 as _music_v2
    music_requested = _music_v2.requested(music) or _music_v2.env_enabled()
    version = ytdlp_version(ytdlp_bin)
    warning = ""
    if extractor in {"youtube-url", "ytsearch1"} and version_is_stale(version):
        warning = (
            f"yt-dlp {version} is older than {STALE_BEFORE}; YouTube may bot-gate "
            "this fetch. Update yt-dlp, pass --cookies / MH_COOKIES_FILE, or use an "
            "allowlisted host (e.g. archive.org)."
        )
    critic_block = None
    music_block = None
    music_description = None
    with tempfile.TemporaryDirectory(prefix="music-hearing-") as td:
        tdpath = pathlib.Path(td)
        audio = _download_excerpt(
            source, tdpath, bounded_seconds, ytdlp_bin=ytdlp_bin,
            cookies_file=cookies_file, cookies_from_browser=cookies_from_browser,
            native_audio=native_audio, extractor_args=extractor_args,
            extra_hosts=extra_hosts, write_info_json=critic)
        profile = dsp.acoustic_profile(audio, max_seconds=bounded_seconds)
        profile_dict = asdict(profile)
        description = semantics.describe(profile_dict)
        rich_data = None
        if rich or critic:
            from . import spectral
            rich_data = spectral.rich_profile(str(audio), max_seconds=bounded_seconds)
        elif music_requested:
            try:
                from . import spectral
                rich_data = spectral.rich_profile(str(audio), max_seconds=bounded_seconds)
            except Exception:
                rich_data = {}
        if music_requested:
            source_kind = "youtube_excerpt" if extractor in {"youtube-url", "ytsearch1"} else "external_excerpt"
            music_block = _music_v2.build_music_v2(
                audio,
                base_profile=profile_dict,
                rich_profile=rich_data,
                seconds=bounded_seconds,
                source_kind=source_kind,
            )
            music_description = semantics.describe_music_v2(music_block)
        if critic:
            from . import critic as _critic
            from . import metadata as _metadata
            info_files = sorted(tdpath.glob("*.info.json"))
            meta = _metadata.read_info_json(info_files[0]) if info_files else _metadata.parse_info({})
            critic_block = _critic.critique(description, rich=rich_data, metadata=meta)
            if llm:
                try:
                    critic_block["verdict"] = _critic.llm_verdict(
                        critic_block["prompt"], base_url=llm_base_url,
                        api_key=llm_api_key, model=llm_model)
                except Exception as exc:
                    critic_block["verdict_error"] = f"{type(exc).__name__}: {exc}"
    return MusicHearingProfile(
        source=str(source),
        resolved_input=resolved,
        extractor=extractor,
        seconds_requested=round(bounded_seconds, 2),
        profile=profile_dict,
        yt_dlp_version=version,
        stale_warning=warning,
        description=description,
        rich=(rich_data if rich else None),
        music_v2=music_block,
        music_description=music_description,
        critic=critic_block,
    )
