"""Track metadata from yt-dlp's info JSON.

``parse_info`` normalizes a yt-dlp info dict (or playlist wrapper) into a small
flat record — title / artist / album / uploader / tags / description — that the
critic uses to ground genre and similar-artist judgments. ``fetch_metadata``
runs a metadata-only yt-dlp call (``--skip-download``) when no info JSON was
written during the audio fetch.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from typing import Any

_DESC_MAX = 600


def _first(d: dict, *keys):
    for k in keys:
        v = d.get(k)
        if v:
            return v
    return None


def parse_info(info: dict[str, Any]) -> dict[str, Any]:
    """Flatten a yt-dlp info dict into a small metadata record. Unwraps a
    single-entry playlist (what ``ytsearch1`` returns)."""
    info = info or {}
    entries = info.get("entries")
    if entries:
        info = entries[0] or {}
    tags = info.get("tags") or []
    if isinstance(tags, str):
        tags = [tags]
    cats = info.get("categories") or []
    desc = info.get("description") or ""
    if len(desc) > _DESC_MAX:
        desc = desc[:_DESC_MAX - 1].rstrip() + "…"
    return {
        "title": info.get("title"),
        "artist": _first(info, "artist", "creator", "uploader", "channel"),
        "album": info.get("album"),
        "uploader": info.get("uploader"),
        "tags": [str(t) for t in tags],
        "categories": [str(c) for c in cats],
        "description": desc,
        "webpage_url": info.get("webpage_url"),
        "duration": info.get("duration"),
    }


def fetch_metadata(source: str, *, ytdlp_bin: str | None = None,
                   cookies_file: str | None = None, extra_hosts: str | None = None,
                   timeout: float = 60.0) -> dict[str, Any]:
    """Metadata-only yt-dlp call (no audio download). Best-effort: returns an
    empty record on any failure so the critic still works from acoustics."""
    from . import fetch
    exe = ytdlp_bin or fetch._env("MH_YTDLP_BIN") or shutil.which("yt-dlp")
    if not exe:
        return parse_info({})
    resolved, _ = fetch.resolve_source(source, fetch._extra_hosts(extra_hosts))
    cmd = [exe, "--no-warnings", "--no-playlist", "--skip-download", "--proxy", "",
           "--dump-single-json"]
    cf = fetch._cookies_file_if_present(cookies_file)
    if cf:
        cmd += ["--cookies", cf]
    cmd += [resolved]
    try:
        out = subprocess.run(cmd, check=True, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE, timeout=timeout)
        return parse_info(json.loads(out.stdout.decode("utf-8", "replace")))
    except Exception:
        return parse_info({})


def read_info_json(path: str | os.PathLike) -> dict[str, Any]:
    """Parse a yt-dlp ``*.info.json`` file into a metadata record."""
    try:
        with open(path, encoding="utf-8") as f:
            return parse_info(json.load(f))
    except Exception:
        return parse_info({})
