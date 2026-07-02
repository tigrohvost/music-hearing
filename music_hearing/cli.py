"""``music-hearing`` command-line entry point."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict

from .fetch import DEFAULT_SECONDS, MAX_SECONDS, profile_music


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="music-hearing",
        description="Hear a track: fetch a short excerpt and print its acoustic "
                    "profile + a plain-language description.",
    )
    from . import __version__
    ap.add_argument("--version", action="version", version=f"music-hearing {__version__}")
    ap.add_argument("source", help="YouTube/YouTube Music URL, an allowlisted host URL, "
                                   "or a search phrase (e.g. 'Meg Bowles Organic Lullaby')")
    ap.add_argument("--seconds", type=float, default=DEFAULT_SECONDS,
                    help=f"excerpt length to analyze, capped at {MAX_SECONDS}s")
    ap.add_argument("--cookies", help="path to a Netscape cookies.txt (YouTube auth)")
    ap.add_argument("--ytdlp-bin", help="path to the yt-dlp binary (default: search PATH)")
    ap.add_argument("--extra-hosts", help="comma/space list of extra allowed hosts, e.g. archive.org")
    ap.add_argument("--cookies-from-browser", help="yt-dlp --cookies-from-browser value, e.g. chromium")
    ap.add_argument("--extractor-args", help="raw yt-dlp --extractor-args value")
    ap.add_argument("--remote-components",
                    help="yt-dlp --remote-components value (or MH_REMOTE_COMPONENTS); "
                         "current yt-dlp needs 'ejs:github' to solve YouTube's JS challenge")
    ap.add_argument("--native-audio", action="store_true", help="skip the lossy mp3 re-encode")
    ap.add_argument("--rich", action="store_true",
                    help="add numpy spectral features: key/tempo/mfcc/chroma (needs the 'rich' extra)")
    ap.add_argument("--music", action="store_true",
                    help="add the music_v2 musical-hearing block "
                         "(also enabled by MH_MUSIC_V2)")
    ap.add_argument("--critic", action="store_true",
                    help="add a critic block: metadata + genre hints + evidence brief + prompt "
                         "for naming genre / similar artists / impression")
    ap.add_argument("--llm", action="store_true",
                    help="with --critic, also call an OpenAI-compatible endpoint to fill the verdict")
    ap.add_argument("--llm-base-url", help="OpenAI-compatible base URL (or MH_LLM_BASE_URL)")
    ap.add_argument("--llm-model", help="model id for --llm (or MH_LLM_MODEL)")
    ap.add_argument("--summary", action="store_true", help="print only the one-line summary")
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        prof = profile_music(
            args.source, args.seconds,
            cookies_file=args.cookies,
            ytdlp_bin=args.ytdlp_bin,
            extra_hosts=args.extra_hosts,
            cookies_from_browser=args.cookies_from_browser,
            extractor_args=args.extractor_args,
            remote_components=args.remote_components,
            native_audio=(True if args.native_audio else None),
            rich=args.rich,
            music=args.music,
            critic=(args.critic or args.llm),
            llm=args.llm,
            llm_base_url=args.llm_base_url,
            llm_model=args.llm_model,
        )
    except Exception as exc:  # surface a readable one-line failure
        print(f"music-hearing: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if args.summary:
        print(prof.description.get("summary", ""))
    else:
        print(json.dumps(asdict(prof), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
