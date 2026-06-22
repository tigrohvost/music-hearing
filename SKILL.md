---
name: music-hearing
description: Use when you need to actually "hear" a piece of music — turn a YouTube/YouTube Music URL, an allowlisted host URL, or a search phrase into an acoustic profile (loudness, bands, BPM, spectral centroid) plus a plain-language description ("slow, dark, sub-heavy, dynamic"). Optional rich mode adds key, tempo, MFCC, and chroma. Optional critic mode adds metadata + an evidence brief and prompt so a model can name genre, similar artists, and an impression.
---

# Music Hearing

Give the model ears: fetch a short audio excerpt and measure it, so an agent can
reason about how music *sounds* instead of guessing from titles.

This is a standalone Python package + CLI. It does not depend on any particular
agent runtime — any agent that can run a shell command or import Python can use
it (Claude, openclaw, Hermes, custom).

## When to use

- A user/agent references a track and you want its real character, not a guess.
- Comparing two tracks ("is this like that reference?") — see `compare()`.
- Steering generation toward a sonic target heard from a reference.

## How to use

Prefer the CLI (works for any agent via shell):

```bash
music-hearing "Meg Bowles Organic Lullaby"
music-hearing "https://www.youtube.com/watch?v=EfaFcjpuwkg" --seconds 30
music-hearing "<url-or-search>" --rich          # + key/tempo/mfcc/chroma (needs the rich extra)
music-hearing "<url-or-search>" --critic         # + metadata, genre hints, evidence brief, critic prompt
music-hearing "<url-or-search>" --critic --llm   # also fill genre/similar-artists/impression via an LLM
music-hearing "<url-or-search>" --summary       # just the one-line description
```

Output is JSON: `profile` (numbers), `description` (`summary` + labels),
`rich` when `--rich` is set, and `critic` when `--critic` is set. With
`--summary`, only the one-line summary prints.

To name **genre / similar artists / impression**: run with `--critic`, then have
your own model answer the `critic.prompt` (it embeds the evidence). Or use
`--llm` (with `MH_LLM_BASE_URL` / `MH_LLM_API_KEY` / `MH_LLM_MODEL`) to get a
`critic.verdict` directly.

Or import it:

```python
from music_hearing import profile_music, acoustic_profile, describe, compare
prof = profile_music("Meg Bowles Organic Lullaby", cookies_file="/path/yt-cookies.txt")
print(prof.description["summary"])     # "slow, dark, sub-heavy, dynamic, textured (~67 BPM)"
```

## Prerequisites (runtime, not pip)

- **ffmpeg** — decode audio. Required.
- **yt-dlp** — fetch excerpts. Required for URLs/search. (`--ytdlp-bin` / `MH_YTDLP_BIN` to point at a specific binary.)
- **deno** (or another JS runtime on PATH) — yt-dlp needs it to solve YouTube's
  signature challenge. Without it YouTube returns "Requested format is not
  available". Put `deno` on the PATH the tool's subprocess actually uses (often
  a sanitized `/usr/bin`).
- **YouTube cookies** — a logged-in (ideally throwaway) account exported as a
  Netscape `cookies.txt`, passed via `--cookies` / `MH_COOKIES_FILE`. Never
  commit it. Cookies rotate; re-export periodically.

`--rich` additionally needs numpy (`pip install music-hearing[rich]`).

See `README.md` for the full schema, config (`MH_*` env vars), and gotchas.
