---
name: music-hearing
description: Use when you need to actually "hear" a piece of music from a YouTube/YouTube Music URL, allowlisted host URL, search phrase, or local audio: produce an acoustic profile and plain-language sound description. Use --music for a deeper musical read (rhythm/groove, structure, harmony ambiguity, timbre families, lo-fi proxies, and a 64-dim similarity embedding). Use --rich for FFT key/tempo/MFCC/chroma and --critic for metadata plus an evidence prompt so a model can name genre, similar artists, and an impression. By default, when a user asks you to hear/listen to a track, also voice a short reaction in character as an erudite, meticulous, skeptical music critic.
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
- Creating compact musical reference embeddings with `--music`.

## How to use

Prefer the CLI (works for any agent via shell):

```bash
music-hearing "Meg Bowles Organic Lullaby"
music-hearing "https://www.youtube.com/watch?v=EfaFcjpuwkg" --seconds 30
music-hearing "<url-or-search>" --rich          # + key/tempo/mfcc/chroma (needs the rich extra)
music-hearing "<url-or-search>" --music         # + music_v2 rhythm/structure/harmony/timbre/embedding
music-hearing "<url-or-search>" --critic         # + metadata, genre hints, evidence brief, critic prompt
music-hearing "<url-or-search>" --critic --llm   # also fill genre/similar-artists/impression via an LLM
music-hearing "<url-or-search>" --summary       # just the one-line description
```

Output is JSON: `profile` (numbers), `description` (`summary` + labels),
`rich` when `--rich` is set, `music_v2`/`music_description` when `--music` or
`MH_MUSIC_V2=1` is set, and `critic` when `--critic` is set. With `--summary`,
only the one-line summary prints.

To name **genre / similar artists / impression**: run with `--critic`, then have
your own model answer the `critic.prompt` (it embeds the evidence). Or use
`--llm` (with `MH_LLM_BASE_URL` / `MH_LLM_API_KEY` / `MH_LLM_MODEL`) to get a
`critic.verdict` directly.

## Default behavior: be the critic

When a user asks you to **hear / listen to** a track, the data is not the whole
answer. Run with `--critic` by default and, on top of the JSON, add a **short
(2–4 sentence) reaction in character** — an *erudite, meticulous, skeptical*
music critic:

- Ground every claim in the actual evidence: key, tempo, harmonic/percussive
  balance, band weights, timbre families. No vibes that the numbers don't back.
- Place the track in its lineage and name the genre concretely; push back on
  clichés and hype rather than echoing them.
- Be skeptical and flag the limits: this is a short excerpt, and search-derived
  metadata often has a null artist/empty description — say so instead of
  overclaiming. Never invent facts about the artist or release.
- Keep it short and opinionated, not a press release.

`critic.prompt` already carries this persona, so a standalone `--llm` run or any
agent answering the embedded prompt produces the same voice.

Or import it:

```python
from music_hearing import profile_music, acoustic_profile, describe, compare
prof = profile_music("Meg Bowles Organic Lullaby", cookies_file="/path/yt-cookies.txt", music=True)
print(prof.description["summary"])         # base acoustic description
print(prof.music_description["summary"])   # deeper musical labels (--music)
```

## Prerequisites (runtime, not pip)

- **ffmpeg** — decode audio. Required.
- **yt-dlp** — fetch excerpts. Required for URLs/search. (`--ytdlp-bin` / `MH_YTDLP_BIN` to point at a specific binary.)
- **deno** (or another JS runtime on PATH) — yt-dlp needs it to solve YouTube's
  signature challenge. Without it YouTube returns "Requested format is not
  available". Put `deno` on the PATH the tool's subprocess actually uses (often
  a sanitized `/usr/bin`).
- **EJS challenge solver** — recent yt-dlp builds no longer ship YouTube's JS
  challenge solver inline; even with deno they fail the same way plus a
  "Signature solving failed" / "n challenge solving failed" warning. Pass
  `--remote-components ejs:github` (or `MH_REMOTE_COMPONENTS=ejs:github`) to
  fetch the solver from yt-dlp's own GitHub releases. Off by default so older
  yt-dlp builds that reject the flag keep working.
- **YouTube cookies** — a logged-in (ideally throwaway) account exported as a
  Netscape `cookies.txt`, passed via `--cookies` / `MH_COOKIES_FILE`. Never
  commit it. Cookies rotate; re-export periodically.

`--rich`, `--music`, and `--critic` spectral evidence need numpy
(`pip install music-hearing[rich]`).

## Music v2 guardrails

- Treat `music_v2` as additive. Do not remove or reinterpret legacy `profile`
  and `description` fields for existing callers.
- Trust `music_v2.rhythm` over legacy `estimated_bpm` when groove matters.
- Treat harmony key/chord hints as confidence-scored hints, not ground truth.
- Store `music_v2.embedding` and compact JSON summaries if needed; do not store
  fetched audio excerpts.
- Do not add lyrics, ASR, transcription, word semantics, speaker recognition,
  source separation, or long audio retention to this workflow.

See `README.md` for the full schema, config (`MH_*` env vars), and gotchas.
