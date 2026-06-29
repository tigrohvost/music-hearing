# music-hearing

Hear a track: turn a YouTube / YouTube Music URL, an allowlisted host URL, or a
search phrase into an **acoustic profile**, a **plain-language description**,
and optionally a bounded `music_v2` musical-hearing profile — so an agent (or
you) can reason about how music actually *sounds* instead of guessing from the
title.

```
$ music-hearing "Meg Bowles Organic Lullaby" --summary
slow, dark, sub-heavy, dynamic, textured (~67 BPM)
```

Agent-agnostic: a standalone Python package + CLI with a portable `SKILL.md`. No
runtime is assumed — Claude, openclaw, Hermes, or a custom agent can all call it
via the shell or by importing it.

## In an agent's hands

An agent with this skill connected, asked about a track:

```text
✻ Claude Code · music-hearing skill enabled
───────────────────────────────────────────────────────────────────────────────

> what's this track, and what does it actually sound like?
  https://www.youtube.com/watch?v=EfaFcjpuwkg

● Not going to review it from the title. Let me listen.

● Bash(music-hearing "…/watch?v=EfaFcjpuwkg" --critic --rich --music)
  ⎿  summary : slow, dark, sub-heavy, dynamic (~67 BPM)
     key     : A minor (0.90)  ·  tempo ~107 BPM  ·  harmonic 0.73 / perc 0.27
     v2      : blurred pulse, sparse rhythm, loop-like, A minor, warm_pad/drone
     artist  : Carbon Based Lifeforms — "Hydroponic Garden"
     hints   : ambient · downtempo · psybient

● A slow A-minor exhalation for people who find Brian Eno a touch too eventful.
  Carbon Based Lifeforms spend these ~107 imaginary BPM buffing one sub-bass
  swell until it gleams — 73% harmonic, essentially beatless, a lava lamp with a
  graduate degree. It is genuinely gorgeous and it is wallpaper, and it knows
  exactly which one you need it to be at 2 a.m. Spa music for the Ableton crowd,
  fine — you will still let the whole album run and pretend you were "working."

  Genre    ambient / psybient, with a downtempo undertow
  Like     Solar Fields · Sync24 · Aes Dana · early Biosphere
  Verdict  beautiful background music that would be offended to hear you say so

> ▌
```

The verdict leads with one paragraph — opinionated, but pinned to the actual
numbers (A-minor, ~107 BPM, 73% harmonic, beatless). The agent ran the skill,
read the `critic` block, and wrote that in **its own voice** — no LLM is baked
into the tool. (Standalone, no agent? add `--llm`.)

## Install

### 1. The Python package

```bash
pip install git+https://github.com/tigrohvost/music-hearing.git          # base (stdlib only)
pip install "music-hearing[rich] @ git+https://github.com/tigrohvost/music-hearing.git"   # + numpy spectral/music_v2
```

Or from a clone:

```bash
git clone https://github.com/tigrohvost/music-hearing.git
cd music-hearing
pip install .            # base
pip install .[rich]      # + numpy for key/tempo/mfcc/chroma/music_v2
```

Python 3.10+. The base package has **zero** Python dependencies; `[rich]` adds
numpy for `--rich`, `--music`, and `--critic` spectral evidence. The CLI entry
point `music-hearing` is installed on your PATH.

### 2. Runtime prerequisites (system tools, not pip)

| Tool   | Needed for | Without it |
|--------|------------|-----------|
| **ffmpeg** | decoding any non-WAV audio | `ffmpeg is required to profile non-WAV audio` |
| **yt-dlp** | fetching from a URL/search | `yt-dlp is required for music hearing` |
| **deno** (or another JS runtime) | YouTube's signature challenge | YouTube returns `Requested format is not available` |
| **cookies** | YouTube auth | YouTube returns `Sign in to confirm you're not a bot` |

```bash
# Debian/Ubuntu
sudo apt install ffmpeg
# yt-dlp — use a recent build (2025.06+); the distro one is often too old
sudo curl -L https://github.com/yt-dlp/yt-dlp/releases/latest/download/yt-dlp -o /usr/local/bin/yt-dlp && sudo chmod +x /usr/local/bin/yt-dlp
# deno — a JS runtime yt-dlp uses to solve YouTube's challenge
curl -fsSL https://deno.land/install.sh | sh        # puts deno in ~/.deno/bin
```

> **deno must be on the PATH that actually runs yt-dlp.** Some agents run tool
> subprocesses with a sanitized `PATH` (e.g. just `/usr/bin`). If YouTube keeps
> saying "Requested format is not available", symlink deno somewhere always on
> PATH: `sudo ln -s "$(command -v deno)" /usr/bin/deno`. If GitHub's release CDN
> is blocked, deno also publishes at `https://dl.deno.land/`.
>
> **Recent yt-dlp also needs the EJS challenge solver.** Builds since the EJS
> rework no longer ship the solver inline — without it you get the same
> "Requested format is not available" plus a `Signature solving failed` /
> `n challenge solving failed` warning, even with deno present. Pass
> `--remote-components ejs:github` (or set `MH_REMOTE_COMPONENTS=ejs:github`) so
> yt-dlp fetches the solver from its own GitHub releases. It is off by default
> because older yt-dlp builds reject the unknown flag.

Point at a specific yt-dlp binary with `--ytdlp-bin` / `MH_YTDLP_BIN` if it
isn't named `yt-dlp` on PATH.

### 3. YouTube cookies

YouTube bot-gates anonymous downloads (`Sign in to confirm you're not a bot`),
so the tool needs a logged-in session exported as a Netscape `cookies.txt`. You
only need this for YouTube; archive.org / direct URLs don't.

**Use a throwaway Google account.** yt-dlp-style downloading can get an account
rate-limited, flagged, or locked — don't use your main account.

**Export the cookies** (browser extension, most reliable):

1. Install a Netscape `cookies.txt` exporter, e.g. **"Get cookies.txt LOCALLY"**
   (Chrome/Chromium/Firefox).
2. Log the throwaway account into `https://www.youtube.com` in a normal window
   and play a few seconds of any video so the full session cookies are set.
3. Click the extension → **Export** → save `yt-cookies.txt`.

> **Why not just copy the browser's cookie DB?** Incognito cookies live only in
> the browser's memory (never written to the on-disk profile), and the on-disk
> DB is keyring-encrypted — only an in-browser extension can export a usable
> file. If you prefer incognito (yt-dlp's anti-rotation trick: log in, open a new
> tab, close the original, export, then close the window), you must first enable
> the extension for incognito in your browser's extension settings.

**Give it to the tool** — flag or env (the flag wins):

```bash
music-hearing "<url-or-search>" --cookies /path/to/yt-cookies.txt
# or
export MH_COOKIES_FILE=/path/to/yt-cookies.txt
music-hearing "<url-or-search>"
```

**Keep it safe.** Treat `yt-cookies.txt` like a password — it *is* a live login.

```bash
chmod 600 yt-cookies.txt          # owner-only
```

Store it outside any repo. This project's `.gitignore` already blocks
`*cookies*.txt` and `secrets/`, but double-check before committing anywhere else.

**Cookies expire / rotate** — every few weeks YouTube invalidates them and you'll
see `Sign in to confirm you're not a bot` again. Just re-export and replace the
file; no other change needed (a missing/expired file is ignored, so the
archive.org path keeps working).

### 4. Verify

```bash
music-hearing "https://archive.org/details/some-public-audio" --extra-hosts archive.org --summary   # no cookies needed
music-hearing "Carbon Based Lifeforms Comsat" --cookies ./yt-cookies.txt --summary                  # YouTube path
```

## Usage

### CLI

```bash
music-hearing "Meg Bowles Organic Lullaby"
music-hearing "https://www.youtube.com/watch?v=EfaFcjpuwkg" --seconds 30 --cookies ./yt-cookies.txt
music-hearing "https://archive.org/details/some-item" --extra-hosts archive.org
music-hearing "<url-or-search>" --rich          # add key/tempo/mfcc/chroma
music-hearing "<url-or-search>" --music         # add music_v2 rhythm/structure/harmony/timbre/embedding
music-hearing "<url-or-search>" --critic         # genre/similar-artist/impression brief + prompt
music-hearing "<url-or-search>" --critic --llm   # fill the verdict via an LLM (see config)
music-hearing "<url-or-search>" --summary       # one-line description only
```

### Library

```python
from music_hearing import profile_music, acoustic_profile, describe, compare

# fetch + profile a remote track
prof = profile_music("Meg Bowles Organic Lullaby", cookies_file="yt-cookies.txt")
print(prof.description["summary"])
prof = profile_music("Meg Bowles Organic Lullaby", cookies_file="yt-cookies.txt", music=True)
print(prof.music_description["summary"])

# profile a local file directly (no network)
p = describe(vars(acoustic_profile("track.mp3")))

# compare two profiles
print(compare(vars(acoustic_profile("a.mp3")), vars(acoustic_profile("b.mp3"))))
```

## Music v2 analysis (`--music`)

`--music` adds the additive `music_v2` contract from Rain's music-hearing
workflow. It is not a replacement for the old `profile` and `description`
fields; those stay stable for existing callers.

```jsonc
"music_v2": {
  "schema": "music_hearing_profile.v2",
  "rhythm":    { "bpm": ..., "beat_grid": ..., "density": ..., "syncopation": ... },
  "structure": { "sections": ..., "arc": ..., "novelty": ... },
  "harmony":   { "key": ..., "mode": ..., "key_ambiguity": ..., "chord_hints": ... },
  "timbre":    { "families": ..., "attack_profile": ..., "spectral_texture": ... },
  "lofi":      { "hiss_level": ..., "wow_flutter_proxy": ..., "noise_floor_dbfs_est": ... },
  "embedding": { "schema": "music_embedding.handcrafted.v1", "dim": 64, "values": [...] }
}
```

Use it when you need musical understanding: groove, arrangement shape, harmony
ambiguity, mixture-level timbre families, low-fidelity artifact proxies, or a
compact similarity vector. It does **not** do lyrics, ASR, speaker recognition,
source separation, or long audio retention.

`--music` needs numpy, so install `music-hearing[rich]`. The compatibility env
flag `RAIN_HEARING_MUSIC_V2=1` and the standalone alias `MH_MUSIC_V2=1` include
`music_v2` even when the caller does not pass `--music`.

## Music-critic analysis (`--critic`)

Genre, similar artists, and impressions are world-knowledge judgments — the
acoustics alone can't honestly produce them. `--critic` therefore gives a model
everything it needs to make that call:

```jsonc
"critic": {
  "metadata":    { "title": ..., "artist": ..., "album": ..., "tags": [...] },  // from yt-dlp
  "genre_hints": ["ambient", "downtempo"],   // coarse acoustic guesses (not a verdict)
  "brief":       "Acoustics: slow, dark, sub-heavy ... Spectral: key ~D minor ... Metadata: artist ...",
  "prompt":      "You are a seasoned music critic. ... Return JSON {genre, similar_artists, impression}"
}
```

An **agent** reads `critic` and answers with its own model (no LLM dependency in
the tool — stays agent-agnostic). For **standalone** use, `--llm` calls an
OpenAI-compatible endpoint and adds `critic.verdict` =
`{genre, similar_artists, impression}`:

```bash
export MH_LLM_BASE_URL=https://api.openai.com/v1
export MH_LLM_API_KEY=sk-...
music-hearing "Meg Bowles Organic Lullaby" --critic --llm --llm-model gpt-4o-mini
```

The critic prompt instructs the model to ground genre in the audio evidence,
only name artists it's confident about, and avoid inventing.

## Configuration

Every CLI flag has an `MH_*` environment fallback (the flag wins):

| Flag | Env | Meaning |
|------|-----|---------|
| `--cookies` | `MH_COOKIES_FILE` | Netscape cookies.txt for YouTube auth |
| `--ytdlp-bin` | `MH_YTDLP_BIN` | path to the yt-dlp binary |
| `--extra-hosts` | `MH_EXTRA_HOSTS` | extra allowed hosts (comma/space), e.g. `archive.org,freemusicarchive.org` |
| `--cookies-from-browser` | `MH_COOKIES_FROM_BROWSER` | read cookies from a local browser profile |
| `--extractor-args` | `MH_EXTRACTOR_ARGS` | raw yt-dlp `--extractor-args` |
| `--remote-components` | `MH_REMOTE_COMPONENTS` | raw yt-dlp `--remote-components`, e.g. `ejs:github` (YouTube JS challenge solver) |
| `--native-audio` | `MH_NATIVE_AUDIO` | skip the lossy mp3 re-encode |
| `--music` | `MH_MUSIC_V2` | include additive `music_v2`; install `[rich]` |
| `--llm-base-url` | `MH_LLM_BASE_URL` | OpenAI-compatible base URL for `--llm` |
| `--llm-model` | `MH_LLM_MODEL` | model id for `--llm` |
| (env only) | `RAIN_HEARING_MUSIC_V2` | compatibility flag to include `music_v2` |
| (env only) | `MH_LLM_API_KEY` | bearer key for `--llm` (never a CLI flag) |

A missing cookies file is ignored (the `--cookies` flag is dropped) so the
non-YouTube path keeps working.

## Output schema

`profile_music` returns / the CLI prints:

- `source`, `resolved_input`, `extractor` (`youtube-url` / `url` / `ytsearch1`)
- `seconds_requested`, `yt_dlp_version`, `stale_warning`
- `profile` — `rms_dbfs`, `peak`, `crest_factor`, `dynamic_range_db`,
  `zero_crossing_rate`, `spectral_centroid_hz`, `bands` (sub_bass/bass/low_mid/mid/high),
  `estimated_bpm`
- `description` — `summary` plus `brightness` / `weight` / `tempo_feel` /
  `dynamics` / `texture`
- `rich` (only with `--rich`) — true FFT centroid/rolloff/flatness/bandwidth,
  `key` (Krumhansl), `tempo` (octave-corrected, with confidence), `mfcc`,
  `chroma`, `lufs_approx`, `hpss`, `stereo`, `timeline`
- `music_v2` (only with `--music` / env flag) — `rhythm`, `structure`,
  `harmony`, `timbre`, `lofi`, 64-dim handcrafted `embedding`, quality flags
- `music_description` (with `music_v2`) — compact labels for groove, density,
  arrangement, key hint, timbre families, and lo-fi artifacts
- `critic` (only with `--critic`) — `metadata`, `genre_hints`, `brief`,
  `prompt`, and (with `--llm`) `verdict` = `{genre, similar_artists, impression}`

## How it works

1. **fetch** — yt-dlp pulls only the first N seconds to a temp file (always
   direct, `--proxy ""`, so an inherited app proxy can't break the media fetch).
2. **dsp** — decode to 8 kHz mono (ffmpeg), measure loudness/dynamics/ZCR, a
   5-band Goertzel sketch, a coarse centroid, and an autocorrelation BPM.
3. **semantics** — map the numbers to words and a one-line summary.
4. **spectral** *(optional)* — numpy FFT for the detailed view.
5. **music_v2** *(optional)* — bounded mixture-level musical analysis and a
   compact handcrafted embedding.

## Gotchas

- **"Requested format is not available"** → no JS runtime (put `deno` on PATH) or, on recent yt-dlp, no EJS solver (add `--remote-components ejs:github`).
- **"Sign in to confirm you're not a bot"** → cookies missing/expired; re-export.
- **Hangs / connection errors only inside an agent** → the agent's env had a
  proxy; this tool forces `--proxy ""`, but check that the host has direct egress.
- **`--music` returns a warning envelope** → install `music-hearing[rich]`; the
  v2 submodules need numpy and degrade instead of crashing the caller.
- Cookies expire — re-export every few weeks. Never commit them (`.gitignore`
  already blocks `*cookies*.txt` and `secrets/`).

## License

MIT.
