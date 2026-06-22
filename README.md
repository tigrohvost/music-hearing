# music-hearing

Hear a track: turn a YouTube / YouTube Music URL, an allowlisted host URL, or a
search phrase into an **acoustic profile** plus a **plain-language description** —
so an agent (or you) can reason about how music actually *sounds* instead of
guessing from the title.

```
$ music-hearing "Meg Bowles Organic Lullaby" --summary
slow, dark, sub-heavy, dynamic, textured (~67 BPM)
```

Agent-agnostic: a standalone Python package + CLI with a portable `SKILL.md`. No
runtime is assumed — Claude, openclaw, Hermes, or a custom agent can all call it
via the shell or by importing it.

## Install

```bash
pip install .            # base (stdlib only)
pip install .[rich]      # + numpy for spectral features (key/tempo/mfcc/chroma)
```

### Runtime prerequisites (not Python packages)

| Tool   | Needed for | Notes |
|--------|------------|-------|
| ffmpeg | decoding any non-WAV audio | required |
| yt-dlp | fetching from a URL/search | required for `profile_music` |
| deno   | YouTube signature challenge | a JS runtime must be on PATH, else YouTube yields "Requested format is not available" |
| cookies| YouTube auth | a logged-in `cookies.txt`; pass with `--cookies` |

## Usage

### CLI

```bash
music-hearing "Meg Bowles Organic Lullaby"
music-hearing "https://www.youtube.com/watch?v=EfaFcjpuwkg" --seconds 30 --cookies ./yt-cookies.txt
music-hearing "https://archive.org/details/some-item" --extra-hosts archive.org
music-hearing "<url-or-search>" --rich          # add key/tempo/mfcc/chroma
music-hearing "<url-or-search>" --summary       # one-line description only
```

### Library

```python
from music_hearing import profile_music, acoustic_profile, describe, compare

# fetch + profile a remote track
prof = profile_music("Meg Bowles Organic Lullaby", cookies_file="yt-cookies.txt")
print(prof.description["summary"])

# profile a local file directly (no network)
p = describe(vars(acoustic_profile("track.mp3")))

# compare two profiles
print(compare(vars(acoustic_profile("a.mp3")), vars(acoustic_profile("b.mp3"))))
```

## Configuration

Every CLI flag has an `MH_*` environment fallback (the flag wins):

| Flag | Env | Meaning |
|------|-----|---------|
| `--cookies` | `MH_COOKIES_FILE` | Netscape cookies.txt for YouTube auth |
| `--ytdlp-bin` | `MH_YTDLP_BIN` | path to the yt-dlp binary |
| `--extra-hosts` | `MH_EXTRA_HOSTS` | extra allowed hosts (comma/space), e.g. `archive.org,freemusicarchive.org` |
| `--cookies-from-browser` | `MH_COOKIES_FROM_BROWSER` | read cookies from a local browser profile |
| `--extractor-args` | `MH_EXTRACTOR_ARGS` | raw yt-dlp `--extractor-args` |
| `--native-audio` | `MH_NATIVE_AUDIO` | skip the lossy mp3 re-encode |

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

## How it works

1. **fetch** — yt-dlp pulls only the first N seconds to a temp file (always
   direct, `--proxy ""`, so an inherited app proxy can't break the media fetch).
2. **dsp** — decode to 8 kHz mono (ffmpeg), measure loudness/dynamics/ZCR, a
   5-band Goertzel sketch, a coarse centroid, and an autocorrelation BPM.
3. **semantics** — map the numbers to words and a one-line summary.
4. **spectral** *(optional)* — numpy FFT for the detailed view.

## Gotchas

- **"Requested format is not available"** → no JS runtime; put `deno` on PATH.
- **"Sign in to confirm you're not a bot"** → cookies missing/expired; re-export.
- **Hangs / connection errors only inside an agent** → the agent's env had a
  proxy; this tool forces `--proxy ""`, but check that the host has direct egress.
- Cookies expire — re-export every few weeks. Never commit them (`.gitignore`
  already blocks `*cookies*.txt` and `secrets/`).

## License

MIT.
