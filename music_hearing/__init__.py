"""music-hearing: hear a track -> acoustic profile + plain-language description.

A small, agent-agnostic music perception toolkit. Fetch a short excerpt from a
URL or search phrase (``fetch.profile_music``), or profile a local file
directly (``dsp.acoustic_profile``), then turn the numbers into words
(``semantics.describe``). Optional numpy spectral features (key/tempo/mfcc/
chroma) live in ``spectral`` and load only when requested.
"""
from __future__ import annotations

from .critic import critique, genre_hints
from .dsp import AcousticProfile, acoustic_profile, load_mono_samples
from .fetch import MusicHearingProfile, profile_music, resolve_source
from .music_v2 import build_music_v2, empty_music_v2
from .semantics import compare, describe, describe_music_v2

__version__ = "0.3.0"

__all__ = [
    "AcousticProfile",
    "acoustic_profile",
    "load_mono_samples",
    "MusicHearingProfile",
    "profile_music",
    "resolve_source",
    "build_music_v2",
    "empty_music_v2",
    "describe",
    "describe_music_v2",
    "compare",
    "critique",
    "genre_hints",
    "__version__",
]
