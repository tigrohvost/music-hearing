"""music-hearing: hear a track -> acoustic profile + plain-language description.

A small, agent-agnostic music perception toolkit. Fetch a short excerpt from a
URL or search phrase (``fetch.profile_music``), or profile a local file
directly (``dsp.acoustic_profile``), then turn the numbers into words
(``semantics.describe``). Optional numpy spectral features (key/tempo/mfcc/
chroma) live in ``spectral`` and load only when requested.
"""
from __future__ import annotations

from .dsp import AcousticProfile, acoustic_profile, load_mono_samples
from .fetch import MusicHearingProfile, profile_music, resolve_source
from .semantics import compare, describe

__version__ = "0.1.0"

__all__ = [
    "AcousticProfile",
    "acoustic_profile",
    "load_mono_samples",
    "MusicHearingProfile",
    "profile_music",
    "resolve_source",
    "describe",
    "compare",
    "__version__",
]
