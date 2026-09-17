"""OpenBreath-ID research utilities."""

from .data import SAMPLE_RATE_HZ, WINDOW_SECONDS, Recording, discover_recordings, load_signal
from .protocol import IdentitySplit, make_outer_splits

__all__ = [
    "SAMPLE_RATE_HZ",
    "WINDOW_SECONDS",
    "IdentitySplit",
    "Recording",
    "discover_recordings",
    "load_signal",
    "make_outer_splits",
]

__version__ = "0.1.0"

