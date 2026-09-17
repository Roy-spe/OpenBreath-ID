"""Dataset discovery, validation, and short-window access."""

from __future__ import annotations

from dataclasses import dataclass
import csv
from pathlib import Path
import re
from typing import Iterator, Sequence

import numpy as np
from scipy.io import loadmat, whosmat


SAMPLE_RATE_HZ = 6.0
WINDOW_SECONDS = 30.0
SIGNAL_VARIABLE = "fieldValue"
SUBJECT_PATTERN = re.compile(r"^subj_\d{3}$")


@dataclass(frozen=True, slots=True)
class Recording:
    """One state-specific recording session belonging to a canonical identity."""

    subject_id: str
    session_id: str
    state: str
    path: Path


@dataclass(frozen=True, slots=True)
class SignalMetadata:
    variable: str
    n_samples: int
    n_channels: int
    matlab_class: str

    @property
    def duration_seconds(self) -> float:
        return self.n_samples / SAMPLE_RATE_HZ


def discover_recordings(
    dataset_root: str | Path,
    states: Sequence[str] = ("wake", "sleep"),
    repeat_session_map: dict[str, str] | None = None,
    unmapped_repeat_policy: str = "error",
) -> list[Recording]:
    """Discover all sessions while preserving the top-level canonical subject ID."""

    root = Path(dataset_root).resolve()
    requested = {state.lower() for state in states}
    if not requested or not requested <= {"wake", "sleep"}:
        raise ValueError("states must contain 'wake', 'sleep', or both")
    if unmapped_repeat_policy not in {"error", "skip", "parent"}:
        raise ValueError("unmapped_repeat_policy must be 'error', 'skip', or 'parent'")

    recordings: list[Recording] = []
    for path in root.rglob("*.mat"):
        state = path.stem.lower()
        if state not in requested:
            continue
        relative = path.relative_to(root)
        if len(relative.parts) < 2 or not SUBJECT_PATTERN.match(relative.parts[0]):
            continue
        subject_id = relative.parts[0]
        session_parts = relative.parts[1:-1]
        session_id = "primary" if not session_parts else "/".join(session_parts)
        if session_parts and repeat_session_map is not None:
            directory_key = "/".join(relative.parts[:-1])
            if directory_key not in repeat_session_map:
                if unmapped_repeat_policy == "skip":
                    continue
                if unmapped_repeat_policy == "error":
                    raise ValueError(
                        f"repeat-session map has no entry for {directory_key!r}"
                    )
            else:
                subject_id = repeat_session_map[directory_key]
        recordings.append(
            Recording(
                subject_id=subject_id,
                session_id=session_id,
                state=state,
                path=path,
            )
        )
    return sorted(recordings, key=lambda item: (item.subject_id, item.session_id, item.state))


def read_repeat_session_map(path: str | Path) -> dict[str, str]:
    """Read a verified mapping from repeat directories to canonical IDs."""

    mapping: dict[str, str] = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"relative_directory", "canonical_subject_id"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError(
                "repeat-session map must contain relative_directory and canonical_subject_id columns"
            )
        for row in reader:
            directory = row["relative_directory"].strip().replace("\\", "/").strip("/")
            subject = row["canonical_subject_id"].strip()
            if not directory or not SUBJECT_PATTERN.match(subject):
                raise ValueError(f"invalid repeat-session map row: {row}")
            if directory in mapping:
                raise ValueError(f"duplicate repeat-session map entry for {directory!r}")
            mapping[directory] = subject
    if not mapping:
        raise ValueError("repeat-session map is empty")
    return mapping


def read_signal_metadata(path: str | Path, variable: str = SIGNAL_VARIABLE) -> SignalMetadata:
    """Read the MATLAB directory without materializing the signal array."""

    entries = {name: (shape, matlab_class) for name, shape, matlab_class in whosmat(path)}
    if variable not in entries:
        raise ValueError(f"{path}: MATLAB variable {variable!r} was not found")
    shape, matlab_class = entries[variable]
    if len(shape) != 2:
        raise ValueError(f"{path}: expected a 2-D signal, found shape {shape}")
    if shape[1] == 2:
        n_samples, n_channels = shape
    elif shape[0] == 2:
        n_channels, n_samples = shape
    else:
        raise ValueError(f"{path}: expected exactly two nostril channels, found shape {shape}")
    return SignalMetadata(variable, int(n_samples), int(n_channels), matlab_class)


def load_signal(path: str | Path, variable: str = SIGNAL_VARIABLE) -> np.ndarray:
    """Load a bilateral signal as a finite-or-NaN float array shaped `(samples, 2)`."""

    payload = loadmat(path, variable_names=[variable])
    if variable not in payload:
        raise ValueError(f"{path}: MATLAB variable {variable!r} was not found")
    signal = np.asarray(payload[variable], dtype=np.float64)
    if signal.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D signal, found shape {signal.shape}")
    if signal.shape[1] == 2:
        return signal
    if signal.shape[0] == 2:
        return signal.T
    raise ValueError(f"{path}: expected exactly two nostril channels, found shape {signal.shape}")


def iter_windows(
    signal: np.ndarray,
    *,
    window_seconds: float = WINDOW_SECONDS,
    stride_seconds: float | None = None,
    sample_rate_hz: float = SAMPLE_RATE_HZ,
    require_finite: bool = True,
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield `(start_sample, window)` pairs; incomplete trailing windows are omitted."""

    values = np.asarray(signal)
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"signal must have shape (samples, 2), found {values.shape}")
    if window_seconds <= 0 or sample_rate_hz <= 0:
        raise ValueError("window_seconds and sample_rate_hz must be positive")
    stride_seconds = window_seconds if stride_seconds is None else stride_seconds
    if stride_seconds <= 0:
        raise ValueError("stride_seconds must be positive")

    width = int(round(window_seconds * sample_rate_hz))
    stride = int(round(stride_seconds * sample_rate_hz))
    if width < 1 or stride < 1:
        raise ValueError("window and stride must contain at least one sample")

    for start in range(0, values.shape[0] - width + 1, stride):
        window = values[start : start + width]
        if require_finite and not np.isfinite(window).all():
            continue
        yield start, window
