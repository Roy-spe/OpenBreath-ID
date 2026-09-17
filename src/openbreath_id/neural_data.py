"""Leakage-safe 30-second window datasets for neural identity learning."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from .data import Recording, load_signal, read_signal_metadata


@dataclass(frozen=True, slots=True)
class WindowReference:
    subject_id: str
    path: Path
    start_sample: int


def build_window_references(
    recordings: Iterable[Recording],
    subjects: Iterable[str],
    *,
    window_samples: int = 180,
    stride_samples: int = 180,
    maximum_windows_per_recording: int = 0,
) -> list[WindowReference]:
    """Index complete windows without loading the underlying signal arrays."""

    if window_samples < 1 or stride_samples < 1:
        raise ValueError("window_samples and stride_samples must be positive")
    allowed = set(subjects)
    references: list[WindowReference] = []
    for recording in recordings:
        if recording.subject_id not in allowed:
            continue
        metadata = read_signal_metadata(recording.path)
        starts = np.arange(
            0, metadata.n_samples - window_samples + 1, stride_samples, dtype=int
        )
        if maximum_windows_per_recording > 0 and len(starts) > maximum_windows_per_recording:
            selected = np.linspace(
                0, len(starts) - 1, maximum_windows_per_recording, dtype=int
            )
            starts = starts[selected]
        references.extend(
            WindowReference(recording.subject_id, recording.path, int(start))
            for start in starts
        )
    return references


def preprocess_window(
    window: np.ndarray,
    *,
    normalization: str = "robust",
    canonicalize_polarity: bool = True,
    channel_view: str = "bilateral",
) -> np.ndarray:
    """Prepare one bilateral window using only samples inside that window."""

    values = np.asarray(window, dtype=np.float32).copy()
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError(f"window must have shape (samples, 2), found {values.shape}")
    for channel in range(2):
        finite = np.isfinite(values[:, channel])
        fill = float(np.median(values[finite, channel])) if finite.any() else 0.0
        values[~finite, channel] = fill
    if channel_view == "bilateral":
        active_channels = (0, 1)
    elif channel_view == "channel_1":
        active_channels = (0,)
        values[:, 1] = 0.0
    elif channel_view == "channel_2":
        active_channels = (1,)
        values[:, 0] = 0.0
    else:
        raise ValueError("channel_view must be 'bilateral', 'channel_1', or 'channel_2'")
    if canonicalize_polarity and channel_view == "bilateral":
        left = values[:, 0] - values[:, 0].mean()
        right = values[:, 1] - values[:, 1].mean()
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        if denominator > 1e-8 and float(left @ right) / denominator < 0.0:
            values[:, 1] *= -1.0

    if normalization == "robust":
        center = np.median(values, axis=0, keepdims=True)
        q25, q75 = np.percentile(values, (25, 75), axis=0, keepdims=True)
        scale = q75 - q25
        fallback = values.std(axis=0, keepdims=True)
        scale = np.where(scale > 1e-6, scale, np.where(fallback > 1e-6, fallback, 1.0))
        values = (values - center) / scale
    elif normalization == "shared_robust":
        center = np.median(values, axis=0, keepdims=True)
        mad = np.median(np.abs(values - center), axis=0)
        shared_scale = float(np.median(mad[list(active_channels)]))
        if shared_scale <= 1e-6:
            shared_scale = float(np.median(values[:, list(active_channels)].std(axis=0)))
        if shared_scale <= 1e-6:
            shared_scale = 1.0
        values = (values - center) / shared_scale
    elif normalization == "zscore":
        center = values.mean(axis=0, keepdims=True)
        scale = values.std(axis=0, keepdims=True)
        values = (values - center) / np.where(scale > 1e-6, scale, 1.0)
    elif normalization != "none":
        raise ValueError(
            "normalization must be 'none', 'zscore', 'robust', or 'shared_robust'"
        )
    return np.clip(values, -12.0, 12.0).T.astype(np.float32, copy=False)


def augment_training_window(window: torch.Tensor) -> torch.Tensor:
    """Apply mild, label-preserving perturbations to one normalized window."""

    if window.ndim != 2 or window.shape[0] != 2:
        raise ValueError(f"window must have shape (2, samples), found {tuple(window.shape)}")
    values = window.clone()

    # Respiratory identity should not depend on the starting phase of the crop.
    shift = int(torch.randint(-18, 19, ()).item())
    values = torch.roll(values, shifts=shift, dims=-1)

    # Model modest channel-specific calibration drift and sensor noise.
    gains = torch.empty((2, 1), dtype=values.dtype).uniform_(0.9, 1.1)
    values = values * gains
    values = values + 0.03 * torch.randn_like(values)

    # Mask at most two seconds of a 30-second, 6-Hz crop.
    mask_samples = int(torch.randint(0, 13, ()).item())
    if mask_samples:
        start = int(
            torch.randint(0, values.shape[-1] - mask_samples + 1, ()).item()
        )
        values[:, start : start + mask_samples] = 0.0
    return values


def sample_channel_dropout_view(
    *, probability: float = 0.15, draw: float | None = None
) -> str:
    """Select a bilateral or single-channel training view without using labels."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("channel-dropout probability must be between zero and one")
    value = float(torch.rand(())) if draw is None else float(draw)
    if not 0.0 <= value < 1.0:
        raise ValueError("draw must be in [0, 1)")
    if value >= probability:
        return "bilateral"
    return "channel_1" if value < probability / 2.0 else "channel_2"


def augment_robustness_window(
    window: torch.Tensor,
    profile: str,
    *,
    swap_draw: float | None = None,
    gain_draw: float | None = None,
    gain_choice: int | None = None,
    gap_draw: float | None = None,
    gap_choice: int | None = None,
) -> torch.Tensor:
    """Apply a frozen label-preserving robustness profile to one normalized crop."""

    if profile not in {"channel_swap", "robust_mixed"}:
        raise ValueError("profile must be 'channel_swap' or 'robust_mixed'")
    if window.ndim != 2 or window.shape[0] != 2:
        raise ValueError(f"window must have shape (2, samples), found {tuple(window.shape)}")
    values = window.clone()
    swap_value = float(torch.rand(())) if swap_draw is None else float(swap_draw)
    if not 0.0 <= swap_value < 1.0:
        raise ValueError("swap_draw must be in [0, 1)")
    if swap_value < 0.5:
        values = values.flip(0)
    if profile == "channel_swap":
        return values

    gain_value = float(torch.rand(())) if gain_draw is None else float(gain_draw)
    gap_value = float(torch.rand(())) if gap_draw is None else float(gap_draw)
    if not 0.0 <= gain_value < 1.0 or not 0.0 <= gap_value < 1.0:
        raise ValueError("gain_draw and gap_draw must be in [0, 1)")
    if gain_value < 0.5:
        choice = int(torch.randint(0, 4, ()).item()) if gain_choice is None else int(gain_choice)
        if choice not in range(4):
            raise ValueError("gain_choice must be in range(4)")
        channel = choice // 2
        factor = 0.5 if choice % 2 == 0 else 2.0
        values[channel] *= factor
    if gap_value < 0.3:
        choice = int(torch.randint(0, 2, ()).item()) if gap_choice is None else int(gap_choice)
        if choice not in range(2):
            raise ValueError("gap_choice must be 0 or 1")
        width = 30 if choice == 0 else 90
        if width > values.shape[-1]:
            raise ValueError("gap width exceeds the training crop")
        begin = (values.shape[-1] - width) // 2
        median = values.median(dim=-1, keepdim=True).values
        values[:, begin : begin + width] = median
    return values


class _SignalCache:
    def __init__(self, maximum_recordings: int = 8) -> None:
        self.maximum_recordings = maximum_recordings
        self.values: OrderedDict[Path, np.ndarray] = OrderedDict()

    def get(self, path: Path) -> np.ndarray:
        if path in self.values:
            signal = self.values.pop(path)
            self.values[path] = signal
            return signal
        signal = load_signal(path)
        self.values[path] = signal
        while len(self.values) > self.maximum_recordings:
            self.values.popitem(last=False)
        return signal


class RespiratoryWindowDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    def __init__(
        self,
        references: Sequence[WindowReference],
        label_by_subject: dict[str, int],
        *,
        window_samples: int = 180,
        normalization: str = "robust",
        canonicalize_polarity: bool = True,
        augmentation: str = "none",
        channel_view: str = "bilateral",
        cache_recordings: int = 8,
    ) -> None:
        self.references = tuple(references)
        self.label_by_subject = dict(label_by_subject)
        self.window_samples = window_samples
        self.normalization = normalization
        self.canonicalize_polarity = canonicalize_polarity
        if channel_view not in {"bilateral", "channel_1", "channel_2"}:
            raise ValueError(
                "channel_view must be 'bilateral', 'channel_1', or 'channel_2'"
            )
        self.channel_view = channel_view
        if augmentation not in {
            "none",
            "mild",
            "channel_dropout",
            "channel_swap",
            "robust_mixed",
        }:
            raise ValueError(
                "augmentation must be 'none', 'mild', 'channel_dropout', "
                "'channel_swap', or 'robust_mixed'"
            )
        if augmentation == "channel_dropout" and channel_view != "bilateral":
            raise ValueError("channel_dropout augmentation requires bilateral channel_view")
        self.augmentation = augmentation
        self.cache = _SignalCache(cache_recordings)
        missing = {row.subject_id for row in self.references} - set(self.label_by_subject)
        if missing:
            raise ValueError(f"subjects missing from label map: {sorted(missing)}")

    def __len__(self) -> int:
        return len(self.references)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        reference = self.references[index]
        signal = self.cache.get(reference.path)
        end = reference.start_sample + self.window_samples
        effective_channel_view = (
            sample_channel_dropout_view()
            if self.augmentation == "channel_dropout"
            else self.channel_view
        )
        window = preprocess_window(
            signal[reference.start_sample:end],
            normalization=self.normalization,
            canonicalize_polarity=self.canonicalize_polarity,
            channel_view=effective_channel_view,
        )
        tensor = torch.from_numpy(window)
        if self.augmentation == "mild":
            tensor = augment_training_window(tensor)
        elif self.augmentation in {"channel_swap", "robust_mixed"}:
            tensor = augment_robustness_window(tensor, self.augmentation)
        return (
            tensor,
            torch.tensor(self.label_by_subject[reference.subject_id], dtype=torch.long),
        )


class PKBatchSampler(Sampler[list[int]]):
    """Draw P identities and K windows per identity in every training batch."""

    def __init__(
        self,
        references: Sequence[WindowReference],
        *,
        subjects_per_batch: int,
        windows_per_subject: int,
        batches_per_epoch: int,
        seed: int = 2027,
    ) -> None:
        if min(subjects_per_batch, windows_per_subject, batches_per_epoch) < 1:
            raise ValueError("P, K, and batches_per_epoch must be positive")
        by_subject: dict[str, list[int]] = {}
        for index, reference in enumerate(references):
            by_subject.setdefault(reference.subject_id, []).append(index)
        if len(by_subject) < subjects_per_batch:
            raise ValueError("subjects_per_batch exceeds the number of indexed identities")
        self.by_subject = {key: np.asarray(value) for key, value in by_subject.items()}
        self.subjects = np.asarray(sorted(by_subject))
        self.subjects_per_batch = subjects_per_batch
        self.windows_per_subject = windows_per_subject
        self.batches_per_epoch = batches_per_epoch
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.batches_per_epoch

    def __iter__(self) -> Iterator[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        for _ in range(self.batches_per_epoch):
            subjects = rng.choice(
                self.subjects, size=self.subjects_per_batch, replace=False
            )
            batch: list[int] = []
            for subject in subjects:
                candidates = self.by_subject[str(subject)]
                selected = rng.choice(
                    candidates,
                    size=self.windows_per_subject,
                    replace=len(candidates) < self.windows_per_subject,
                )
                batch.extend(int(index) for index in selected)
            yield batch
