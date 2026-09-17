"""Validation-only quality estimation for frozen respiratory embeddings.

This module deliberately separates signal quality from identity learning.  The
identity encoder is frozen, quality inputs are detached, and the command-line
workflow never accesses the outer-test identities.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .data import Recording, discover_recordings, load_signal
from .metrics import equal_error_rate, roc_points, weighted_roc_points
from .neural_data import (
    PKBatchSampler,
    RespiratoryWindowDataset,
    build_window_references,
    preprocess_window,
)
from .neural_train import build_encoder, resolve_device, set_reproducible_seed
from .protocol import IdentitySplit, make_outer_splits


QUALITY_DESCRIPTOR_NAMES = (
    "missing_fraction",
    "longest_gap_fraction",
    "flatline_fraction",
    "clipping_fraction",
    "derivative_mad",
    "left_trend_slope",
    "right_trend_slope",
    "log_energy_ratio",
    "bilateral_correlation",
    "spectral_entropy",
    "out_of_band_energy_fraction",
    "amplitude_instability",
    "breath_band_plausibility",
    "left_available",
    "right_available",
)


def _longest_true_run(mask: torch.Tensor) -> torch.Tensor:
    """Longest run along the final dimension for a Boolean batch."""

    current = torch.zeros(mask.shape[:-1], device=mask.device, dtype=torch.long)
    longest = current.clone()
    for index in range(mask.shape[-1]):
        current = torch.where(mask[..., index], current + 1, 0)
        longest = torch.maximum(longest, current)
    return longest


def quality_descriptors(windows: torch.Tensor, *, sample_rate_hz: float = 6.0) -> torch.Tensor:
    """Compute finite, identity-agnostic descriptors from normalized windows."""

    if windows.ndim != 3 or windows.shape[1] != 2:
        raise ValueError(f"expected windows with shape (batch, 2, samples), found {tuple(windows.shape)}")
    if windows.shape[-1] < 8:
        raise ValueError("quality descriptors require at least eight samples")
    values = windows.float()
    finite = torch.isfinite(values)
    missing_fraction = 1.0 - finite.float().mean(dim=(1, 2))
    values = torch.where(finite, values, torch.zeros_like(values))
    near_zero = values.abs() < 1e-5
    longest_gap = _longest_true_run(near_zero).amax(dim=1).float() / values.shape[-1]
    differences = torch.diff(values, dim=-1)
    flatline = (differences.abs() < 1e-4).float().mean(dim=(1, 2))
    clipping = (values.abs() >= 11.95).float().mean(dim=(1, 2))
    diff_median = differences.median(dim=-1, keepdim=True).values
    derivative_mad = (differences - diff_median).abs().median(dim=-1).values.mean(dim=1)

    time = torch.linspace(-1.0, 1.0, values.shape[-1], device=values.device)
    denominator = time.square().sum().clamp_min(1e-8)
    centered = values - values.mean(dim=-1, keepdim=True)
    slopes = (centered * time).sum(dim=-1) / denominator
    rms = values.square().mean(dim=-1).sqrt()
    log_energy_ratio = torch.log((rms[:, 0] + 1e-6) / (rms[:, 1] + 1e-6))
    covariance = (centered[:, 0] * centered[:, 1]).sum(dim=-1)
    correlation = covariance / (
        centered[:, 0].square().sum(dim=-1).sqrt()
        * centered[:, 1].square().sum(dim=-1).sqrt()
    ).clamp_min(1e-8)

    power = torch.fft.rfft(values, dim=-1).abs().square()
    power[..., 0] = 0.0
    distribution = power / power.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    entropy = -(distribution * distribution.clamp_min(1e-12).log()).sum(dim=-1)
    entropy = entropy / np.log(max(2, distribution.shape[-1] - 1))
    spectral_entropy = entropy.mean(dim=1)
    frequencies = torch.fft.rfftfreq(
        values.shape[-1], d=1.0 / sample_rate_hz, device=values.device
    )
    breath_band = (frequencies >= 0.08) & (frequencies <= 0.80)
    plausible_power = power[..., breath_band].sum(dim=-1)
    total_power = power.sum(dim=-1).clamp_min(1e-12)
    breath_plausibility = (plausible_power / total_power).mean(dim=1)
    out_of_band = 1.0 - breath_plausibility

    chunks = torch.tensor_split(values, 6, dim=-1)
    chunk_rms = torch.stack(
        [chunk.square().mean(dim=-1).sqrt().mean(dim=1) for chunk in chunks], dim=1
    )
    amplitude_instability = chunk_rms.std(dim=1, unbiased=False) / chunk_rms.mean(
        dim=1
    ).clamp_min(1e-6)
    availability = rms.gt(1e-5).to(values.dtype)

    descriptors = torch.stack(
        (
            missing_fraction,
            longest_gap,
            flatline,
            clipping,
            derivative_mad,
            slopes[:, 0],
            slopes[:, 1],
            log_energy_ratio,
            correlation,
            spectral_entropy,
            out_of_band,
            amplitude_instability,
            breath_plausibility,
            availability[:, 0],
            availability[:, 1],
        ),
        dim=1,
    )
    return torch.nan_to_num(descriptors, nan=0.0, posinf=20.0, neginf=-20.0)


class QualityEstimator(nn.Module):
    """Small quality head over detached embeddings and explicit descriptors."""

    def __init__(self, embedding_dimension: int, hidden_dimension: int = 128) -> None:
        super().__init__()
        descriptor_dimension = len(QUALITY_DESCRIPTOR_NAMES)
        self.register_buffer("descriptor_mean", torch.zeros(descriptor_dimension))
        self.register_buffer("descriptor_scale", torch.ones(descriptor_dimension))
        self.network = nn.Sequential(
            nn.Linear(embedding_dimension + descriptor_dimension, hidden_dimension),
            nn.LayerNorm(hidden_dimension),
            nn.SiLU(),
            nn.Dropout(0.10),
            nn.Linear(hidden_dimension, 32),
            nn.SiLU(),
            nn.Linear(32, 1),
        )

    def set_descriptor_statistics(self, descriptors: torch.Tensor) -> None:
        if descriptors.ndim != 2 or descriptors.shape[1] != len(QUALITY_DESCRIPTOR_NAMES):
            raise ValueError("descriptor matrix has the wrong shape")
        with torch.no_grad():
            self.descriptor_mean.copy_(descriptors.float().mean(dim=0))
            self.descriptor_scale.copy_(
                descriptors.float().std(dim=0, unbiased=False).clamp_min(1e-4)
            )

    def forward(self, embeddings: torch.Tensor, descriptors: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2 or descriptors.ndim != 2 or len(embeddings) != len(descriptors):
            raise ValueError("embeddings and descriptors must be aligned matrices")
        standardized = (descriptors.float() - self.descriptor_mean) / self.descriptor_scale
        inputs = torch.cat((embeddings.detach().float(), standardized), dim=1)
        return torch.sigmoid(self.network(inputs)).squeeze(1)


def corrupt_windows(
    windows: torch.Tensor,
    severity: int,
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply controlled, progressively stronger acquisition degradation."""

    if windows.ndim != 3 or windows.shape[1] != 2:
        raise ValueError("windows must have shape (batch, 2, samples)")
    if severity not in (1, 2, 3):
        raise ValueError("severity must be 1 (mild), 2 (moderate), or 3 (severe)")
    if windows.device.type != "cpu" and generator is not None:
        raise ValueError("an explicit generator is supported for CPU corruption only")
    noise_scale = (0.04, 0.12, 0.30)[severity - 1]
    gain_scale = (0.06, 0.16, 0.35)[severity - 1]
    maximum_gap = (4, 12, 30)[severity - 1]
    drift_scale = (0.03, 0.10, 0.25)[severity - 1]
    result = windows.clone().float()
    gains = 1.0 + gain_scale * torch.randn(
        (len(result), 2, 1), device=result.device, generator=generator
    )
    result = result * gains
    result = result + noise_scale * torch.randn(
        result.shape, device=result.device, generator=generator
    )
    time = torch.linspace(-1.0, 1.0, result.shape[-1], device=result.device)
    drift = drift_scale * torch.randn(
        (len(result), 2, 1), device=result.device, generator=generator
    )
    result = result + drift * time
    gap_lengths = torch.randint(
        1, maximum_gap + 1, (len(result),), device=result.device, generator=generator
    )
    for index, length_tensor in enumerate(gap_lengths):
        length = int(length_tensor)
        start = int(
            torch.randint(
                0,
                result.shape[-1] - length + 1,
                (),
                device=result.device,
                generator=generator,
            )
        )
        result[index, :, start : start + length] = 0.0
    if severity == 3:
        missing = torch.rand(len(result), device=result.device, generator=generator) < 0.25
        channels = torch.randint(
            0, 2, (len(result),), device=result.device, generator=generator
        )
        for index in torch.nonzero(missing, as_tuple=False).flatten():
            result[int(index), int(channels[int(index)])] = 0.0
    return result.clamp(-12.0, 12.0)


def verification_utility_targets(
    embeddings: torch.Tensor, labels: torch.Tensor, *, temperature: float = 0.10
) -> torch.Tensor:
    """Detached per-window utility from positive and hardest-negative centroids."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if embeddings.ndim != 2 or labels.ndim != 1 or len(embeddings) != len(labels):
        raise ValueError("embeddings and labels must be aligned")
    normalized = F.normalize(embeddings.detach().float(), dim=1)
    unique = torch.unique(labels, sorted=True)
    if len(unique) < 2:
        raise ValueError("utility targets require at least two identities")
    centroids = torch.stack(
        [F.normalize(normalized[labels == label].mean(dim=0), dim=0) for label in unique]
    )
    targets: list[torch.Tensor] = []
    for index, label in enumerate(labels):
        positive_mask = labels == label
        positive_mask[index] = False
        if not positive_mask.any():
            raise ValueError("each identity requires at least two windows")
        positive = F.normalize(normalized[positive_mask].mean(dim=0), dim=0)
        positive_score = normalized[index] @ positive
        negative_centroids = centroids[unique != label]
        negative_score = (negative_centroids @ normalized[index]).max()
        targets.append(torch.sigmoid((positive_score - negative_score) / temperature))
    return torch.stack(targets)


def quality_weighted_pool(
    embeddings: torch.Tensor,
    quality: torch.Tensor,
    *,
    gamma: float = 2.0,
    epsilon: float = 1e-4,
) -> torch.Tensor:
    """Pool a set of embeddings with normalized quality weights."""

    if embeddings.ndim != 2 or quality.ndim != 1 or len(embeddings) != len(quality):
        raise ValueError("embeddings and quality must be aligned")
    if gamma <= 0 or epsilon <= 0:
        raise ValueError("gamma and epsilon must be positive")
    weights = (quality.float().clamp(0.0, 1.0) + epsilon).pow(gamma)
    pooled = (embeddings.float() * weights[:, None]).sum(dim=0) / weights.sum().clamp_min(
        epsilon
    )
    return F.normalize(pooled, dim=0)


def eer_threshold(scores: np.ndarray, labels: np.ndarray) -> float:
    thresholds, far, tar = roc_points(scores, labels)
    return float(thresholds[int(np.argmin(np.abs(far - (1.0 - tar))))])


def class_balanced_risk_coverage(
    scores: np.ndarray,
    labels: np.ndarray,
    quality: np.ndarray,
    *,
    threshold: float | None = None,
    coverages: Iterable[float] = tuple(np.linspace(0.1, 1.0, 10)),
) -> dict[str, object]:
    """Risk after retaining the same top-quality fraction within each class."""

    scores = np.asarray(scores, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    quality = np.asarray(quality, dtype=float).reshape(-1)
    if not (len(scores) == len(labels) == len(quality)) or not len(scores):
        raise ValueError("scores, labels, and quality must have equal non-zero length")
    if not np.isin(labels, (0, 1)).all() or not all((labels == value).any() for value in (0, 1)):
        raise ValueError("labels must contain genuine and impostor trials")
    operating_threshold = eer_threshold(scores, labels) if threshold is None else threshold
    coverage_values = np.asarray(tuple(coverages), dtype=float)
    if len(coverage_values) < 2 or np.any((coverage_values <= 0) | (coverage_values > 1)):
        raise ValueError("at least two coverages in (0, 1] are required")
    risks: list[float] = []
    retained: list[int] = []
    for coverage in coverage_values:
        selections: dict[int, np.ndarray] = {}
        for label in (0, 1):
            indices = np.flatnonzero(labels == label)
            count = max(1, int(np.ceil(coverage * len(indices))))
            order = np.argsort(-quality[indices], kind="stable")
            selections[label] = indices[order[:count]]
        false_acceptance = np.mean(scores[selections[0]] >= operating_threshold)
        false_rejection = np.mean(scores[selections[1]] < operating_threshold)
        risks.append(float(0.5 * (false_acceptance + false_rejection)))
        retained.append(int(len(selections[0]) + len(selections[1])))
    aurc = float(np.trapezoid(risks, coverage_values) / (coverage_values[-1] - coverage_values[0]))
    return {
        "threshold": float(operating_threshold),
        "coverage": coverage_values.tolist(),
        "class_balanced_risk": risks,
        "retained_trials": retained,
        "normalized_aurc_10_to_100": aurc,
    }


def weighted_class_balanced_risk_coverage(
    scores: np.ndarray,
    labels: np.ndarray,
    quality: np.ndarray,
    weights: np.ndarray,
    *,
    threshold: float | None = None,
    coverages: Iterable[float] = tuple(np.linspace(0.1, 1.0, 10)),
) -> dict[str, object]:
    """Weighted selective risk for subject-multiplicity bootstrap replicates."""

    scores = np.asarray(scores, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    quality = np.asarray(quality, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if not (len(scores) == len(labels) == len(quality) == len(weights)) or not len(
        scores
    ):
        raise ValueError("scores, labels, quality, and weights must align")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("weights must be finite and non-negative")
    if not np.isin(labels, (0, 1)).all() or any(
        float(weights[labels == value].sum()) <= 0.0 for value in (0, 1)
    ):
        raise ValueError("both classes must have positive total weight")
    if threshold is None:
        thresholds, far, tar = weighted_roc_points(scores, labels, weights)
        threshold = float(thresholds[int(np.argmin(np.abs(far - (1.0 - tar))))])
    coverage_values = np.asarray(tuple(coverages), dtype=float)
    if len(coverage_values) < 2 or np.any(
        (coverage_values <= 0) | (coverage_values > 1)
    ):
        raise ValueError("at least two coverages in (0, 1] are required")
    risks: list[float] = []
    retained_weights: list[float] = []
    for coverage in coverage_values:
        selections: dict[int, np.ndarray] = {}
        for label in (0, 1):
            indices = np.flatnonzero(labels == label)
            order = np.argsort(-quality[indices], kind="stable")
            ordered = indices[order]
            cumulative = np.cumsum(weights[ordered])
            target = coverage * float(cumulative[-1])
            count = int(np.searchsorted(cumulative, target, side="left")) + 1
            selections[label] = ordered[:count]
        impostor = selections[0]
        genuine = selections[1]
        false_acceptance = float(
            weights[impostor][scores[impostor] >= threshold].sum()
            / weights[impostor].sum()
        )
        false_rejection = float(
            weights[genuine][scores[genuine] < threshold].sum()
            / weights[genuine].sum()
        )
        risks.append(0.5 * (false_acceptance + false_rejection))
        retained_weights.append(float(weights[np.r_[impostor, genuine]].sum()))
    aurc = float(
        np.trapezoid(risks, coverage_values)
        / (coverage_values[-1] - coverage_values[0])
    )
    return {
        "threshold": float(threshold),
        "coverage": coverage_values.tolist(),
        "class_balanced_risk": risks,
        "retained_trial_weight": retained_weights,
        "normalized_aurc_10_to_100": aurc,
    }


@dataclass(slots=True)
class CachedSubject:
    subject_id: str
    windows: torch.Tensor
    embeddings: torch.Tensor
    descriptors: torch.Tensor
    enrollment_indices: torch.Tensor
    probe_indices_by_seconds: dict[int, torch.Tensor]


def _evenly_select(length: int, maximum: int) -> np.ndarray:
    if maximum <= 0 or length <= maximum:
        return np.arange(length, dtype=int)
    return np.linspace(0, length - 1, maximum, dtype=int)


@torch.inference_mode()
def build_validation_cache(
    encoder: nn.Module,
    recordings: list[Recording],
    subjects: Iterable[str],
    *,
    device: torch.device,
    probe_seconds: Iterable[int] = (60, 30),
    guard_minutes: int = 30,
    max_probes_per_subject: int = 30,
    inference_batch_size: int = 512,
    normalization: str = "shared_robust",
    canonicalize_polarity: bool = True,
) -> list[CachedSubject]:
    """Encode validation windows once; no test identity is accepted by this API."""

    by_subject = {
        row.subject_id: row
        for row in recordings
        if row.session_id == "primary" and row.subject_id in set(subjects)
    }
    result: list[CachedSubject] = []
    encoder.eval()
    for subject in sorted(set(subjects)):
        recording = by_subject.get(subject)
        if recording is None:
            continue
        signal = load_signal(recording.path)
        total_windows = len(signal) // 180
        starts = list(range(10))
        groups_by_seconds: dict[int, list[list[int]]] = {}
        for seconds in probe_seconds:
            width = seconds // 30
            first = 10 + guard_minutes * 2
            groups = (total_windows - first) // width
            selected = _evenly_select(groups, max_probes_per_subject)
            groups_by_seconds[seconds] = [
                [first + int(group) * width + within for within in range(width)]
                for group in selected
            ]
            starts.extend(index for group in groups_by_seconds[seconds] for index in group)
        unique_starts = sorted(set(starts))
        position = {start: index for index, start in enumerate(unique_starts)}
        windows = torch.from_numpy(
            np.stack(
                [
                    preprocess_window(
                        signal[start * 180 : (start + 1) * 180],
                        normalization=normalization,
                        canonicalize_polarity=canonicalize_polarity,
                    )
                    for start in unique_starts
                ]
            )
        )
        embedding_parts: list[torch.Tensor] = []
        for offset in range(0, len(windows), inference_batch_size):
            embedding_parts.append(
                encoder(windows[offset : offset + inference_batch_size].to(device)).float().cpu()
            )
        probe_indices = {
            seconds: torch.tensor(
                [[position[index] for index in group] for group in groups], dtype=torch.long
            )
            for seconds, groups in groups_by_seconds.items()
        }
        result.append(
            CachedSubject(
                subject_id=subject,
                windows=windows,
                embeddings=torch.cat(embedding_parts),
                descriptors=quality_descriptors(windows),
                enrollment_indices=torch.tensor([position[index] for index in range(10)]),
                probe_indices_by_seconds=probe_indices,
            )
        )
    if len(result) < 2:
        raise ValueError("at least two validation identities are required")
    return result


@torch.inference_mode()
def trials_from_cache(
    estimator: QualityEstimator,
    cache: list[CachedSubject],
    *,
    device: torch.device,
    probe_seconds: int,
    weighted: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[str, ...]]:
    """Create within-validation verification trials and per-trial quality."""

    estimator.eval()
    enrollment_embeddings: list[torch.Tensor] = []
    enrollment_quality: list[float] = []
    probe_embeddings: list[torch.Tensor] = []
    probe_quality: list[float] = []
    probe_subjects: list[str] = []
    enrollment_subjects: list[str] = []
    for row in cache:
        embeddings = row.embeddings.to(device)
        descriptors = row.descriptors.to(device)
        q = estimator(embeddings, descriptors)
        enroll_indices = row.enrollment_indices.to(device)
        enroll_values = embeddings[enroll_indices]
        enroll_q = q[enroll_indices]
        enrollment_embeddings.append(
            quality_weighted_pool(enroll_values, enroll_q)
            if weighted
            else F.normalize(enroll_values.mean(dim=0), dim=0)
        )
        enrollment_quality.append(float(enroll_q.mean()))
        for group in row.probe_indices_by_seconds[probe_seconds].to(device):
            probe_values = embeddings[group]
            group_q = q[group]
            probe_embeddings.append(
                quality_weighted_pool(probe_values, group_q)
                if weighted
                else F.normalize(probe_values.mean(dim=0), dim=0)
            )
            probe_quality.append(float(group_q.mean()))
            probe_subjects.append(row.subject_id)
        enrollment_subjects.append(row.subject_id)
    enrollments = torch.stack(enrollment_embeddings)
    probes = torch.stack(probe_embeddings)
    scores = (probes @ enrollments.T).float().cpu().numpy().reshape(-1)
    labels = np.asarray(
        [probe == enrollment for probe in probe_subjects for enrollment in enrollment_subjects],
        dtype=np.int8,
    )
    quality_matrix = np.sqrt(
        np.asarray(probe_quality)[:, None] * np.asarray(enrollment_quality)[None, :]
    )
    trial_subjects = tuple(
        probe for probe in probe_subjects for _ in enrollment_subjects
    )
    return scores, labels, quality_matrix.reshape(-1), trial_subjects


@torch.inference_mode()
def controlled_quality_means(
    encoder: nn.Module,
    estimator: QualityEstimator,
    cache: list[CachedSubject],
    *,
    device: torch.device,
    maximum_windows: int = 256,
    seed: int = 2027,
) -> dict[str, float]:
    windows = torch.cat([row.windows for row in cache])
    indices = _evenly_select(len(windows), maximum_windows)
    clean = windows[indices]
    generator = torch.Generator().manual_seed(seed)
    variants = [clean] + [corrupt_windows(clean, severity, generator=generator) for severity in (1, 2, 3)]
    names = ("clean", "mild", "moderate", "severe")
    means: dict[str, float] = {}
    encoder.eval()
    estimator.eval()
    for name, variant in zip(names, variants, strict=True):
        embeddings = encoder(variant.to(device)).float()
        descriptors = quality_descriptors(variant).to(device)
        means[name] = float(estimator(embeddings, descriptors).mean())
    return means


@torch.inference_mode()
def quality_only_identity_control(
    estimator: QualityEstimator,
    cache: list[CachedSubject],
    *,
    device: torch.device,
) -> dict[str, float | int]:
    """Nearest scalar-quality centroid identity classification on validation only."""

    centroids: list[float] = []
    probes: list[float] = []
    targets: list[int] = []
    estimator.eval()
    for class_index, row in enumerate(cache):
        quality = estimator(row.embeddings.to(device), row.descriptors.to(device)).cpu()
        enrollment = quality[row.enrollment_indices]
        probe_indices = row.probe_indices_by_seconds[30].flatten()
        centroids.append(float(enrollment.mean()))
        probes.extend(quality[probe_indices].tolist())
        targets.extend([class_index] * len(probe_indices))
    distances = np.abs(np.asarray(probes)[:, None] - np.asarray(centroids)[None, :])
    predictions = distances.argmin(axis=1)
    accuracy = float(np.mean(predictions == np.asarray(targets)))
    return {
        "identities": len(cache),
        "probe_windows": len(probes),
        "nearest_centroid_accuracy": accuracy,
        "chance_accuracy": 1.0 / len(cache),
    }


def _make_training_cache(
    encoder: nn.Module,
    loader: DataLoader,
    sampler: PKBatchSampler,
    *,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return [clean,mild,moderate,severe] embeddings/descriptors and utilities."""

    embeddings_all: list[torch.Tensor] = []
    descriptors_all: list[torch.Tensor] = []
    utilities_all: list[torch.Tensor] = []
    labels_all: list[torch.Tensor] = []
    generator = torch.Generator().manual_seed(seed)
    sampler.set_epoch(0)
    encoder.eval()
    with torch.inference_mode():
        for windows, labels in loader:
            variants = [windows] + [
                corrupt_windows(windows, severity, generator=generator)
                for severity in (1, 2, 3)
            ]
            stacked = torch.cat(variants).to(device)
            embedded = encoder(stacked).float().cpu().reshape(4, len(windows), -1)
            described = torch.stack([quality_descriptors(variant) for variant in variants])
            utility = verification_utility_targets(embedded[0], labels)
            embeddings_all.append(embedded)
            descriptors_all.append(described)
            utilities_all.append(utility)
            labels_all.append(labels)
    return (
        torch.cat(embeddings_all, dim=1),
        torch.cat(descriptors_all, dim=1),
        torch.cat(utilities_all),
        torch.cat(labels_all),
    )


def _quality_epoch(
    estimator: QualityEstimator,
    embeddings: torch.Tensor,
    descriptors: torch.Tensor,
    utility: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    *,
    device: torch.device,
    batch_size: int,
    ranking_margin: float,
    utility_weight: float,
    generator: torch.Generator,
) -> dict[str, float]:
    estimator.train()
    order = torch.randperm(embeddings.shape[1], generator=generator)
    losses: list[float] = []
    ranks: list[float] = []
    utilities: list[float] = []
    for offset in range(0, len(order), batch_size):
        index = order[offset : offset + batch_size]
        embedded = embeddings[:, index].to(device)
        described = descriptors[:, index].to(device)
        target = utility[index].to(device)
        optimizer.zero_grad(set_to_none=True)
        quality = torch.stack(
            [estimator(embedded[level], described[level]) for level in range(4)]
        )
        ranking_loss = sum(
            F.relu(ranking_margin - quality[level - 1] + quality[level]).mean()
            for level in range(1, 4)
        ) / 3.0
        utility_loss = F.huber_loss(quality[0], target)
        loss = ranking_loss + utility_weight * utility_loss
        loss.backward()
        nn.utils.clip_grad_norm_(estimator.parameters(), 5.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        ranks.append(float(ranking_loss.detach()))
        utilities.append(float(utility_loss.detach()))
    return {
        "loss": float(np.mean(losses)),
        "ranking_loss": float(np.mean(ranks)),
        "utility_loss": float(np.mean(utilities)),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--encoder-checkpoint", type=Path, default=Path("reports/v3_bie_dev_fold0.pt"))
    parser.add_argument("--encoder-report", type=Path, default=Path("reports/v3_bie_dev_fold0.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/v3_quality_dev_fold0.json"))
    parser.add_argument("--fold", type=int, default=0, choices=range(5))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--cache-batches", type=int, default=100)
    parser.add_argument("--subjects-per-batch", type=int, default=16)
    parser.add_argument("--windows-per-subject", type=int, default=4)
    parser.add_argument("--maximum-training-windows-per-recording", type=int, default=600)
    parser.add_argument("--quality-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--ranking-margin", type=float, default=0.08)
    parser.add_argument("--utility-weight", type=float, default=1.0)
    parser.add_argument("--guard-minutes", type=int, default=30)
    parser.add_argument("--max-validation-probes", type=int, default=30)
    parser.add_argument("--inference-batch-size", type=int, default=512)
    parser.add_argument("--cache-recordings", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--split-seed", type=int, default=2027)
    return parser.parse_args(argv)


def _load_frozen_encoder(
    checkpoint_path: Path, report_path: Path, device: torch.device, fold: int
) -> tuple[nn.Module, dict[str, object]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("fold") != fold or report.get("evaluations") != []:
        raise ValueError("encoder report must be the matching validation-only fold report")
    if "outer-test identities not scored" not in report.get("evaluation_status", ""):
        raise ValueError("encoder report does not prove validation-only screening")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arguments = checkpoint["args"]
    if checkpoint.get("fold") != fold or arguments.get("model") != "v3_bie":
        raise ValueError("quality gate requires the promoted fold-matched v3_bie checkpoint")
    encoder = build_encoder(
        arguments["model"],
        embedding_dimension=arguments["embedding_dimension"],
        branch_dimension=arguments["branch_dimension"],
        base_channels=arguments["base_channels"],
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    encoder.eval()
    encoder.requires_grad_(False)
    return encoder, arguments


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if min(args.epochs, args.cache_batches, args.quality_batch_size) < 1:
        raise SystemExit("epochs, cache batches, and batch size must be positive")
    set_reproducible_seed(args.seed)
    device = resolve_device(args.device)
    encoder, encoder_args = _load_frozen_encoder(
        args.encoder_checkpoint, args.encoder_report, device, args.fold
    )
    recordings = [
        row
        for row in discover_recordings(args.dataset_root, states=("wake",))
        if row.session_id == "primary"
    ]
    subjects = sorted({row.subject_id for row in recordings})
    split: IdentitySplit = make_outer_splits(subjects, seed=args.split_seed)[args.fold]
    references = build_window_references(
        recordings,
        split.train,
        maximum_windows_per_recording=args.maximum_training_windows_per_recording,
    )
    labels = {subject: index for index, subject in enumerate(split.train)}
    dataset = RespiratoryWindowDataset(
        references,
        labels,
        normalization=encoder_args["normalization"],
        canonicalize_polarity=encoder_args["channel_polarity"] == "correlation",
        cache_recordings=args.cache_recordings,
    )
    sampler = PKBatchSampler(
        references,
        subjects_per_batch=args.subjects_per_batch,
        windows_per_subject=args.windows_per_subject,
        batches_per_epoch=args.cache_batches,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    print(f"device={device} caching frozen embeddings from {len(split.train)} training identities")
    train_embeddings, train_descriptors, utilities, _ = _make_training_cache(
        encoder, loader, sampler, device=device, seed=args.seed
    )
    validation_cache = build_validation_cache(
        encoder,
        recordings,
        split.validation,
        device=device,
        guard_minutes=args.guard_minutes,
        max_probes_per_subject=args.max_validation_probes,
        inference_batch_size=args.inference_batch_size,
        normalization=encoder_args["normalization"],
        canonicalize_polarity=encoder_args["channel_polarity"] == "correlation",
    )
    estimator = QualityEstimator(encoder_args["embedding_dimension"]).to(device)
    estimator.set_descriptor_statistics(train_descriptors.flatten(0, 1).to(device))
    optimizer = torch.optim.AdamW(
        estimator.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    generator = torch.Generator().manual_seed(args.seed + 1)
    history: list[dict[str, object]] = []
    best_score = float("inf")
    best_epoch = 0
    checkpoint_path = args.output.with_suffix(".pt")
    for epoch in range(args.epochs):
        training = _quality_epoch(
            estimator,
            train_embeddings,
            train_descriptors,
            utilities,
            optimizer,
            device=device,
            batch_size=args.quality_batch_size,
            ranking_margin=args.ranking_margin,
            utility_weight=args.utility_weight,
            generator=generator,
        )
        validation_eers: dict[int, float] = {}
        for seconds in (60, 30):
            scores, trial_labels, _, _ = trials_from_cache(
                estimator,
                validation_cache,
                device=device,
                probe_seconds=seconds,
                weighted=True,
            )
            validation_eers[seconds] = equal_error_rate(scores, trial_labels)
        selection_score = validation_eers[60] + 0.5 * validation_eers[30]
        row = {
            "epoch": epoch + 1,
            **training,
            "validation_eer_by_probe_seconds": {
                str(key): value for key, value in validation_eers.items()
            },
            "validation_selection_score": selection_score,
        }
        history.append(row)
        print(
            f"epoch={epoch + 1} loss={training['loss']:.4f} "
            f"eer60={100 * validation_eers[60]:.2f}% "
            f"eer30={100 * validation_eers[30]:.2f}% score={selection_score:.4f}"
        )
        if selection_score < best_score:
            best_score = selection_score
            best_epoch = epoch + 1
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "quality_estimator": estimator.state_dict(),
                    "fold": args.fold,
                    "epoch": best_epoch,
                    "validation_selection_score": best_score,
                    "encoder_checkpoint": str(args.encoder_checkpoint.resolve()),
                    "args": vars(args),
                },
                checkpoint_path,
            )
        scheduler.step()

    selected = torch.load(checkpoint_path, map_location=device, weights_only=False)
    estimator.load_state_dict(selected["quality_estimator"])
    evaluations: dict[str, object] = {}
    all_aurc_improved = True
    mean_selection = 0.0
    weighted_selection = 0.0
    for seconds, selection_weight in ((60, 1.0), (30, 0.5)):
        method_results: dict[str, object] = {}
        for name, weighted in (("mean", False), ("quality_weighted", True)):
            scores, trial_labels, trial_quality, _ = trials_from_cache(
                estimator,
                validation_cache,
                device=device,
                probe_seconds=seconds,
                weighted=weighted,
            )
            risk = class_balanced_risk_coverage(scores, trial_labels, trial_quality)
            method_results[name] = {
                "eer": equal_error_rate(scores, trial_labels),
                "risk_coverage": risk,
            }
        mean_eer = method_results["mean"]["eer"]
        weighted_eer = method_results["quality_weighted"]["eer"]
        mean_selection += selection_weight * mean_eer
        weighted_selection += selection_weight * weighted_eer
        mean_aurc = method_results["mean"]["risk_coverage"]["normalized_aurc_10_to_100"]
        weighted_aurc = method_results["quality_weighted"]["risk_coverage"]["normalized_aurc_10_to_100"]
        all_aurc_improved &= weighted_aurc < mean_aurc
        evaluations[str(seconds)] = method_results

    severity_means = controlled_quality_means(
        encoder, estimator, validation_cache, device=device, seed=args.seed + 2
    )
    monotonic = all(
        severity_means[left] > severity_means[right]
        for left, right in zip(
            ("clean", "mild", "moderate"), ("mild", "moderate", "severe"), strict=True
        )
    )
    passed = weighted_selection < mean_selection and all_aurc_improved and monotonic
    payload = {
        "protocol": f"fold-{args.fold} validation-only quality prototype over frozen v3_bie embeddings",
        "fold": args.fold,
        "split_seed": args.split_seed,
        "training_seed": args.seed,
        "encoder_checkpoint": str(args.encoder_checkpoint.resolve()),
        "quality_checkpoint": str(checkpoint_path.resolve()),
        "encoder_frozen": True,
        "embedding_detached": True,
        "natural_quality_labels_used": False,
        "train_identities": len(split.train),
        "validation_identities": len(split.validation),
        "test_identities": len(split.test),
        "outer_test_identities_scored": 0,
        "training_cached_windows": int(train_embeddings.shape[1]),
        "descriptor_names": QUALITY_DESCRIPTOR_NAMES,
        "best_epoch": best_epoch,
        "best_validation_selection_score": best_score,
        "history": history,
        "validation_evaluations": evaluations,
        "mean_pooling_selection_score": mean_selection,
        "quality_weighted_selection_score": weighted_selection,
        "controlled_corruption_mean_quality": severity_means,
        "controlled_corruption_strictly_monotonic": monotonic,
        "quality_only_identity_negative_control": quality_only_identity_control(
            estimator, validation_cache, device=device
        ),
        "gate": {
            "quality_weighted_selection_improved": weighted_selection < mean_selection,
            "quality_weighted_aurc_improved_both_durations": all_aurc_improved,
            "controlled_corruption_monotonic": monotonic,
            "passed": passed,
            "decision": "retain_quality_prototype" if passed else "downscope_quality_claim",
        },
        "limitations": [
            f"All results use fold-{args.fold} validation identities; outer-test identities were not loaded or scored.",
            "No reliable natural quality annotations were available, so supervision uses detached verification utility and fixed controlled corruptions only.",
            "This gate evaluates quality-weighted mean pooling; the proposal's learned mean-plus-standard-deviation projection remains deferred.",
            "Missingness descriptors operate after preprocessing and therefore detect explicit zero/dropout regions, not unrecovered raw acquisition metadata.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["gate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
