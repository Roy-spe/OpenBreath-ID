"""Subject-cluster uncertainty for verification metrics."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .metrics import (
    calibration_metrics,
    equal_error_rate,
    minimum_detection_cost,
    rates_at_threshold,
)


def subject_multiway_trial_weights(
    enrollment_subjects: Sequence[str],
    probe_subjects: Sequence[str],
    labels: np.ndarray,
    multiplicities: dict[str, int],
) -> np.ndarray:
    """Map identity bootstrap multiplicities to genuine/impostor trial weights."""

    enrollment = np.asarray(enrollment_subjects, dtype=object).reshape(-1)
    probe = np.asarray(probe_subjects, dtype=object).reshape(-1)
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    if not (len(enrollment) == len(probe) == len(labels)) or not len(labels):
        raise ValueError("enrollment subjects, probe subjects, and labels must align")
    if not np.isin(labels, (0, 1)).all():
        raise ValueError("labels must be zero or one")
    if np.any((labels == 1) != (enrollment == probe)):
        raise ValueError("trial labels do not match enrollment/probe identities")
    if any(value < 0 or int(value) != value for value in multiplicities.values()):
        raise ValueError("multiplicities must be non-negative integers")
    missing = (set(enrollment) | set(probe)) - set(multiplicities)
    if missing:
        raise ValueError(f"missing identity multiplicities: {sorted(missing)}")
    enrollment_weight = np.asarray(
        [multiplicities[str(subject)] for subject in enrollment], dtype=float
    )
    probe_weight = np.asarray(
        [multiplicities[str(subject)] for subject in probe], dtype=float
    )
    return np.where(labels == 1, probe_weight, enrollment_weight * probe_weight)


def subject_cluster_bootstrap(
    scores: np.ndarray,
    labels: np.ndarray,
    trial_subjects: Sequence[str],
    *,
    probabilities: np.ndarray,
    operating_threshold: float,
    replicates: int = 2000,
    confidence_level: float = 0.95,
    seed: int = 2027,
) -> dict[str, object]:
    """Bootstrap probe identities while retaining every dependent trial per identity."""

    scores = np.asarray(scores, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    subjects = np.asarray(trial_subjects, dtype=object).reshape(-1)
    if not (len(scores) == len(labels) == len(probabilities) == len(subjects)):
        raise ValueError("scores, labels, probabilities, and trial_subjects must align")
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    unique_subjects = np.unique(subjects)
    if len(unique_subjects) < 2:
        raise ValueError("at least two probe subjects are required")

    cluster_indices = [np.flatnonzero(subjects == subject) for subject in unique_subjects]
    rng = np.random.default_rng(seed)
    values = {
        name: np.empty(replicates, dtype=float)
        for name in ("eer", "normalized_min_dcf_p01", "tar", "far", "nll", "brier", "ece")
    }
    for replicate in range(replicates):
        selected = rng.integers(0, len(cluster_indices), size=len(cluster_indices))
        indices = np.concatenate([cluster_indices[index] for index in selected])
        sampled_scores = scores[indices]
        sampled_labels = labels[indices]
        sampled_probabilities = probabilities[indices]
        values["eer"][replicate] = equal_error_rate(sampled_scores, sampled_labels)
        values["normalized_min_dcf_p01"][replicate] = minimum_detection_cost(
            sampled_scores, sampled_labels
        )
        tar, far = rates_at_threshold(
            sampled_scores, sampled_labels, operating_threshold
        )
        values["tar"][replicate] = tar
        values["far"][replicate] = far
        calibration = calibration_metrics(sampled_probabilities, sampled_labels)
        for name in ("nll", "brier", "ece"):
            values[name][replicate] = calibration[name]

    tail = (1.0 - confidence_level) / 2.0
    intervals = {
        name: {
            "lower": float(np.quantile(samples, tail)),
            "upper": float(np.quantile(samples, 1.0 - tail)),
        }
        for name, samples in values.items()
    }
    return {
        "method": "percentile bootstrap over probe-subject clusters",
        "resampling_unit": "probe identity with all genuine and impostor trials retained",
        "replicates": replicates,
        "confidence_level": confidence_level,
        "seed": seed,
        "intervals": intervals,
    }
