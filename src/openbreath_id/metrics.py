"""Verification operating points and validation-only logistic calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


def _validate_trials(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    if len(scores) != len(labels) or not len(scores):
        raise ValueError("scores and labels must have the same non-zero length")
    if not np.isin(labels, (0, 1)).all() or not (labels == 0).any() or not (labels == 1).any():
        raise ValueError("labels must contain both 0 (impostor) and 1 (genuine)")
    return scores, labels


def roc_points(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return thresholds, FAR, and TAR, including reject-all and accept-all endpoints."""

    scores, labels = _validate_trials(scores, labels)
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    cumulative_positive = np.cumsum(sorted_labels == 1)
    cumulative_negative = np.cumsum(sorted_labels == 0)
    group_ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    positives = int((labels == 1).sum())
    negatives = int((labels == 0).sum())
    thresholds = np.r_[np.inf, sorted_scores[group_ends], -np.inf]
    tar = np.r_[0.0, cumulative_positive[group_ends] / positives, 1.0]
    far = np.r_[0.0, cumulative_negative[group_ends] / negatives, 1.0]
    return thresholds, far, tar


def _validate_weighted_trials(
    scores: np.ndarray, labels: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scores, labels = _validate_trials(scores, labels)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if len(weights) != len(scores):
        raise ValueError("weights must align with scores and labels")
    if not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("weights must be finite and non-negative")
    if any(float(weights[labels == value].sum()) <= 0.0 for value in (0, 1)):
        raise ValueError("both classes must have positive total weight")
    return scores, labels, weights


def weighted_roc_points(
    scores: np.ndarray, labels: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return a ROC whose trial contributions are non-negative weights."""

    scores, labels, weights = _validate_weighted_trials(scores, labels, weights)
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    sorted_weights = weights[order]
    cumulative_positive = np.cumsum(sorted_weights * (sorted_labels == 1))
    cumulative_negative = np.cumsum(sorted_weights * (sorted_labels == 0))
    group_ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    positives = float(weights[labels == 1].sum())
    negatives = float(weights[labels == 0].sum())
    thresholds = np.r_[np.inf, sorted_scores[group_ends], -np.inf]
    tar = np.r_[0.0, cumulative_positive[group_ends] / positives, 1.0]
    far = np.r_[0.0, cumulative_negative[group_ends] / negatives, 1.0]
    return thresholds, far, tar


def equal_error_rate(scores: np.ndarray, labels: np.ndarray) -> float:
    _, far, tar = roc_points(scores, labels)
    frr = 1.0 - tar
    index = int(np.argmin(np.abs(far - frr)))
    return float((far[index] + frr[index]) / 2.0)


def weighted_equal_error_rate(
    scores: np.ndarray, labels: np.ndarray, weights: np.ndarray
) -> float:
    _, far, tar = weighted_roc_points(scores, labels, weights)
    frr = 1.0 - tar
    index = int(np.argmin(np.abs(far - frr)))
    return float((far[index] + frr[index]) / 2.0)


def minimum_detection_cost(
    scores: np.ndarray,
    labels: np.ndarray,
    *,
    target_prior: float = 0.01,
    miss_cost: float = 1.0,
    false_alarm_cost: float = 1.0,
    normalize: bool = True,
) -> float:
    if not 0 < target_prior < 1:
        raise ValueError("target_prior must be between zero and one")
    _, far, tar = roc_points(scores, labels)
    cost = miss_cost * target_prior * (1.0 - tar) + false_alarm_cost * (1.0 - target_prior) * far
    result = float(cost.min())
    if normalize:
        result /= min(miss_cost * target_prior, false_alarm_cost * (1.0 - target_prior))
    return result


def weighted_minimum_detection_cost(
    scores: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    target_prior: float = 0.01,
    miss_cost: float = 1.0,
    false_alarm_cost: float = 1.0,
    normalize: bool = True,
) -> float:
    if not 0 < target_prior < 1:
        raise ValueError("target_prior must be between zero and one")
    _, far, tar = weighted_roc_points(scores, labels, weights)
    cost = (
        miss_cost * target_prior * (1.0 - tar)
        + false_alarm_cost * (1.0 - target_prior) * far
    )
    result = float(cost.min())
    if normalize:
        result /= min(
            miss_cost * target_prior, false_alarm_cost * (1.0 - target_prior)
        )
    return result


def threshold_at_far(scores: np.ndarray, labels: np.ndarray, target_far: float = 0.01) -> float:
    if not 0 <= target_far <= 1:
        raise ValueError("target_far must be between zero and one")
    thresholds, far, tar = roc_points(scores, labels)
    valid = np.flatnonzero(far <= target_far + 1e-12)
    if not len(valid):
        return float("inf")
    best_tar = tar[valid].max()
    candidates = valid[tar[valid] == best_tar]
    return float(thresholds[candidates[-1]])


def weighted_threshold_at_far(
    scores: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    target_far: float = 0.01,
) -> float:
    if not 0 <= target_far <= 1:
        raise ValueError("target_far must be between zero and one")
    thresholds, far, tar = weighted_roc_points(scores, labels, weights)
    valid = np.flatnonzero(far <= target_far + 1e-12)
    if not len(valid):
        return float("inf")
    best_tar = tar[valid].max()
    candidates = valid[tar[valid] == best_tar]
    return float(thresholds[candidates[-1]])


def rates_at_threshold(
    scores: np.ndarray, labels: np.ndarray, threshold: float
) -> tuple[float, float]:
    scores, labels = _validate_trials(scores, labels)
    accepted = scores >= threshold
    tar = float(accepted[labels == 1].mean())
    far = float(accepted[labels == 0].mean())
    return tar, far


def weighted_rates_at_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    threshold: float,
) -> tuple[float, float]:
    scores, labels, weights = _validate_weighted_trials(scores, labels, weights)
    accepted = scores >= threshold
    tar = float(weights[(labels == 1) & accepted].sum() / weights[labels == 1].sum())
    far = float(weights[(labels == 0) & accepted].sum() / weights[labels == 0].sum())
    return tar, far


@dataclass(frozen=True, slots=True)
class LogisticCalibrator:
    slope: float
    intercept: float

    @classmethod
    def fit(
        cls, scores: np.ndarray, labels: np.ndarray, *, l2_strength: float = 1e-4
    ) -> "LogisticCalibrator":
        scores, labels = _validate_trials(scores, labels)

        def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
            slope, intercept = parameters
            logits = np.clip(slope * scores + intercept, -40.0, 40.0)
            probabilities = 1.0 / (1.0 + np.exp(-logits))
            loss = -np.mean(
                labels * np.log(np.clip(probabilities, 1e-12, 1.0))
                + (1 - labels) * np.log(np.clip(1.0 - probabilities, 1e-12, 1.0))
            ) + 0.5 * l2_strength * slope**2
            residual = probabilities - labels
            gradient = np.array(
                [np.mean(residual * scores) + l2_strength * slope, np.mean(residual)]
            )
            return float(loss), gradient

        prior = np.clip(labels.mean(), 1e-6, 1.0 - 1e-6)
        initial = np.array([1.0, np.log(prior / (1.0 - prior))])
        result = minimize(objective, initial, jac=True, method="L-BFGS-B")
        if not result.success:
            raise RuntimeError(f"logistic calibration failed: {result.message}")
        return cls(slope=float(result.x[0]), intercept=float(result.x[1]))

    def predict_proba(self, scores: np.ndarray) -> np.ndarray:
        logits = np.clip(self.slope * np.asarray(scores) + self.intercept, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-logits))


def calibration_metrics(
    probabilities: np.ndarray, labels: np.ndarray, *, bins: int = 10
) -> dict[str, float]:
    probabilities = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int8).reshape(-1)
    if len(probabilities) != len(labels) or not len(labels):
        raise ValueError("probabilities and labels must have the same non-zero length")
    clipped = np.clip(probabilities, 1e-12, 1.0 - 1e-12)
    nll = -np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped))
    brier = np.mean((probabilities - labels) ** 2)
    edges = np.linspace(0.0, 1.0, bins + 1)
    assignments = np.minimum(np.digitize(probabilities, edges[1:-1]), bins - 1)
    ece = 0.0
    for index in range(bins):
        mask = assignments == index
        if mask.any():
            ece += mask.mean() * abs(probabilities[mask].mean() - labels[mask].mean())
    return {"nll": float(nll), "brier": float(brier), "ece": float(ece)}
