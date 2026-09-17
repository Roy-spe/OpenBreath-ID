import numpy as np

from openbreath_id.metrics import (
    LogisticCalibrator,
    calibration_metrics,
    equal_error_rate,
    minimum_detection_cost,
    rates_at_threshold,
    threshold_at_far,
    weighted_equal_error_rate,
    weighted_minimum_detection_cost,
    weighted_rates_at_threshold,
    weighted_threshold_at_far,
)


def test_perfect_scores_have_zero_eer_and_detection_cost() -> None:
    scores = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 1, 0, 0])
    assert equal_error_rate(scores, labels) == 0.0
    assert minimum_detection_cost(scores, labels) == 0.0
    threshold = threshold_at_far(scores, labels, target_far=0.01)
    assert rates_at_threshold(scores, labels, threshold) == (1.0, 0.0)


def test_logistic_calibration_returns_probabilities_and_metrics() -> None:
    scores = np.array([1.2, 0.8, 0.6, -0.2, -0.5, -1.0])
    labels = np.array([1, 1, 1, 0, 0, 0])
    calibrator = LogisticCalibrator.fit(scores, labels)
    probabilities = calibrator.predict_proba(scores)
    metrics = calibration_metrics(probabilities, labels)
    assert np.all((probabilities > 0) & (probabilities < 1))
    assert probabilities[:3].mean() > probabilities[3:].mean()
    assert set(metrics) == {"nll", "brier", "ece"}


def test_weighted_metrics_match_explicit_integer_replication() -> None:
    scores = np.array([0.9, 0.7, 0.6, 0.4, 0.2, 0.1])
    labels = np.array([1, 0, 1, 0, 1, 0])
    weights = np.array([2, 1, 3, 2, 1, 4])
    repeated_scores = np.repeat(scores, weights)
    repeated_labels = np.repeat(labels, weights)
    assert weighted_equal_error_rate(scores, labels, weights) == equal_error_rate(
        repeated_scores, repeated_labels
    )
    assert weighted_minimum_detection_cost(
        scores, labels, weights
    ) == minimum_detection_cost(repeated_scores, repeated_labels)
    threshold = weighted_threshold_at_far(scores, labels, weights, target_far=0.25)
    assert threshold == threshold_at_far(
        repeated_scores, repeated_labels, target_far=0.25
    )
    assert weighted_rates_at_threshold(
        scores, labels, weights, threshold
    ) == rates_at_threshold(repeated_scores, repeated_labels, threshold)
