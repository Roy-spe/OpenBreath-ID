import numpy as np

from openbreath_id.uncertainty import (
    subject_cluster_bootstrap,
    subject_multiway_trial_weights,
)


def test_subject_cluster_bootstrap_is_reproducible_for_perfect_scores() -> None:
    scores = np.array([0.9, 0.1, 0.8, 0.2, 0.15, 0.85, 0.05, 0.95])
    labels = np.array([1, 0, 1, 0, 0, 1, 0, 1])
    subjects = ("a", "a", "a", "a", "b", "b", "b", "b")
    probabilities = np.where(labels == 1, 0.9, 0.1)

    result = subject_cluster_bootstrap(
        scores,
        labels,
        subjects,
        probabilities=probabilities,
        operating_threshold=0.5,
        replicates=25,
        seed=11,
    )

    assert result["replicates"] == 25
    assert result["intervals"]["eer"] == {"lower": 0.0, "upper": 0.0}
    assert result["intervals"]["tar"] == {"lower": 1.0, "upper": 1.0}
    assert result["intervals"]["far"] == {"lower": 0.0, "upper": 0.0}


def test_subject_multiway_weights_follow_genuine_and_impostor_rules() -> None:
    weights = subject_multiway_trial_weights(
        ("a", "a", "b", "b"),
        ("a", "b", "a", "b"),
        np.array([1, 0, 0, 1]),
        {"a": 2, "b": 3},
    )
    np.testing.assert_array_equal(weights, np.array([2.0, 6.0, 6.0, 3.0]))
