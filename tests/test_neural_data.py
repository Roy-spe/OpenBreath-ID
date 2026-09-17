from pathlib import Path

import numpy as np
import torch

from openbreath_id.neural_data import (
    PKBatchSampler,
    WindowReference,
    augment_robustness_window,
    augment_training_window,
    preprocess_window,
    sample_channel_dropout_view,
)


def test_preprocess_window_canonicalizes_negative_channel_correlation() -> None:
    left = np.sin(np.linspace(0, 4 * np.pi, 180))
    raw = np.column_stack((left, -left))
    processed = preprocess_window(raw, normalization="zscore", canonicalize_polarity=True)
    assert processed.shape == (2, 180)
    assert np.corrcoef(processed)[0, 1] > 0.999


def test_mild_augmentation_preserves_shape_and_input() -> None:
    torch.manual_seed(7)
    window = torch.linspace(-1.0, 1.0, 360).reshape(2, 180)
    original = window.clone()
    augmented = augment_training_window(window)

    assert augmented.shape == window.shape
    assert torch.isfinite(augmented).all()
    assert torch.equal(window, original)
    assert not torch.equal(augmented, window)


def test_shared_robust_normalization_preserves_relative_channel_scale() -> None:
    phase = np.linspace(0, 4 * np.pi, 180)
    raw = np.column_stack((np.sin(phase), -2.0 * np.sin(phase)))
    processed = preprocess_window(
        raw, normalization="shared_robust", canonicalize_polarity=True
    )
    ratio = float(processed[1].std() / processed[0].std())
    assert np.isclose(ratio, 2.0, rtol=0.02)


def test_unilateral_preprocessing_never_uses_omitted_channel() -> None:
    phase = np.linspace(0, 4 * np.pi, 180)
    left = np.sin(phase)
    first = np.column_stack((left, 2.0 * np.cos(phase)))
    second = np.column_stack((left, 100.0 * np.sign(np.cos(phase))))
    processed_first = preprocess_window(
        first,
        normalization="shared_robust",
        canonicalize_polarity=True,
        channel_view="channel_1",
    )
    processed_second = preprocess_window(
        second,
        normalization="shared_robust",
        canonicalize_polarity=True,
        channel_view="channel_1",
    )
    assert np.array_equal(processed_first, processed_second)
    assert np.count_nonzero(processed_first[1]) == 0


def test_channel_dropout_view_uses_frozen_total_probability() -> None:
    assert sample_channel_dropout_view(probability=0.15, draw=0.01) == "channel_1"
    assert sample_channel_dropout_view(probability=0.15, draw=0.10) == "channel_2"
    assert sample_channel_dropout_view(probability=0.15, draw=0.15) == "bilateral"
    assert sample_channel_dropout_view(probability=0.15, draw=0.90) == "bilateral"


def test_channel_swap_augmentation_is_label_independent_and_non_mutating() -> None:
    window = torch.stack((torch.arange(180), -torch.arange(180))).float()
    original = window.clone()
    swapped = augment_robustness_window(
        window, "channel_swap", swap_draw=0.1
    )
    unchanged = augment_robustness_window(
        window, "channel_swap", swap_draw=0.9
    )
    assert torch.equal(window, original)
    assert torch.equal(swapped, window.flip(0))
    assert torch.equal(unchanged, window)


def test_mixed_robustness_augmentation_applies_frozen_gain_and_gap() -> None:
    window = torch.stack(
        (torch.linspace(-1.0, 1.0, 180), torch.linspace(2.0, -2.0, 180))
    )
    augmented = augment_robustness_window(
        window,
        "robust_mixed",
        swap_draw=0.9,
        gain_draw=0.1,
        gain_choice=1,
        gap_draw=0.1,
        gap_choice=1,
    )
    assert torch.allclose(augmented[0, :45], 2.0 * window[0, :45])
    assert torch.allclose(augmented[1, :45], window[1, :45])
    assert torch.unique(augmented[:, 45:135], dim=1).shape[1] == 1


def test_pk_sampler_draws_requested_examples_per_identity() -> None:
    references = [
        WindowReference(subject, Path(f"{subject}.mat"), start)
        for subject in ("subj_001", "subj_002", "subj_003")
        for start in range(5)
    ]
    sampler = PKBatchSampler(
        references,
        subjects_per_batch=2,
        windows_per_subject=3,
        batches_per_epoch=2,
        seed=7,
    )
    for batch in sampler:
        subjects = [references[index].subject_id for index in batch]
        unique, counts = np.unique(subjects, return_counts=True)
        assert len(unique) == 2
        assert counts.tolist() == [3, 3]
