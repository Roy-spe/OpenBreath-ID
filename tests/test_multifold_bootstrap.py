import numpy as np

from openbreath_id.multifold_bootstrap import (
    FrozenScoreTable,
    _draw_multiplicities_by_fold,
    _metrics,
    _prepare_table,
    _prepared_metrics,
    paired_primary_eer_holm,
    paired_multifold_bootstrap,
)


def _tables() -> list[FrozenScoreTable]:
    enrollment = np.array(["a", "a", "b", "b", "c", "c"])
    probe = np.array(["a", "b", "b", "c", "c", "a"])
    labels = (enrollment == probe).astype(np.int8)
    quality = np.linspace(0.9, 0.4, len(labels))
    rows = []
    for fold in (0, 1):
        fold_enrollment = np.char.add(enrollment, str(fold))
        fold_probe = np.char.add(probe, str(fold))
        for seed in (2027, 2028):
            for system, scores in (
                ("better", np.where(labels == 1, 0.9, 0.1)),
                ("worse", np.where(labels == 1, 0.1, 0.9)),
            ):
                rows.append(
                    FrozenScoreTable(
                        system=system,
                        fold=fold,
                        seed=seed,
                        condition="p60",
                        scores=scores,
                        labels=labels,
                        enrollment_subjects=fold_enrollment,
                        probe_subjects=fold_probe,
                        far_threshold=0.5,
                        risk_threshold=0.5,
                        quality=quality,
                    )
                )
    return rows


def _nonlinear_tables() -> list[FrozenScoreTable]:
    enrollment = np.array(["a", "a", "b", "b", "c", "c"])
    probe = np.array(["a", "b", "b", "c", "c", "a"])
    labels = (enrollment == probe).astype(np.int8)
    rows = []
    for fold in (0, 1):
        fold_enrollment = np.char.add(enrollment, str(fold))
        fold_probe = np.char.add(probe, str(fold))
        for seed in (2027, 2028):
            offset = 0.01 * (fold + seed - 2027)
            for system, scores in (
                ("a", np.array([0.7, 0.6, 0.8, 0.3, 0.4, 0.2]) + offset),
                ("b", np.array([0.6, 0.5, 0.7, 0.4, 0.3, 0.2]) + offset),
            ):
                rows.append(
                    FrozenScoreTable(
                        system=system,
                        fold=fold,
                        seed=seed,
                        condition="p60",
                        scores=scores,
                        labels=labels,
                        enrollment_subjects=fold_enrollment,
                        probe_subjects=fold_probe,
                        far_threshold=0.5,
                        risk_threshold=0.5,
                    )
                )
    return rows


def test_paired_multifold_bootstrap_reuses_identity_draws() -> None:
    result = paired_multifold_bootstrap(
        _tables(), comparisons=(("better", "worse"),), replicates=20, seed=11
    )
    better = result["estimates"]["better"]["p60"]["eer"]
    difference = result["paired_differences"]["better_minus_worse"]["p60"]["eer"]
    assert better == {"estimate": 0.0, "lower": 0.0, "upper": 0.0}
    assert difference == {"estimate": -1.0, "lower": -1.0, "upper": -1.0}


def test_threaded_bootstrap_matches_sequential_draws() -> None:
    sequential = paired_multifold_bootstrap(
        _tables(), comparisons=(("better", "worse"),), replicates=20, seed=11
    )
    threaded = paired_multifold_bootstrap(
        _tables(), comparisons=(("better", "worse"),), replicates=20, seed=11,
        workers=2,
    )
    sequential.pop("workers")
    threaded.pop("workers")
    assert threaded == sequential


def test_paired_multifold_bootstrap_rejects_incomplete_design() -> None:
    rows = _tables()
    try:
        paired_multifold_bootstrap(
            rows[:-1], comparisons=(("better", "worse"),), replicates=2
        )
    except ValueError as error:
        assert "complete paired design" in str(error)
    else:
        raise AssertionError("incomplete paired score tables were accepted")


def test_prepared_metrics_match_reference_weighted_metrics() -> None:
    row = _tables()[0]
    weights = np.array([2.0, 1.0, 3.0, 2.0, 1.0, 4.0])
    assert _prepared_metrics(_prepare_table(row), weights) == _metrics(row, weights)


def test_primary_eer_holm_is_reproducible_and_corrects_family() -> None:
    result = paired_primary_eer_holm(
        _tables(), comparisons=(("better", "worse"),), replicates=20, seed=11,
        workers=2,
    )
    assert result["family_size"] == 1
    hypothesis = next(iter(result["hypotheses"].values()))
    assert hypothesis["holm_adjusted_p"] == hypothesis["bootstrap_two_sided_sign_p"]
    assert hypothesis["estimate"] == -1.0


def test_primary_holm_estimate_matches_observed_primary_macro() -> None:
    tables = _nonlinear_tables()
    primary = paired_multifold_bootstrap(
        tables, comparisons=(("a", "b"),), replicates=5, seed=11
    )
    holm = paired_primary_eer_holm(
        tables, comparisons=(("a", "b"),), replicates=100, seed=11
    )
    expected = primary["paired_differences"]["a_minus_b"]["p60"]["eer"][
        "estimate"
    ]
    hypothesis = holm["hypotheses"]["a_minus_b__p60"]
    assert hypothesis["estimate"] == expected == 0.0
    assert hypothesis["bootstrap_mean_estimate_diagnostic"] != expected
    assert "observed fold-macro" in holm["estimate_definition"]


def test_global_identity_draw_shares_multiplicity_across_folds() -> None:
    identities_by_fold = {0: ["a", "b", "c"], 1: ["b", "c", "d"]}
    draw = _draw_multiplicities_by_fold(
        identities_by_fold,
        np.random.default_rng(17),
        resampling_mode="global_identity_clustered",
    )
    assert draw[0]["b"] == draw[1]["b"]
    assert draw[0]["c"] == draw[1]["c"]
    assert all(
        sum(draw[fold][subject] > 0 for subject in identities) >= 2
        for fold, identities in identities_by_fold.items()
    )


def test_global_identity_bootstrap_is_thread_reproducible_and_described() -> None:
    sequential = paired_multifold_bootstrap(
        _tables(),
        comparisons=(("better", "worse"),),
        replicates=20,
        seed=11,
        resampling_mode="global_identity_clustered",
    )
    threaded = paired_multifold_bootstrap(
        _tables(),
        comparisons=(("better", "worse"),),
        replicates=20,
        seed=11,
        workers=2,
        resampling_mode="global_identity_clustered",
    )
    assert sequential["resampling_mode"] == "global_identity_clustered"
    assert "shared across every fold" in sequential["identity_resampling_scope"]
    sequential.pop("workers")
    threaded.pop("workers")
    assert threaded == sequential
