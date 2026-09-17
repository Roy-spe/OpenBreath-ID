"""Paired subject-weighted multiway bootstrap for frozen multi-fold score tables."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .metrics import (
    weighted_equal_error_rate,
    weighted_minimum_detection_cost,
    weighted_rates_at_threshold,
)
from .quality import weighted_class_balanced_risk_coverage
from .uncertainty import subject_multiway_trial_weights


RESAMPLING_MODES = ("fold_stratified", "global_identity_clustered")


@dataclass(frozen=True, slots=True)
class FrozenScoreTable:
    system: str
    fold: int
    seed: int
    condition: str
    scores: np.ndarray
    labels: np.ndarray
    enrollment_subjects: Sequence[str]
    probe_subjects: Sequence[str]
    far_threshold: float
    risk_threshold: float
    quality: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class _PreparedTable:
    row: FrozenScoreTable
    scores: np.ndarray
    labels: np.ndarray
    score_order: np.ndarray
    score_group_ends: np.ndarray
    quality_order_by_label: tuple[np.ndarray, np.ndarray] | None


def _prepare_table(row: FrozenScoreTable) -> _PreparedTable:
    scores = np.asarray(row.scores, dtype=float).reshape(-1)
    labels = np.asarray(row.labels, dtype=np.int8).reshape(-1)
    score_order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[score_order]
    group_ends = np.flatnonzero(
        np.r_[sorted_scores[1:] != sorted_scores[:-1], True]
    )
    quality_order = None
    if row.quality is not None:
        quality = np.asarray(row.quality, dtype=float).reshape(-1)
        quality_order = tuple(
            indices[np.argsort(-quality[indices], kind="stable")]
            for indices in (np.flatnonzero(labels == 0), np.flatnonzero(labels == 1))
        )
    return _PreparedTable(
        row=row,
        scores=scores,
        labels=labels,
        score_order=score_order,
        score_group_ends=group_ends,
        quality_order_by_label=quality_order,
    )


def _prepared_metrics(prepared: _PreparedTable, weights: np.ndarray) -> dict[str, float]:
    """Compute the same weighted metrics while reusing frozen sort orders."""

    row = prepared.row
    scores = prepared.scores
    labels = prepared.labels
    weights = np.asarray(weights, dtype=float).reshape(-1)
    ordered_labels = labels[prepared.score_order]
    ordered_weights = weights[prepared.score_order]
    positive_total = float(weights[labels == 1].sum())
    negative_total = float(weights[labels == 0].sum())
    if positive_total <= 0.0 or negative_total <= 0.0:
        raise ValueError("both classes must have positive total weight")
    cumulative_positive = np.cumsum(ordered_weights * (ordered_labels == 1))
    cumulative_negative = np.cumsum(ordered_weights * (ordered_labels == 0))
    tar = np.r_[
        0.0,
        cumulative_positive[prepared.score_group_ends] / positive_total,
        1.0,
    ]
    far = np.r_[
        0.0,
        cumulative_negative[prepared.score_group_ends] / negative_total,
        1.0,
    ]
    frr = 1.0 - tar
    eer_index = int(np.argmin(np.abs(far - frr)))
    eer = float((far[eer_index] + frr[eer_index]) / 2.0)
    cost = 0.01 * frr + 0.99 * far
    accepted = scores >= row.far_threshold
    operating_tar = float(weights[(labels == 1) & accepted].sum() / positive_total)
    operating_far = float(weights[(labels == 0) & accepted].sum() / negative_total)
    result = {
        "eer": eer,
        "normalized_min_dcf_p01": float(cost.min() / 0.01),
        "tar_at_validation_1pct_far": operating_tar,
        "far_at_validation_1pct_far": operating_far,
    }
    if prepared.quality_order_by_label is not None:
        risks = []
        for coverage in np.linspace(0.1, 1.0, 10):
            class_error = []
            for label, ordered in enumerate(prepared.quality_order_by_label):
                ordered_weights_by_quality = weights[ordered]
                cumulative_weight = np.cumsum(ordered_weights_by_quality)
                target = coverage * float(cumulative_weight[-1])
                count = int(np.searchsorted(cumulative_weight, target, side="left")) + 1
                selected = ordered[:count]
                errors = (
                    scores[selected] >= row.risk_threshold
                    if label == 0
                    else scores[selected] < row.risk_threshold
                )
                class_error.append(float(
                    weights[selected][errors].sum() / weights[selected].sum()
                ))
            risks.append(0.5 * sum(class_error))
        result["normalized_aurc_10_to_100"] = float(
            np.trapezoid(risks, np.linspace(0.1, 1.0, 10)) / 0.9
        )
    return result


def _prepared_eer(prepared: _PreparedTable, weights: np.ndarray) -> float:
    labels = prepared.labels
    ordered_labels = labels[prepared.score_order]
    ordered_weights = np.asarray(weights, dtype=float)[prepared.score_order]
    positive_total = float(np.asarray(weights)[labels == 1].sum())
    negative_total = float(np.asarray(weights)[labels == 0].sum())
    cumulative_positive = np.cumsum(ordered_weights * (ordered_labels == 1))
    cumulative_negative = np.cumsum(ordered_weights * (ordered_labels == 0))
    tar = np.r_[0.0, cumulative_positive[prepared.score_group_ends] / positive_total, 1.0]
    far = np.r_[0.0, cumulative_negative[prepared.score_group_ends] / negative_total, 1.0]
    frr = 1.0 - tar
    index = int(np.argmin(np.abs(far - frr)))
    return float((far[index] + frr[index]) / 2.0)


def paired_primary_eer_holm(
    tables: Sequence[FrozenScoreTable],
    *,
    comparisons: Sequence[tuple[str, str]],
    replicates: int = 10_000,
    seed: int = 2027,
    workers: int = 1,
    family_alpha: float = 0.05,
) -> dict[str, object]:
    """Paired EER sign tests with Holm adjustment over all supplied hypotheses.

    The reported contrast is the observed fold-macro of seed-mean EERs, matching
    the primary results table.  The mean of the nonlinear bootstrap contrasts is
    retained only as a diagnostic; it is not the reported effect estimate.
    """

    if replicates < 1 or workers < 1:
        raise ValueError("replicates and workers must be positive")
    systems, folds, seeds, conditions = _validate_tables(tables, comparisons)
    by_key = {(row.system, row.fold, row.seed, row.condition): row for row in tables}
    prepared = {key: _prepare_table(row) for key, row in by_key.items()}
    identities_by_fold = {
        fold: sorted({
            str(subject) for row in tables if row.fold == fold
            for subject in list(row.enrollment_subjects) + list(row.probe_subjects)
        })
        for fold in folds
    }
    rng = np.random.default_rng(seed)
    draws = [
        {fold: _draw_multiplicities(identities_by_fold[fold], rng) for fold in folds}
        for _ in range(replicates)
    ]
    sample_keys = [
        (system_a, system_b, condition)
        for system_a, system_b in comparisons for condition in conditions
    ]
    samples = {key: np.empty(replicates, dtype=float) for key in sample_keys}

    observed_eers = {
        key: _prepared_eer(
            prepared[key], np.ones(len(np.asarray(row.scores).reshape(-1)), dtype=float)
        )
        for key, row in by_key.items()
    }
    observed_macro = {
        (system, condition): float(np.mean([
            np.mean([
                observed_eers[(system, fold, training_seed, condition)]
                for training_seed in seeds
            ])
            for fold in folds
        ]))
        for system in systems for condition in conditions
    }

    def compute_differences(multiplicities: Mapping[int, Mapping[str, int]]):
        weights_by_layout = {}
        for row in by_key.values():
            layout = (row.fold, row.condition)
            if layout not in weights_by_layout:
                weights_by_layout[layout] = subject_multiway_trial_weights(
                    row.enrollment_subjects, row.probe_subjects, row.labels,
                    multiplicities[row.fold],
                )
        eers = {
            key: _prepared_eer(prepared[key], weights_by_layout[(row.fold, row.condition)])
            for key, row in by_key.items()
        }
        macro = {
            (system, condition): float(np.mean([
                np.mean([eers[(system, fold, training_seed, condition)] for training_seed in seeds])
                for fold in folds
            ]))
            for system in systems for condition in conditions
        }
        return {
            key: macro[(key[0], key[2])] - macro[(key[1], key[2])]
            for key in sample_keys
        }

    if workers == 1:
        results = map(compute_differences, draws)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        results = executor.map(compute_differences, draws)
    for index, result in enumerate(results):
        for key, value in result.items():
            samples[key][index] = value
    if executor is not None:
        executor.shutdown()

    raw_p = {
        key: min(1.0, 2.0 * min(
            (float(np.count_nonzero(values <= 0.0)) + 1.0) / (replicates + 1.0),
            (float(np.count_nonzero(values >= 0.0)) + 1.0) / (replicates + 1.0),
        ))
        for key, values in samples.items()
    }
    ordered = sorted(raw_p, key=raw_p.get)
    adjusted = {}
    running = 0.0
    family_size = len(ordered)
    for rank, key in enumerate(ordered):
        running = max(running, (family_size - rank) * raw_p[key])
        adjusted[key] = min(1.0, running)
    hypotheses = {}
    for key in sample_keys:
        system_a, system_b, condition = key
        name = f"{system_a}_minus_{system_b}__{condition}"
        values = samples[key]
        hypotheses[name] = {
            "estimate": (
                observed_macro[(system_a, condition)]
                - observed_macro[(system_b, condition)]
            ),
            "bootstrap_mean_estimate_diagnostic": float(np.mean(values)),
            "bootstrap_two_sided_sign_p": raw_p[key],
            "holm_adjusted_p": adjusted[key],
            "holm_reject_at_family_alpha": adjusted[key] < family_alpha,
        }
    return {
        "method": "paired subject-weighted multiway bootstrap sign p-values with Holm-Bonferroni",
        "replicates": replicates,
        "seed": seed,
        "workers": workers,
        "family_alpha": family_alpha,
        "family_size": len(sample_keys),
        "estimate_definition": (
            "observed fold-macro of seed-mean EER contrast with unit trial weights"
        ),
        "bootstrap_sign_test_definition": (
            "two-sided paired fold-stratified identity-resampled multiway bootstrap "
            "sign p-value with plus-one correction, followed by Holm-Bonferroni"
        ),
        "hypotheses": hypotheses,
    }


def _validate_tables(
    tables: Sequence[FrozenScoreTable], comparisons: Sequence[tuple[str, str]]
) -> tuple[list[str], list[int], list[int], list[str]]:
    if not tables:
        raise ValueError("at least one frozen score table is required")
    keys = [(row.system, row.fold, row.seed, row.condition) for row in tables]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate system/fold/seed/condition score table")
    systems = sorted({row.system for row in tables})
    folds = sorted({row.fold for row in tables})
    seeds = sorted({row.seed for row in tables})
    conditions = sorted({row.condition for row in tables})
    expected = {
        (system, fold, seed, condition)
        for system in systems
        for fold in folds
        for seed in seeds
        for condition in conditions
    }
    if set(keys) != expected:
        raise ValueError("score tables do not form a complete paired design")
    for system_a, system_b in comparisons:
        if system_a not in systems or system_b not in systems or system_a == system_b:
            raise ValueError("comparisons must name two distinct available systems")

    reference: dict[tuple[int, str], FrozenScoreTable] = {}
    fold_identities: dict[int, set[str]] = {}
    for row in tables:
        scores = np.asarray(row.scores).reshape(-1)
        labels = np.asarray(row.labels).reshape(-1)
        enrollment = np.asarray(row.enrollment_subjects, dtype=object).reshape(-1)
        probe = np.asarray(row.probe_subjects, dtype=object).reshape(-1)
        if not (len(scores) == len(labels) == len(enrollment) == len(probe)) or not len(
            scores
        ):
            raise ValueError("each score table must contain aligned non-empty trials")
        if row.quality is not None and len(np.asarray(row.quality).reshape(-1)) != len(
            scores
        ):
            raise ValueError("quality must align with the score table")
        key = (row.fold, row.condition)
        prior = reference.setdefault(key, row)
        if prior is not row:
            if not (
                np.array_equal(labels, np.asarray(prior.labels).reshape(-1))
                and np.array_equal(
                    enrollment,
                    np.asarray(prior.enrollment_subjects, dtype=object).reshape(-1),
                )
                and np.array_equal(
                    probe, np.asarray(prior.probe_subjects, dtype=object).reshape(-1)
                )
            ):
                raise ValueError("paired score tables do not share identical trial rows")
        identities = set(map(str, enrollment)) | set(map(str, probe))
        prior_identities = fold_identities.setdefault(row.fold, identities)
        if prior_identities != identities:
            raise ValueError("conditions within a fold do not share the same identities")
    return systems, folds, seeds, conditions


def _draw_multiplicities(
    identities: Sequence[str], rng: np.random.Generator
) -> dict[str, int]:
    if len(identities) < 2:
        raise ValueError("multiway bootstrap requires at least two identities per fold")
    while True:
        counts = np.bincount(
            rng.integers(0, len(identities), size=len(identities)),
            minlength=len(identities),
        )
        if np.count_nonzero(counts) >= 2:
            return {
                subject: int(count)
                for subject, count in zip(identities, counts, strict=True)
            }


def _draw_global_identity_multiplicities(
    identities_by_fold: Mapping[int, Sequence[str]], rng: np.random.Generator
) -> dict[str, int]:
    """Draw one multiplicity per unique identity and keep every fold estimable."""

    for fold, identities in identities_by_fold.items():
        if len(set(identities)) < 2:
            raise ValueError(
                f"global identity bootstrap requires at least two identities in fold {fold}"
            )
    identities = sorted({subject for values in identities_by_fold.values() for subject in values})
    if len(identities) < 2:
        raise ValueError("global identity bootstrap requires at least two unique identities")
    for _ in range(100_000):
        counts = np.bincount(
            rng.integers(0, len(identities), size=len(identities)),
            minlength=len(identities),
        )
        multiplicities = {
            subject: int(count)
            for subject, count in zip(identities, counts, strict=True)
        }
        if all(
            sum(multiplicities[subject] > 0 for subject in set(fold_identities)) >= 2
            for fold_identities in identities_by_fold.values()
        ):
            return multiplicities
    raise RuntimeError(
        "global identity bootstrap could not draw at least two identities in every fold"
    )


def _draw_multiplicities_by_fold(
    identities_by_fold: Mapping[int, Sequence[str]],
    rng: np.random.Generator,
    *,
    resampling_mode: str,
) -> dict[int, Mapping[str, int]]:
    if resampling_mode == "fold_stratified":
        return {
            fold: _draw_multiplicities(identities, rng)
            for fold, identities in identities_by_fold.items()
        }
    if resampling_mode == "global_identity_clustered":
        global_multiplicities = _draw_global_identity_multiplicities(
            identities_by_fold, rng
        )
        return {fold: global_multiplicities for fold in identities_by_fold}
    raise ValueError(
        f"resampling_mode must be one of {RESAMPLING_MODES}, got {resampling_mode!r}"
    )


def _metrics(row: FrozenScoreTable, weights: np.ndarray) -> dict[str, float]:
    scores = np.asarray(row.scores, dtype=float).reshape(-1)
    labels = np.asarray(row.labels, dtype=np.int8).reshape(-1)
    tar, far = weighted_rates_at_threshold(
        scores, labels, weights, row.far_threshold
    )
    result = {
        "eer": weighted_equal_error_rate(scores, labels, weights),
        "normalized_min_dcf_p01": weighted_minimum_detection_cost(
            scores, labels, weights
        ),
        "tar_at_validation_1pct_far": tar,
        "far_at_validation_1pct_far": far,
    }
    if row.quality is not None:
        risk = weighted_class_balanced_risk_coverage(
            scores,
            labels,
            np.asarray(row.quality, dtype=float),
            weights,
            threshold=row.risk_threshold,
        )
        result["normalized_aurc_10_to_100"] = float(
            risk["normalized_aurc_10_to_100"]
        )
    return result


def _macro_summary(
    table_metrics: Mapping[tuple[str, int, int, str], Mapping[str, float]],
    *,
    systems: Iterable[str],
    folds: Iterable[int],
    seeds: Iterable[int],
    conditions: Iterable[str],
) -> dict[tuple[str, str], dict[str, float]]:
    summary: dict[tuple[str, str], dict[str, float]] = {}
    for system in systems:
        for condition in conditions:
            metric_names = table_metrics[(system, next(iter(folds)), next(iter(seeds)), condition)]
            summary[(system, condition)] = {
                metric: float(
                    np.mean(
                        [
                            np.mean(
                                [
                                    table_metrics[(system, fold, seed, condition)][metric]
                                    for seed in seeds
                                ]
                            )
                            for fold in folds
                        ]
                    )
                )
                for metric in metric_names
            }
    return summary


def paired_multifold_bootstrap(
    tables: Sequence[FrozenScoreTable],
    *,
    comparisons: Sequence[tuple[str, str]],
    replicates: int = 10_000,
    confidence_level: float = 0.95,
    seed: int = 2027,
    workers: int = 1,
    resampling_mode: str = "fold_stratified",
) -> dict[str, object]:
    """Bootstrap identities and reuse their weights for every paired score table.

    ``fold_stratified`` preserves the frozen outer-test analysis.  Use
    ``global_identity_clustered`` when the same validation participant can
    reappear in multiple folds; one multiplicity is then shared everywhere that
    identity occurs.
    """

    if replicates < 1:
        raise ValueError("replicates must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be between zero and one")
    if workers < 1:
        raise ValueError("workers must be positive")
    if resampling_mode not in RESAMPLING_MODES:
        raise ValueError(
            f"resampling_mode must be one of {RESAMPLING_MODES}, got {resampling_mode!r}"
        )
    systems, folds, seeds, conditions = _validate_tables(tables, comparisons)
    folds_tuple = tuple(folds)
    seeds_tuple = tuple(seeds)
    by_key = {
        (row.system, row.fold, row.seed, row.condition): row for row in tables
    }
    identities_by_fold = {
        fold: sorted(
            {
                str(subject)
                for row in tables
                if row.fold == fold
                for subject in (
                    list(row.enrollment_subjects) + list(row.probe_subjects)
                )
            }
        )
        for fold in folds
    }
    unit_metrics = {
        key: _metrics(row, np.ones(len(np.asarray(row.scores).reshape(-1))))
        for key, row in by_key.items()
    }
    prepared = {key: _prepare_table(row) for key, row in by_key.items()}
    point = _macro_summary(
        unit_metrics,
        systems=systems,
        folds=folds_tuple,
        seeds=seeds_tuple,
        conditions=conditions,
    )
    metric_names = {
        key: tuple(values) for key, values in point.items()
    }
    samples = {
        (system, condition, metric): np.empty(replicates, dtype=float)
        for (system, condition), names in metric_names.items()
        for metric in names
    }
    difference_samples = {
        (system_a, system_b, condition, metric): np.empty(replicates, dtype=float)
        for system_a, system_b in comparisons
        for condition in conditions
        for metric in set(metric_names[(system_a, condition)])
        & set(metric_names[(system_b, condition)])
    }
    rng = np.random.default_rng(seed)
    draws = [
        _draw_multiplicities_by_fold(
            identities_by_fold, rng, resampling_mode=resampling_mode
        )
        for _ in range(replicates)
    ]

    def compute_macro(multiplicities: Mapping[int, Mapping[str, int]]):
        weights_by_layout = {}
        for key, row in by_key.items():
            layout = (row.fold, row.condition)
            if layout not in weights_by_layout:
                weights_by_layout[layout] = subject_multiway_trial_weights(
                    row.enrollment_subjects,
                    row.probe_subjects,
                    row.labels,
                    multiplicities[row.fold],
                )
        replicate_metrics = {
            key: _prepared_metrics(prepared[key], weights_by_layout[(row.fold, row.condition)])
            for key, row in by_key.items()
        }
        return _macro_summary(
            replicate_metrics,
            systems=systems,
            folds=folds_tuple,
            seeds=seeds_tuple,
            conditions=conditions,
        )

    if workers == 1:
        macro_results = map(compute_macro, draws)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=workers)
        macro_results = executor.map(compute_macro, draws)
    for replicate, macro in enumerate(macro_results):
        for (system, condition), values in macro.items():
            for metric, value in values.items():
                samples[(system, condition, metric)][replicate] = value
        for key, target in difference_samples.items():
            system_a, system_b, condition, metric = key
            target[replicate] = (
                macro[(system_a, condition)][metric]
                - macro[(system_b, condition)][metric]
            )
    if executor is not None:
        executor.shutdown()

    tail = (1.0 - confidence_level) / 2.0
    estimates = {
        system: {
            condition: {
                metric: {
                    "estimate": point[(system, condition)][metric],
                    "lower": float(np.quantile(samples[(system, condition, metric)], tail)),
                    "upper": float(
                        np.quantile(samples[(system, condition, metric)], 1.0 - tail)
                    ),
                }
                for metric in metric_names[(system, condition)]
            }
            for condition in conditions
        }
        for system in systems
    }
    differences: dict[str, object] = {}
    for system_a, system_b in comparisons:
        name = f"{system_a}_minus_{system_b}"
        differences[name] = {}
        for condition in conditions:
            differences[name][condition] = {}
            for metric in sorted(
                set(metric_names[(system_a, condition)])
                & set(metric_names[(system_b, condition)])
            ):
                values = difference_samples[(system_a, system_b, condition, metric)]
                differences[name][condition][metric] = {
                    "estimate": point[(system_a, condition)][metric]
                    - point[(system_b, condition)][metric],
                    "lower": float(np.quantile(values, tail)),
                    "upper": float(np.quantile(values, 1.0 - tail)),
                }
    global_mode = resampling_mode == "global_identity_clustered"
    return {
        "method": (
            "paired globally identity-clustered multiway percentile bootstrap"
            if global_mode
            else "paired fold-stratified subject-weighted multiway percentile bootstrap"
        ),
        "resampling_mode": resampling_mode,
        "identity_resampling_scope": (
            "one multiplicity per unique participant shared across every fold occurrence"
            if global_mode
            else "independent identity multiplicities within each fold"
        ),
        "unique_identity_count": len({
            subject for values in identities_by_fold.values() for subject in values
        }),
        "identity_appearance_count": sum(
            len(values) for values in identities_by_fold.values()
        ),
        "identities_per_fold": {
            str(fold): len(values) for fold, values in identities_by_fold.items()
        },
        "replicates": replicates,
        "confidence_level": confidence_level,
        "seed": seed,
        "workers": workers,
        "folds": folds,
        "training_seeds": seeds,
        "systems": systems,
        "conditions": conditions,
        "degenerate_draw_rule": (
            "redraw the global identity sample until every fold retains at least two "
            "distinct identities"
            if global_mode
            else "redraw within a fold"
        ),
        "estimates": estimates,
        "paired_differences": differences,
    }
