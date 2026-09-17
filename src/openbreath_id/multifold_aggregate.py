"""Aggregate frozen V3.1 multi-fold test scores and paired uncertainty."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np

from .multifold_bootstrap import FrozenScoreTable, paired_multifold_bootstrap
from .multifold_test import SYSTEMS
from .trial_manifest import sha256_file


METRICS = (
    "eer",
    "normalized_min_dcf_p01",
    "tar_at_validation_1pct_far",
    "far_at_validation_1pct_far",
    "normalized_aurc_10_to_100",
)


def _metric(row: Mapping[str, object], name: str) -> float:
    if name == "normalized_aurc_10_to_100":
        return float(row["risk_coverage"][name])
    return float(row[name])


def aggregate_point_results(reports: list[Mapping[str, object]]) -> dict[str, object]:
    if len(reports) != 15:
        raise ValueError("expected exactly 15 frozen test reports")
    keys = {(int(row["fold"]), int(row["training_seed"])) for row in reports}
    if keys != {(fold, seed) for fold in range(5) for seed in (2027, 2028, 2029)}:
        raise ValueError("test reports do not form the frozen five-fold/three-seed design")
    conditions = [str(row["condition"]) for row in reports[0]["conditions"]]
    result: dict[str, object] = {}
    for condition in conditions:
        result[condition] = {}
        for system in SYSTEMS:
            folds = []
            for fold in range(5):
                seed_rows = []
                for report in sorted(
                    (row for row in reports if int(row["fold"]) == fold),
                    key=lambda row: int(row["training_seed"]),
                ):
                    condition_row = next(
                        row for row in report["conditions"]
                        if row["condition"] == condition
                    )
                    seed_rows.append({
                        "seed": int(report["training_seed"]),
                        **{
                            metric: _metric(condition_row["systems"][system], metric)
                            for metric in METRICS
                        },
                    })
                folds.append({
                    "fold": fold,
                    "seeds": seed_rows,
                    "mean": {
                        metric: float(np.mean([row[metric] for row in seed_rows]))
                        for metric in METRICS
                    },
                    "standard_deviation": {
                        metric: float(np.std([row[metric] for row in seed_rows], ddof=1))
                        for metric in METRICS
                    },
                })
            result[condition][system] = {
                "folds": folds,
                "macro_mean": {
                    metric: float(np.mean([row["mean"][metric] for row in folds]))
                    for metric in METRICS
                },
                "macro_fold_standard_deviation": {
                    metric: float(np.std([row["mean"][metric] for row in folds], ddof=1))
                    for metric in METRICS
                },
            }
        comparisons = {}
        for system_a, system_b in ((SYSTEMS[1], SYSTEMS[0]), (SYSTEMS[2], SYSTEMS[1])):
            comparisons[f"{system_a}_minus_{system_b}"] = {
                metric: (
                    result[condition][system_a]["macro_mean"][metric]
                    - result[condition][system_b]["macro_mean"][metric]
                )
                for metric in METRICS
            }
        result[condition]["paired_point_differences"] = comparisons
    return result


def load_primary_tables(
    reports: list[Mapping[str, object]],
    *,
    scoring_dir: Path,
    primary_conditions: list[str],
) -> list[FrozenScoreTable]:
    tables = []
    for report in reports:
        archive_path = scoring_dir / Path(str(report["score_archive"])).name
        if sha256_file(archive_path) != report["score_archive_sha256"]:
            raise ValueError(f"score archive hash mismatch: {archive_path}")
        with np.load(archive_path, allow_pickle=False) as archive:
            for condition in primary_conditions:
                row = next(
                    item for item in report["conditions"]
                    if item["condition"] == condition
                )
                labels = archive[f"{condition}__labels"].copy()
                quality = archive[f"{condition}__quality"].copy()
                enrollment = archive[f"{condition}__enrollment_subjects"].copy()
                probe = archive[f"{condition}__probe_subjects"].copy()
                for system in SYSTEMS:
                    threshold = float(
                        row["systems"][system]["validation_1pct_far_threshold"]
                    )
                    tables.append(FrozenScoreTable(
                        system=system,
                        fold=int(report["fold"]),
                        seed=int(report["training_seed"]),
                        condition=condition,
                        scores=archive[f"{condition}__{system}__scores"].copy(),
                        labels=labels,
                        enrollment_subjects=enrollment,
                        probe_subjects=probe,
                        far_threshold=threshold,
                        risk_threshold=threshold,
                        quality=quality,
                    ))
    return tables


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--specification", type=Path,
        default=Path("configs/proposal_v3_1_multifold_duration_evaluation.json"),
    )
    parser.add_argument(
        "--scoring-dir", type=Path, default=Path("reports/v3_multifold_scoring")
    )
    parser.add_argument(
        "--authorization", type=Path,
        default=Path("reports/v3_multifold_test_authorization.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/v3_multifold_test_summary.json")
    )
    parser.add_argument("--bootstrap-replicates", type=int)
    parser.add_argument("--bootstrap-workers", type=int, default=8)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    specification = json.loads(args.specification.read_text(encoding="utf-8"))
    authorization = json.loads(args.authorization.read_text(encoding="utf-8"))
    if not authorization.get("authorized"):
        raise ValueError("frozen test scoring was not authorized")
    report_paths = sorted(args.scoring_dir.glob("test_fold*_seed*.json"))
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in report_paths]
    point = aggregate_point_results(reports)
    primary = list(map(str, specification["evaluation"]["primary_conditions"]))
    tables = load_primary_tables(
        reports, scoring_dir=args.scoring_dir, primary_conditions=primary
    )
    uncertainty = specification["uncertainty"]
    replicates = (
        int(uncertainty["replicates"])
        if args.bootstrap_replicates is None
        else args.bootstrap_replicates
    )
    print(
        f"paired bootstrap tables={len(tables)} replicates={replicates}",
        flush=True,
    )
    bootstrap = paired_multifold_bootstrap(
        tables,
        comparisons=((SYSTEMS[1], SYSTEMS[0]), (SYSTEMS[2], SYSTEMS[1])),
        replicates=replicates,
        confidence_level=float(uncertainty["confidence_level"]),
        seed=int(uncertainty["bootstrap_seed"]),
        workers=args.bootstrap_workers,
    )
    payload = {
        "protocol": "V3.1 frozen five-fold/three-seed test aggregate",
        "status": "completed frozen development-identity cross-validation; not external confirmation",
        "test_scoring_passes": 1,
        "test_reports": len(reports),
        "system_condition_evaluations": len(reports) * 15 * len(SYSTEMS),
        "point_estimates": point,
        "primary_uncertainty": bootstrap,
        "provenance": {
            "specification_sha256": sha256_file(args.specification),
            "authorization_sha256": sha256_file(args.authorization),
            "test_report_sha256": {
                path.name: sha256_file(path) for path in report_paths
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "test_reports": payload["test_reports"],
        "system_condition_evaluations": payload["system_condition_evaluations"],
        "bootstrap_replicates": replicates,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
