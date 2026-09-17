"""Score frozen nested-duration manifests on validation identities only."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import csv
import gzip
import json
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy.stats import spearmanr
import torch
from torch import nn
from torch.nn import functional as F

from .data import load_signal
from .metrics import (
    equal_error_rate,
    minimum_detection_cost,
    rates_at_threshold,
    threshold_at_far,
)
from .neural_data import preprocess_window
from .neural_train import build_encoder, resolve_device
from .quality import (
    QualityEstimator,
    class_balanced_risk_coverage,
    quality_descriptors,
    quality_weighted_pool,
)
from .trial_manifest import sha256_file


@dataclass(frozen=True, slots=True)
class DurationCondition:
    name: str
    enrollment_seconds: int
    probe_seconds: int
    enrollment_subjects: tuple[str, ...]
    probes: tuple[tuple[str, int, int], ...]
    rows: int
    genuine: int
    impostor: int


def read_condition(
    path: str | Path,
    *,
    expected_condition: str,
    expected_fold: int = 0,
    expected_role: str,
) -> DurationCondition:
    """Read and validate one complete frozen trial file for an explicit role."""

    if expected_role not in {"validation", "test"}:
        raise ValueError("expected role must be validation or test")

    enrollments: dict[str, tuple[int, int]] = {}
    probes: set[tuple[str, int, int]] = set()
    pairs: set[tuple[str, str, int]] = set()
    genuine = 0
    impostor = 0
    with gzip.open(path, mode="rt", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if int(row["fold"]) != expected_fold:
                raise ValueError(f"{path}: unexpected fold")
            if row["role"] != expected_role:
                raise ValueError(f"{path}: unexpected trial role")
            if row["condition"] != expected_condition:
                raise ValueError(f"{path}: unexpected condition")
            enrollment = (
                int(row["enroll_start_sample"]),
                int(row["enroll_end_sample"]),
            )
            prior = enrollments.setdefault(row["enroll_subject"], enrollment)
            if prior != enrollment:
                raise ValueError(f"{path}: inconsistent enrollment region")
            probe = (
                row["probe_subject"],
                int(row["probe_start_sample"]),
                int(row["probe_end_sample"]),
            )
            probes.add(probe)
            pair = (row["enroll_subject"], row["probe_subject"], probe[1])
            if pair in pairs:
                raise ValueError(f"{path}: duplicate subject/probe pair")
            pairs.add(pair)
            if row["genuine_or_impostor"] == "genuine":
                genuine += 1
            elif row["genuine_or_impostor"] == "impostor":
                impostor += 1
            else:
                raise ValueError(f"{path}: invalid trial label")
    subjects = tuple(sorted(enrollments))
    ordered_probes = tuple(sorted(probes))
    expected_rows = len(subjects) * len(ordered_probes)
    if len(pairs) != expected_rows or genuine != len(ordered_probes):
        raise ValueError(f"{path}: incomplete all-pairs trial construction")
    enrollment_durations = {end - start for start, end in enrollments.values()}
    probe_durations = {end - start for _, start, end in ordered_probes}
    if len(enrollment_durations) != 1 or len(probe_durations) != 1:
        raise ValueError(f"{path}: inconsistent durations")
    sample_rate_hz = 6
    enrollment_seconds = enrollment_durations.pop() // sample_rate_hz
    probe_seconds = probe_durations.pop() // sample_rate_hz
    return DurationCondition(
        name=expected_condition,
        enrollment_seconds=enrollment_seconds,
        probe_seconds=probe_seconds,
        enrollment_subjects=subjects,
        probes=ordered_probes,
        rows=len(pairs),
        genuine=genuine,
        impostor=impostor,
    )


def read_validation_condition(
    path: str | Path,
    *,
    expected_condition: str,
    expected_fold: int = 0,
) -> DurationCondition:
    """Read a validation-role file without permitting test rows."""

    return read_condition(
        path,
        expected_condition=expected_condition,
        expected_fold=expected_fold,
        expected_role="validation",
    )


def _load_frozen_models(
    encoder_checkpoint: Path,
    encoder_report: Path,
    quality_checkpoint: Path,
    quality_report: Path,
    *,
    device: torch.device,
    require_quality_gate: bool = True,
) -> tuple[nn.Module, QualityEstimator, Mapping[str, object]]:
    encoder_result = json.loads(encoder_report.read_text(encoding="utf-8"))
    quality_result = json.loads(quality_report.read_text(encoding="utf-8"))
    if encoder_result.get("evaluations") != [] or quality_result.get(
        "outer_test_identities_scored"
    ) != 0:
        raise ValueError("duration evaluation requires validation-only source reports")
    if require_quality_gate and not quality_result.get("gate", {}).get("passed"):
        raise ValueError("the frozen quality estimator did not pass its gate")
    checkpoint = torch.load(encoder_checkpoint, map_location=device, weights_only=False)
    arguments = checkpoint["args"]
    if checkpoint.get("fold") != 0 or arguments.get("model") != "v3_bie":
        raise ValueError("expected the promoted fold-0 v3_bie checkpoint")
    encoder = build_encoder(
        "v3_bie",
        embedding_dimension=arguments["embedding_dimension"],
        branch_dimension=arguments["branch_dimension"],
        base_channels=arguments["base_channels"],
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    encoder.eval().requires_grad_(False)
    quality_state = torch.load(quality_checkpoint, map_location=device, weights_only=False)
    if quality_state.get("fold") != 0:
        raise ValueError("expected the fold-0 quality checkpoint")
    estimator = QualityEstimator(arguments["embedding_dimension"]).to(device)
    estimator.load_state_dict(quality_state["quality_estimator"])
    estimator.eval().requires_grad_(False)
    return encoder, estimator, arguments


@torch.inference_mode()
def build_segment_cache(
    dataset_root: Path,
    source_registry: Mapping[str, object],
    required_starts: Mapping[str, set[int]],
    encoder: nn.Module,
    estimator: QualityEstimator | None,
    *,
    device: torch.device,
    batch_size: int,
    normalization: str,
    canonicalize_polarity: bool,
    channel_view: str = "bilateral",
) -> dict[str, dict[int, tuple[torch.Tensor, float]]]:
    sources = {
        str(row["subject_id"]): row for row in source_registry["sources"]
    }
    cache: dict[str, dict[int, tuple[torch.Tensor, float]]] = {}
    for subject in sorted(required_starts):
        source = sources.get(subject)
        if source is None:
            raise ValueError(f"source registry is missing {subject}")
        path = dataset_root / str(source["relative_path"])
        if sha256_file(path) != str(source["sha256"]):
            raise ValueError(f"source hash mismatch for {subject}")
        signal = load_signal(path)
        starts = sorted(required_starts[subject])
        windows = torch.from_numpy(
            np.stack(
                [
                    preprocess_window(
                        signal[start : start + 180],
                        normalization=normalization,
                        canonicalize_polarity=canonicalize_polarity,
                        channel_view=channel_view,
                    )
                    for start in starts
                ]
            )
        )
        embedded_parts: list[torch.Tensor] = []
        quality_parts: list[torch.Tensor] = []
        for offset in range(0, len(windows), batch_size):
            batch = windows[offset : offset + batch_size]
            embeddings = encoder(batch.to(device)).float()
            embedded_parts.append(embeddings.cpu())
            if estimator is None:
                quality_parts.append(torch.ones(len(embeddings)))
            else:
                descriptors = quality_descriptors(batch).to(device)
                quality_parts.append(estimator(embeddings, descriptors).cpu())
        embeddings = torch.cat(embedded_parts)
        qualities = torch.cat(quality_parts)
        cache[subject] = {
            start: (embeddings[index], float(qualities[index]))
            for index, start in enumerate(starts)
        }
    return cache


def _pool(
    rows: list[tuple[torch.Tensor, float]], *, weighted: bool
) -> tuple[torch.Tensor, float]:
    embeddings = torch.stack([row[0] for row in rows])
    quality = torch.tensor([row[1] for row in rows])
    pooled = (
        quality_weighted_pool(embeddings, quality)
        if weighted
        else F.normalize(embeddings.mean(dim=0), dim=0)
    )
    return pooled, float(quality.mean())


def score_condition(
    condition: DurationCondition,
    cache: Mapping[str, Mapping[int, tuple[torch.Tensor, float]]],
    *,
    weighted: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    enrollment_segments = condition.enrollment_seconds // 30
    probe_segments = condition.probe_seconds // 30
    enrollments: list[torch.Tensor] = []
    enrollment_quality: list[float] = []
    for subject in condition.enrollment_subjects:
        rows = [cache[subject][index * 180] for index in range(enrollment_segments)]
        pooled, quality = _pool(rows, weighted=weighted)
        enrollments.append(pooled)
        enrollment_quality.append(quality)
    probes: list[torch.Tensor] = []
    probe_quality: list[float] = []
    probe_subjects: list[str] = []
    for subject, start, _ in condition.probes:
        rows = [cache[subject][start + index * 180] for index in range(probe_segments)]
        pooled, quality = _pool(rows, weighted=weighted)
        probes.append(pooled)
        probe_quality.append(quality)
        probe_subjects.append(subject)
    score_matrix = torch.stack(probes) @ torch.stack(enrollments).T
    label_matrix = np.asarray(probe_subjects)[:, None] == np.asarray(
        condition.enrollment_subjects
    )[None, :]
    trial_quality = np.sqrt(
        np.asarray(probe_quality)[:, None] * np.asarray(enrollment_quality)[None, :]
    )
    scores = score_matrix.numpy().reshape(-1)
    labels = label_matrix.astype(np.int8).reshape(-1)
    if len(scores) != condition.rows or int(labels.sum()) != condition.genuine:
        raise ValueError(f"{condition.name}: scored trials do not match the manifest")
    return scores, labels, trial_quality.reshape(-1)


def summarize_scores(
    scores: np.ndarray, labels: np.ndarray, trial_quality: np.ndarray
) -> dict[str, object]:
    threshold = threshold_at_far(scores, labels, target_far=0.01)
    tar, far = rates_at_threshold(scores, labels, threshold)
    return {
        "eer": equal_error_rate(scores, labels),
        "normalized_min_dcf_p01": minimum_detection_cost(scores, labels),
        "empirical_1pct_far_threshold": threshold,
        "tar_at_empirical_1pct_far": tar,
        "far_at_empirical_1pct_far": far,
        "risk_coverage": class_balanced_risk_coverage(
            scores, labels, trial_quality
        ),
    }


def duration_trends(results: Mapping[str, object]) -> dict[str, object]:
    conditions = results["conditions"]
    definitions = {
        "fixed_enrollment_300": lambda row: row["enrollment_seconds"] == 300,
        "fixed_probe_60": lambda row: row["probe_seconds"] == 60,
        "matched_duration": lambda row: row["enrollment_seconds"]
        == row["probe_seconds"],
    }
    trends: dict[str, object] = {}
    for analysis, predicate in definitions.items():
        selected = sorted(
            (row for row in conditions if predicate(row)),
            key=lambda row: (
                row["probe_seconds"]
                if analysis == "fixed_enrollment_300"
                else row["enrollment_seconds"]
            ),
        )
        analysis_result: dict[str, object] = {
            "conditions": [row["condition"] for row in selected]
        }
        durations = np.asarray(
            [
                row["probe_seconds"]
                if analysis == "fixed_enrollment_300"
                else row["enrollment_seconds"]
                for row in selected
            ]
        )
        for method in ("mean", "quality_weighted"):
            eers = np.asarray([row[method]["eer"] for row in selected])
            rho = float(spearmanr(durations, eers).statistic)
            analysis_result[method] = {
                "durations_seconds_ascending": durations.tolist(),
                "eer": eers.tolist(),
                "spearman_duration_vs_eer": rho,
                "longest_minus_shortest_eer": float(eers[-1] - eers[0]),
            }
        trends[analysis] = analysis_result
    return trends


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--scoring-specification",
        type=Path,
        default=Path("configs/proposal_v3_1_duration_validation.json"),
    )
    parser.add_argument(
        "--experiment-specification",
        type=Path,
        default=Path("configs/proposal_v3_1_duration_nostril.json"),
    )
    parser.add_argument(
        "--manifest-dir", type=Path, default=Path("metadata/trials_v3_1_duration")
    )
    parser.add_argument("--encoder-checkpoint", type=Path, default=Path("reports/v3_bie_dev_fold0.pt"))
    parser.add_argument("--encoder-report", type=Path, default=Path("reports/v3_bie_dev_fold0.json"))
    parser.add_argument("--quality-checkpoint", type=Path, default=Path("reports/v3_quality_dev_fold0.pt"))
    parser.add_argument("--quality-report", type=Path, default=Path("reports/v3_quality_dev_fold0.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/v3_duration_validation_fold0.json"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--inference-batch-size", type=int, default=512)
    return parser.parse_args(argv)


def _verify_scoring_provenance(args: argparse.Namespace) -> Mapping[str, object]:
    specification = json.loads(args.scoring_specification.read_text(encoding="utf-8"))
    checks = (
        (args.experiment_specification, specification["parent_experiment_protocol"]["sha256"]),
        (args.manifest_dir / "trial_manifest.json", specification["trial_manifest"]["sha256"]),
        (args.encoder_checkpoint, specification["models"]["encoder"]["sha256"]),
        (args.quality_checkpoint, specification["models"]["quality"]["sha256"]),
    )
    for path, expected in checks:
        observed = sha256_file(path)
        if observed != expected:
            raise ValueError(f"provenance hash mismatch for {path}: {observed}")
    return specification


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    scoring_specification = _verify_scoring_provenance(args)
    manifest = json.loads(
        (args.manifest_dir / "trial_manifest.json").read_text(encoding="utf-8")
    )
    experiment = json.loads(args.experiment_specification.read_text(encoding="utf-8"))
    condition_spec = {row["name"]: row for row in experiment["conditions"]}
    selected_outputs = [
        row
        for row in manifest["outputs"]
        if row["fold"] == 0 and row["role"] == "validation"
    ]
    if len(selected_outputs) != 15:
        raise ValueError("expected exactly 15 fold-0 validation conditions")
    conditions: list[DurationCondition] = []
    for output in selected_outputs:
        path = args.manifest_dir / output["path"]
        if sha256_file(path) != output["sha256"]:
            raise ValueError(f"trial file hash mismatch: {path}")
        condition = read_validation_condition(
            path, expected_condition=output["condition"], expected_fold=0
        )
        expected = condition_spec[condition.name]
        if (
            condition.enrollment_seconds != expected["enrollment_seconds"]
            or condition.probe_seconds != expected["probe_seconds"]
            or condition.rows != output["rows"]
        ):
            raise ValueError(f"manifest/specification mismatch for {condition.name}")
        conditions.append(condition)

    required_starts: dict[str, set[int]] = {}
    for condition in conditions:
        for subject in condition.enrollment_subjects:
            required_starts.setdefault(subject, set()).update(
                index * 180 for index in range(condition.enrollment_seconds // 30)
            )
        for subject, start, _ in condition.probes:
            required_starts.setdefault(subject, set()).update(
                start + index * 180 for index in range(condition.probe_seconds // 30)
            )

    source_registry_path = args.manifest_dir / manifest["source_registry"]
    if sha256_file(source_registry_path) != manifest["source_registry_sha256"]:
        raise ValueError("source registry hash mismatch")
    source_registry = json.loads(source_registry_path.read_text(encoding="utf-8"))
    device = resolve_device(args.device)
    encoder, estimator, encoder_args = _load_frozen_models(
        args.encoder_checkpoint,
        args.encoder_report,
        args.quality_checkpoint,
        args.quality_report,
        device=device,
    )
    print(
        f"device={device} validation_identities={len(required_starts)} "
        f"segments={sum(len(value) for value in required_starts.values())}"
    )
    cache = build_segment_cache(
        args.dataset_root.resolve(),
        source_registry,
        required_starts,
        encoder,
        estimator,
        device=device,
        batch_size=args.inference_batch_size,
        normalization=encoder_args["normalization"],
        canonicalize_polarity=encoder_args["channel_polarity"] == "correlation",
    )
    condition_results: list[dict[str, object]] = []
    for condition in conditions:
        row: dict[str, object] = {
            "condition": condition.name,
            "enrollment_seconds": condition.enrollment_seconds,
            "probe_seconds": condition.probe_seconds,
            "trials": condition.rows,
            "genuine": condition.genuine,
            "impostor": condition.impostor,
        }
        for method, weighted in (("mean", False), ("quality_weighted", True)):
            scores, labels, trial_quality = score_condition(
                condition, cache, weighted=weighted
            )
            row[method] = summarize_scores(scores, labels, trial_quality)
        condition_results.append(row)
        print(
            f"{condition.name} mean_eer={100 * row['mean']['eer']:.2f}% "
            f"quality_eer={100 * row['quality_weighted']['eer']:.2f}%"
        )

    payload: dict[str, object] = {
        "protocol": "V3.1 nested-duration fold-0 validation-only scoring",
        "fold": 0,
        "role": "validation",
        "validation_identities": len(required_starts),
        "outer_test_identities_scored": 0,
        "test_trial_files_opened": 0,
        "duration_specific_retraining": False,
        "encoder_frozen": True,
        "quality_estimator_frozen": True,
        "conditions": condition_results,
    }
    payload["trend_summaries"] = duration_trends(payload)
    payload["provenance"] = {
        "scoring_specification": str(args.scoring_specification.resolve()),
        "scoring_specification_sha256": sha256_file(args.scoring_specification),
        "experiment_specification_sha256": sha256_file(args.experiment_specification),
        "trial_manifest_sha256": sha256_file(args.manifest_dir / "trial_manifest.json"),
        "encoder_checkpoint_sha256": sha256_file(args.encoder_checkpoint),
        "quality_checkpoint_sha256": sha256_file(args.quality_checkpoint),
    }
    payload["interpretation"] = scoring_specification["interpretation"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
