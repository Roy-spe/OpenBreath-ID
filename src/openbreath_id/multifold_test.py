"""Fail-closed scoring for the frozen V3.1 multi-fold duration test protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

from .duration_eval import (
    DurationCondition,
    build_segment_cache,
    read_condition,
    score_condition,
)
from .metrics import (
    equal_error_rate,
    minimum_detection_cost,
    rates_at_threshold,
    threshold_at_far,
)
from .neural_train import build_encoder, resolve_device
from .quality import QualityEstimator, class_balanced_risk_coverage
from .trial_manifest import sha256_file


SYSTEMS = (
    "parameter_matched_stacked_mean",
    "bie_mean",
    "bie_quality_weighted",
)


def _json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _artifact_index(state: Mapping[str, object]) -> dict[tuple[str, str, int, int], dict[str, object]]:
    return {
        (str(row["key"][0]), str(row["key"][1]), int(row["key"][2]), int(row["key"][3])): row
        for row in state["completed"]
    }


def verify_completed_gate(
    specification_path: Path,
    audit_path: Path,
    state_path: Path,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    """Verify every frozen artifact before permitting any test-role read."""

    specification = _json(specification_path)
    audit = _json(audit_path)
    state = _json(state_path)
    if not audit.get("passed") or audit.get("test_trial_files_opened") != 0:
        raise ValueError("the metadata-only execution audit did not pass")
    if audit["provenance"]["specification_sha256"] != sha256_file(specification_path):
        raise ValueError("execution audit is stale for the scoring specification")
    if state.get("completed_jobs") != 45 or state.get("pending_jobs") != 0:
        raise ValueError("all 45 validation-only artifacts must be complete")
    if state.get("outer_test_identities_scored") != 0 or state.get("test_trial_files_opened") != 0:
        raise ValueError("validation state does not prove zero test access")
    if state["provenance"]["specification_sha256"] != sha256_file(specification_path):
        raise ValueError("validation state is stale for the scoring specification")
    if state["provenance"]["execution_audit_sha256"] != sha256_file(audit_path):
        raise ValueError("validation state is stale for the execution audit")
    expected_keys = {
        (
            str(job["stage"]),
            str(job["model"]),
            int(job["fold"]),
            int(job["seed"]),
        )
        for job in audit["job_plan"]["jobs"]
    }
    artifacts = _artifact_index(state)
    if set(artifacts) != expected_keys:
        raise ValueError("validation artifact design does not match the audited job plan")
    for row in artifacts.values():
        report = Path(str(row["report"]))
        checkpoint = Path(str(row["checkpoint"]))
        if sha256_file(report) != row["report_sha256"]:
            raise ValueError(f"validation report hash mismatch: {report}")
        if sha256_file(checkpoint) != row["checkpoint_sha256"]:
            raise ValueError(f"validation checkpoint hash mismatch: {checkpoint}")
    return specification, audit, state


def build_authorization(
    specification_path: Path,
    audit_path: Path,
    state_path: Path,
) -> dict[str, object]:
    specification, audit, state = verify_completed_gate(
        specification_path, audit_path, state_path
    )
    manifest_path = Path(str(specification["trial_manifest"]["path"]))
    if sha256_file(manifest_path) != specification["trial_manifest"]["sha256"]:
        raise ValueError("trial manifest provenance mismatch")
    manifest = _json(manifest_path)
    test_outputs = [row for row in manifest["outputs"] if row["role"] == "test"]
    if len(test_outputs) != 75:
        raise ValueError("expected 75 frozen test-role files")
    for row in test_outputs:
        path = manifest_path.parent / str(row["path"])
        if not path.is_file() or path.stat().st_size <= 0:
            raise ValueError(f"missing or empty test-role file: {path}")
    return {
        "protocol": "V3.1 frozen multi-fold test-scoring authorization",
        "authorized": True,
        "validation_jobs_verified": int(state["completed_jobs"]),
        "pending_validation_jobs": int(state["pending_jobs"]),
        "test_trial_files_metadata_checked": len(test_outputs),
        "test_trial_files_opened": 0,
        "test_trial_bytes_read": 0,
        "systems": [row["name"] for row in specification["runs"]["systems"]],
        "folds": specification["runs"]["folds"],
        "training_seeds": specification["runs"]["training_seeds"],
        "conditions": manifest["conditions"],
        "provenance": {
            "specification_sha256": sha256_file(specification_path),
            "execution_audit_sha256": sha256_file(audit_path),
            "validation_state_sha256": sha256_file(state_path),
            "trial_manifest_sha256": sha256_file(manifest_path),
            "scorer_sha256": sha256_file(Path(__file__)),
        },
        "decision": "authorize_one_frozen_test_scoring_pass",
    }


def verify_authorization(
    authorization_path: Path,
    specification_path: Path,
    audit_path: Path,
    state_path: Path,
) -> dict[str, object]:
    authorization = _json(authorization_path)
    if not authorization.get("authorized") or authorization.get("test_trial_files_opened") != 0:
        raise ValueError("test-scoring authorization did not pass preflight")
    expected = {
        "specification_sha256": sha256_file(specification_path),
        "execution_audit_sha256": sha256_file(audit_path),
        "validation_state_sha256": sha256_file(state_path),
        "scorer_sha256": sha256_file(Path(__file__)),
    }
    mismatches = {
        key: {"expected": value, "observed": authorization["provenance"].get(key)}
        for key, value in expected.items()
        if authorization["provenance"].get(key) != value
    }
    if mismatches:
        raise ValueError(f"test-scoring authorization is stale: {mismatches}")
    return authorization


def _load_conditions(
    manifest: Mapping[str, object],
    manifest_dir: Path,
    *,
    fold: int,
    role: str,
) -> list[DurationCondition]:
    selected = [
        row
        for row in manifest["outputs"]
        if int(row["fold"]) == fold and row["role"] == role
    ]
    if len(selected) != 15:
        raise ValueError(f"expected 15 {role} conditions for fold {fold}")
    conditions = []
    for row in selected:
        path = manifest_dir / str(row["path"])
        if sha256_file(path) != row["sha256"]:
            raise ValueError(f"trial hash mismatch: {path}")
        condition = read_condition(
            path,
            expected_condition=str(row["condition"]),
            expected_fold=fold,
            expected_role=role,
        )
        if condition.rows != int(row["rows"]):
            raise ValueError(f"trial count mismatch: {path}")
        conditions.append(condition)
    return conditions


def _required_starts(conditions: list[DurationCondition]) -> dict[str, set[int]]:
    starts: dict[str, set[int]] = {}
    for condition in conditions:
        for subject in condition.enrollment_subjects:
            starts.setdefault(subject, set()).update(
                index * 180 for index in range(condition.enrollment_seconds // 30)
            )
        for subject, start, _ in condition.probes:
            starts.setdefault(subject, set()).update(
                start + index * 180 for index in range(condition.probe_seconds // 30)
            )
    return starts


def _load_encoder(
    checkpoint_path: Path,
    report_path: Path,
    *,
    model: str,
    fold: int,
    seed: int,
    device: torch.device,
) -> tuple[torch.nn.Module, Mapping[str, object]]:
    report = _json(report_path)
    if (
        report.get("fold") != fold
        or report.get("training_seed") != seed
        or report.get("model") != model
        or report.get("evaluations") != []
    ):
        raise ValueError("encoder report is not the matching frozen validation artifact")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    arguments = checkpoint["args"]
    if checkpoint.get("fold") != fold or arguments.get("model") != model:
        raise ValueError("encoder checkpoint fold/model mismatch")
    encoder = build_encoder(
        model,
        embedding_dimension=arguments["embedding_dimension"],
        branch_dimension=arguments["branch_dimension"],
        base_channels=arguments["base_channels"],
    ).to(device)
    encoder.load_state_dict(checkpoint["encoder"])
    encoder.eval().requires_grad_(False)
    return encoder, arguments


def _load_quality(
    checkpoint_path: Path,
    report_path: Path,
    *,
    fold: int,
    seed: int,
    embedding_dimension: int,
    device: torch.device,
) -> QualityEstimator:
    report = _json(report_path)
    if (
        report.get("fold") != fold
        or report.get("training_seed") != seed
        or report.get("outer_test_identities_scored") != 0
    ):
        raise ValueError("quality report is not the matching frozen validation artifact")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("fold") != fold or checkpoint.get("args", {}).get("seed") != seed:
        raise ValueError("quality checkpoint fold/seed mismatch")
    estimator = QualityEstimator(embedding_dimension).to(device)
    estimator.load_state_dict(checkpoint["quality_estimator"])
    estimator.eval().requires_grad_(False)
    return estimator


def attach_reference_quality(
    cache: Mapping[str, Mapping[int, tuple[torch.Tensor, float]]],
    quality_cache: Mapping[str, Mapping[int, tuple[torch.Tensor, float]]],
) -> dict[str, dict[int, tuple[torch.Tensor, float]]]:
    """Attach the fold/seed BIE quality estimate to comparator embeddings."""

    if set(cache) != set(quality_cache):
        raise ValueError("embedding and quality caches have different subjects")
    result = {}
    for subject, rows in cache.items():
        if set(rows) != set(quality_cache[subject]):
            raise ValueError("embedding and quality caches have different segments")
        result[subject] = {
            start: (embedding, quality_cache[subject][start][1])
            for start, (embedding, _) in rows.items()
        }
    return result


def trial_subject_vectors(condition: DurationCondition) -> tuple[np.ndarray, np.ndarray]:
    enrollment = np.asarray(condition.enrollment_subjects, dtype=str)
    probes = np.asarray([row[0] for row in condition.probes], dtype=str)
    return np.tile(enrollment, len(probes)), np.repeat(probes, len(enrollment))


def summarize_role_scores(
    scores: np.ndarray,
    labels: np.ndarray,
    quality: np.ndarray,
    *,
    validation_threshold: float | None = None,
) -> dict[str, object]:
    threshold = (
        threshold_at_far(scores, labels, target_far=0.01)
        if validation_threshold is None
        else float(validation_threshold)
    )
    tar, far = rates_at_threshold(scores, labels, threshold)
    return {
        "eer": equal_error_rate(scores, labels),
        "normalized_min_dcf_p01": minimum_detection_cost(scores, labels),
        "validation_1pct_far_threshold": threshold,
        "tar_at_validation_1pct_far": tar,
        "far_at_validation_1pct_far": far,
        "risk_coverage": class_balanced_risk_coverage(
            scores, labels, quality, threshold=threshold
        ),
    }


def _bundle_paths(output_dir: Path, role: str, fold: int, seed: int) -> tuple[Path, Path]:
    stem = output_dir / f"{role}_fold{fold}_seed{seed}"
    return stem.with_suffix(".json"), stem.with_suffix(".npz")


def _verified_existing(report_path: Path, scores_path: Path, role: str, fold: int, seed: int) -> bool:
    if not report_path.is_file() or not scores_path.is_file():
        return False
    report = _json(report_path)
    if (
        report.get("role") != role
        or report.get("fold") != fold
        or report.get("training_seed") != seed
        or report.get("score_archive_sha256") != sha256_file(scores_path)
    ):
        raise ValueError(f"existing scoring bundle failed verification: {report_path}")
    return True


def _score_bundle(
    *,
    role: str,
    fold: int,
    seed: int,
    dataset_root: Path,
    specification: Mapping[str, object],
    state: Mapping[str, object],
    manifest: Mapping[str, object],
    manifest_dir: Path,
    output_dir: Path,
    authorization_path: Path,
    device: torch.device,
    inference_batch_size: int,
) -> None:
    report_path, scores_path = _bundle_paths(output_dir, role, fold, seed)
    if _verified_existing(report_path, scores_path, role, fold, seed):
        print(f"{role} fold={fold} seed={seed} already verified", flush=True)
        return
    if report_path.exists() or scores_path.exists():
        raise ValueError(f"partial scoring bundle exists and will not be overwritten: {report_path}")
    index = _artifact_index(state)
    stacked_row = index[("encoder_validation_only", "v3_stacked", fold, seed)]
    bie_row = index[("encoder_validation_only", "v3_bie", fold, seed)]
    quality_row = index[("quality_validation_only", "v3_bie_quality", fold, seed)]
    conditions = _load_conditions(manifest, manifest_dir, fold=fold, role=role)
    starts = _required_starts(conditions)
    source_registry_path = manifest_dir / str(manifest["source_registry"])
    if sha256_file(source_registry_path) != manifest["source_registry_sha256"]:
        raise ValueError("source registry hash mismatch")
    source_registry = _json(source_registry_path)
    stacked, stacked_args = _load_encoder(
        Path(str(stacked_row["checkpoint"])), Path(str(stacked_row["report"])),
        model="v3_stacked", fold=fold, seed=seed, device=device,
    )
    bie, bie_args = _load_encoder(
        Path(str(bie_row["checkpoint"])), Path(str(bie_row["report"])),
        model="v3_bie", fold=fold, seed=seed, device=device,
    )
    estimator = _load_quality(
        Path(str(quality_row["checkpoint"])), Path(str(quality_row["report"])),
        fold=fold, seed=seed, embedding_dimension=int(bie_args["embedding_dimension"]), device=device,
    )
    print(
        f"{role} fold={fold} seed={seed} identities={len(starts)} "
        f"segments={sum(map(len, starts.values()))} device={device}",
        flush=True,
    )
    bie_cache = build_segment_cache(
        dataset_root, source_registry, starts, bie, estimator,
        device=device, batch_size=inference_batch_size,
        normalization=str(bie_args["normalization"]),
        canonicalize_polarity=bie_args["channel_polarity"] == "correlation",
    )
    stacked_cache = build_segment_cache(
        dataset_root, source_registry, starts, stacked, None,
        device=device, batch_size=inference_batch_size,
        normalization=str(stacked_args["normalization"]),
        canonicalize_polarity=stacked_args["channel_polarity"] == "correlation",
    )
    stacked_cache = attach_reference_quality(stacked_cache, bie_cache)
    validation_report = None
    if role == "test":
        validation_path, validation_scores = _bundle_paths(output_dir, "validation", fold, seed)
        if not _verified_existing(validation_path, validation_scores, "validation", fold, seed):
            raise ValueError("every validation scoring bundle must exist before test scoring")
        validation_report = _json(validation_path)
    arrays: dict[str, np.ndarray] = {}
    condition_results = []
    for condition in conditions:
        enroll_subjects, probe_subjects = trial_subject_vectors(condition)
        row: dict[str, object] = {
            "condition": condition.name,
            "enrollment_seconds": condition.enrollment_seconds,
            "probe_seconds": condition.probe_seconds,
            "trials": condition.rows,
            "genuine": condition.genuine,
            "impostor": condition.impostor,
            "systems": {},
        }
        scored = (
            (SYSTEMS[0], stacked_cache, False),
            (SYSTEMS[1], bie_cache, False),
            (SYSTEMS[2], bie_cache, True),
        )
        shared_labels = None
        shared_quality = None
        for system, cache, weighted in scored:
            scores, labels, trial_quality = score_condition(
                condition, cache, weighted=weighted
            )
            if shared_labels is None:
                shared_labels, shared_quality = labels, trial_quality
            elif not np.array_equal(labels, shared_labels) or not np.allclose(
                trial_quality, shared_quality
            ):
                raise ValueError("paired systems did not produce aligned trials")
            threshold = None
            if validation_report is not None:
                validation_condition = next(
                    item for item in validation_report["conditions"]
                    if item["condition"] == condition.name
                )
                threshold = validation_condition["systems"][system][
                    "validation_1pct_far_threshold"
                ]
            row["systems"][system] = summarize_role_scores(
                scores, labels, trial_quality, validation_threshold=threshold
            )
            arrays[f"{condition.name}__{system}__scores"] = scores.astype(np.float32)
        arrays[f"{condition.name}__labels"] = shared_labels.astype(np.int8)
        arrays[f"{condition.name}__quality"] = shared_quality.astype(np.float32)
        arrays[f"{condition.name}__enrollment_subjects"] = enroll_subjects
        arrays[f"{condition.name}__probe_subjects"] = probe_subjects
        condition_results.append(row)
        print(
            f"  {condition.name} "
            + " ".join(
                f"{system}={100 * row['systems'][system]['eer']:.2f}%"
                for system in SYSTEMS
            ),
            flush=True,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_scores = scores_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary_scores, **arrays)
    temporary_scores.replace(scores_path)
    payload = {
        "protocol": "V3.1 frozen multi-fold duration scoring",
        "role": role,
        "fold": fold,
        "training_seed": seed,
        "systems": list(SYSTEMS),
        "conditions": condition_results,
        "duration_specific_retraining": False,
        "encoder_checkpoints_frozen": True,
        "quality_estimator_frozen": True,
        "quality_reference": (
            "The fold/seed BIE detached quality head ranks the identical signal trials "
            "for every system; it is not used to pool stacked embeddings."
        ),
        "validation_thresholds_applied_unchanged": role == "test",
        "score_archive": str(scores_path),
        "score_archive_sha256": sha256_file(scores_path),
        "authorization_sha256": sha256_file(authorization_path),
        "provenance": {
            "specification_sha256": sha256_file(
                Path("configs/proposal_v3_1_multifold_duration_evaluation.json")
            ),
            "stacked_checkpoint_sha256": stacked_row["checkpoint_sha256"],
            "bie_checkpoint_sha256": bie_row["checkpoint_sha256"],
            "quality_checkpoint_sha256": quality_row["checkpoint_sha256"],
        },
    }
    temporary_report = report_path.with_suffix(".tmp.json")
    temporary_report.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary_report.replace(report_path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("preflight", "validation", "test"), required=True)
    parser.add_argument("--dataset-root", type=Path)
    parser.add_argument(
        "--specification", type=Path,
        default=Path("configs/proposal_v3_1_multifold_duration_evaluation.json"),
    )
    parser.add_argument(
        "--audit", type=Path, default=Path("reports/v3_multifold_execution_audit.json")
    )
    parser.add_argument(
        "--state", type=Path, default=Path("reports/v3_multifold_validation_state.json")
    )
    parser.add_argument(
        "--authorization", type=Path,
        default=Path("reports/v3_multifold_test_authorization.json"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/v3_multifold_scoring")
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--inference-batch-size", type=int, default=512)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.phase == "preflight":
        authorization = build_authorization(args.specification, args.audit, args.state)
        args.authorization.parent.mkdir(parents=True, exist_ok=True)
        args.authorization.write_text(
            json.dumps(authorization, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps({key: authorization[key] for key in (
            "authorized", "validation_jobs_verified", "test_trial_files_opened", "decision"
        )}, indent=2))
        return 0
    if args.dataset_root is None:
        raise SystemExit("--dataset-root is required for validation and test scoring")
    specification, _, state = verify_completed_gate(
        args.specification, args.audit, args.state
    )
    verify_authorization(
        args.authorization, args.specification, args.audit, args.state
    )
    manifest_path = Path(str(specification["trial_manifest"]["path"]))
    manifest = _json(manifest_path)
    device = resolve_device(args.device)
    for fold in map(int, specification["runs"]["folds"]):
        for seed in map(int, specification["runs"]["training_seeds"]):
            _score_bundle(
                role=args.phase,
                fold=fold,
                seed=seed,
                dataset_root=args.dataset_root.resolve(),
                specification=specification,
                state=state,
                manifest=manifest,
                manifest_dir=manifest_path.parent,
                output_dir=args.output_dir,
                authorization_path=args.authorization,
                device=device,
                inference_batch_size=args.inference_batch_size,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
