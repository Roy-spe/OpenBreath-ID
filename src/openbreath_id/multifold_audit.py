"""Metadata-only execution audit for the frozen V3.1 multi-fold duration plan."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Mapping

from .trial_manifest import sha256_file


def audit_trial_outputs(
    manifest: Mapping[str, object],
    manifest_dir: Path,
    *,
    expected_folds: set[int],
    expected_conditions: set[str],
) -> dict[str, object]:
    """Hash validation files but use metadata-only checks for test files."""

    root = manifest_dir.resolve()
    expected = {
        (fold, role, condition)
        for fold in expected_folds
        for role in ("validation", "test")
        for condition in expected_conditions
    }
    observed: set[tuple[int, str, str]] = set()
    seen_paths: set[Path] = set()
    role_counts = {"validation": 0, "test": 0}
    validation_hashed = 0
    test_bytes = 0
    for output in manifest["outputs"]:
        fold = int(output["fold"])
        role = str(output["role"])
        condition = str(output["condition"])
        key = (fold, role, condition)
        if key in observed:
            raise ValueError(f"duplicate manifest output {key}")
        observed.add(key)
        if role not in role_counts:
            raise ValueError(f"unexpected trial role {role}")
        path = (manifest_dir / str(output["path"])).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"trial path escapes manifest directory: {path}")
        if path in seen_paths:
            raise ValueError(f"duplicate trial path: {path}")
        seen_paths.add(path)
        if not path.is_file():
            raise ValueError(f"trial file is missing: {path}")
        rows = int(output["rows"])
        genuine = int(output["genuine"])
        impostor = int(output["impostor"])
        if rows <= 0 or genuine <= 0 or impostor <= 0 or rows != genuine + impostor:
            raise ValueError(f"invalid manifest counts for {path}")
        expected_hash = str(output["sha256"])
        if len(expected_hash) != 64:
            raise ValueError(f"invalid SHA-256 metadata for {path}")
        if role == "validation":
            if sha256_file(path) != expected_hash:
                raise ValueError(f"validation trial hash mismatch for {path}")
            validation_hashed += 1
        else:
            # Deliberately call stat only: never open or decompress a test trial here.
            if path.stat().st_size <= 0:
                raise ValueError(f"test trial file is empty: {path}")
            test_bytes += 0
        role_counts[role] += 1
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"manifest design mismatch; missing={missing} extra={extra}")
    return {
        "files_resolved": len(observed),
        "validation_files_hashed": validation_hashed,
        "test_files_metadata_checked": role_counts["test"],
        "test_trial_files_opened": 0,
        "test_trial_bytes_read": test_bytes,
        "role_counts": role_counts,
    }


def audit_identity_registry(
    path: Path, *, expected_folds: set[int], expected_identities: int
) -> dict[str, object]:
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    required = {"fold", "subject_id", "split"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("identity registry has an invalid schema")
    fold_counts: dict[str, dict[str, int]] = {}
    all_subjects: set[str] = set()
    for fold in expected_folds:
        selected = [row for row in rows if int(row["fold"]) == fold]
        subjects = [row["subject_id"] for row in selected]
        if len(subjects) != expected_identities or len(set(subjects)) != expected_identities:
            raise ValueError(f"fold {fold} does not assign every identity exactly once")
        split_sets = {
            split: {row["subject_id"] for row in selected if row["split"] == split}
            for split in ("train", "validation", "test")
        }
        if set.union(*split_sets.values()) != set(subjects) or any(
            split_sets[left] & split_sets[right]
            for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
        ):
            raise ValueError(f"fold {fold} identity roles overlap or are incomplete")
        fold_counts[str(fold)] = {
            split: len(subjects_in_split)
            for split, subjects_in_split in split_sets.items()
        }
        all_subjects.update(subjects)
    if len(all_subjects) != expected_identities:
        raise ValueError("identity registry fold universes are inconsistent")
    return {"identities": len(all_subjects), "fold_counts": fold_counts}


def build_validation_only_job_plan(
    specification: Mapping[str, object], *, dataset_root_token: str = "${DATASET_ROOT}"
) -> list[dict[str, object]]:
    runs = specification["runs"]
    training = specification["encoder_training"]
    quality = specification["quality_training"]
    jobs: list[dict[str, object]] = []
    for fold in runs["folds"]:
        for seed in runs["training_seeds"]:
            for model, base_channels in (
                ("v3_stacked", training["stacked_base_channels"]),
                ("v3_bie", training["bie_base_channels"]),
            ):
                branch_dimension = (
                    training["stacked_branch_dimension"]
                    if model == "v3_stacked"
                    else training["bie_branch_dimension"]
                )
                stem = f"reports/v3_multifold/encoder_{model}_fold{fold}_seed{seed}"
                jobs.append(
                    {
                        "stage": "encoder_validation_only",
                        "fold": fold,
                        "seed": seed,
                        "model": model,
                        "output": f"{stem}.json",
                        "depends_on": [],
                        "argv": [
                            "openbreath-neural",
                            "--dataset-root", dataset_root_token,
                            "--output", f"{stem}.json",
                            "--fold", str(fold),
                            "--model", model,
                            "--epochs", str(training["epochs"]),
                            "--batches-per-epoch", str(training["batches_per_epoch"]),
                            "--subjects-per-batch", str(training["subjects_per_batch"]),
                            "--windows-per-subject", str(training["windows_per_subject"]),
                            "--maximum-training-windows-per-recording", str(training["maximum_training_windows_per_recording"]),
                            "--embedding-dimension", str(training["embedding_dimension"]),
                            "--branch-dimension", str(branch_dimension),
                            "--base-channels", str(base_channels),
                            "--learning-rate", str(training["learning_rate"]),
                            "--weight-decay", str(training["weight_decay"]),
                            "--arcface-scale", str(training["arcface_scale"]),
                            "--arcface-margin", str(training["arcface_margin"]),
                            "--loss", "arcface_supcon",
                            "--contrastive-temperature", str(training["contrastive_temperature"]),
                            "--supcon-weight", "0.5",
                            "--normalization", "shared_robust",
                            "--augmentation", "none",
                            "--channel-polarity", "correlation",
                            "--selection-probe-seconds", "60", "30",
                            "--selection-probe-weights", "1.0", "0.5",
                            "--guard-minutes", "30",
                            "--max-probes-per-subject", "120",
                            "--device", "auto",
                            "--validation-only",
                            "--seed", str(seed),
                            "--split-seed", "2027"
                        ]
                    }
                )
            encoder_stem = f"reports/v3_multifold/encoder_v3_bie_fold{fold}_seed{seed}"
            quality_output = f"reports/v3_multifold/quality_v3_bie_fold{fold}_seed{seed}.json"
            jobs.append(
                {
                    "stage": "quality_validation_only",
                    "fold": fold,
                    "seed": seed,
                    "model": "v3_bie_quality",
                    "output": quality_output,
                    "depends_on": [f"{encoder_stem}.json", f"{encoder_stem}.pt"],
                    "argv": [
                        "openbreath-quality",
                        "--dataset-root", dataset_root_token,
                        "--encoder-checkpoint", f"{encoder_stem}.pt",
                        "--encoder-report", f"{encoder_stem}.json",
                        "--output", quality_output,
                        "--fold", str(fold),
                        "--epochs", str(quality["quality_epochs"]),
                        "--cache-batches", str(quality["cache_batches"]),
                        "--subjects-per-batch", str(quality["subjects_per_batch"]),
                        "--windows-per-subject", str(quality["windows_per_subject"]),
                        "--maximum-training-windows-per-recording", str(quality["maximum_training_windows_per_recording"]),
                        "--quality-batch-size", str(quality["quality_batch_size"]),
                        "--learning-rate", str(quality["learning_rate"]),
                        "--weight-decay", str(quality["weight_decay"]),
                        "--ranking-margin", str(quality["ranking_margin"]),
                        "--utility-weight", str(quality["utility_loss_weight"]),
                        "--guard-minutes", str(quality["guard_minutes"]),
                        "--max-validation-probes", str(quality["maximum_validation_probes_per_identity"]),
                        "--inference-batch-size", str(quality["inference_batch_size"]),
                        "--device", "auto",
                        "--seed", str(seed),
                        "--split-seed", "2027"
                    ]
                }
            )
    return jobs


def _artifact_decision(payload: Mapping[str, object]) -> str | None:
    decision = payload.get("decision")
    if decision is not None:
        return str(decision)
    gate = payload.get("gate")
    if isinstance(gate, Mapping) and gate.get("decision") is not None:
        return str(gate["decision"])
    return None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--specification",
        type=Path,
        default=Path("configs/proposal_v3_1_multifold_duration_evaluation.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/v3_multifold_execution_audit.json")
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    specification = json.loads(args.specification.read_text(encoding="utf-8"))
    for artifact in specification["proposal_provenance"]:
        path = Path(artifact["path"])
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"proposal provenance mismatch for {path}")
    registry_spec = specification["development_identity_registry"]
    registry_path = Path(registry_spec["path"])
    if sha256_file(registry_path) != registry_spec["sha256"]:
        raise ValueError("identity registry provenance mismatch")
    identity_audit = audit_identity_registry(
        registry_path,
        expected_folds=set(map(int, registry_spec["folds"])),
        expected_identities=int(registry_spec["identities"]),
    )
    manifest_spec = specification["trial_manifest"]
    manifest_path = Path(manifest_spec["path"])
    if sha256_file(manifest_path) != manifest_spec["sha256"]:
        raise ValueError("trial manifest provenance mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    conditions = set(map(str, manifest["conditions"]))
    if len(conditions) != int(manifest_spec["conditions"]):
        raise ValueError("trial manifest condition count mismatch")
    trial_audit = audit_trial_outputs(
        manifest,
        manifest_path.parent,
        expected_folds=set(map(int, specification["runs"]["folds"])),
        expected_conditions=conditions,
    )
    artifact_audit = []
    for artifact in specification["completed_fold0_screening_provenance"]:
        path = Path(artifact["path"])
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"screening provenance mismatch for {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        decision = _artifact_decision(payload)
        if decision != artifact["decision"]:
            raise ValueError(f"screening decision mismatch for {path}: {decision}")
        artifact_audit.append({"path": str(path), "decision": decision})
    jobs = build_validation_only_job_plan(specification)
    payload = {
        "protocol": "V3.1 multi-fold validation-only execution dry run",
        "passed": True,
        "outer_test_identities_scored": 0,
        **trial_audit,
        "identity_registry": identity_audit,
        "screening_artifacts": artifact_audit,
        "job_plan": {
            "encoder_validation_only_jobs": sum(
                row["stage"] == "encoder_validation_only" for row in jobs
            ),
            "quality_validation_only_jobs": sum(
                row["stage"] == "quality_validation_only" for row in jobs
            ),
            "jobs": jobs,
        },
        "provenance": {
            "specification_sha256": sha256_file(args.specification),
            "identity_registry_sha256": sha256_file(registry_path),
            "trial_manifest_sha256": sha256_file(manifest_path),
        },
        "next_gate": "Run validation-only jobs; verify every report and checkpoint before authorizing any test-role scoring."
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": True,
                "files_resolved": trial_audit["files_resolved"],
                "test_trial_files_opened": 0,
                "validation_only_jobs": len(jobs),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
