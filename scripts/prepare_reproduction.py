"""Local-only data preflight, deterministic split reconstruction and job preview.

No training, scoring, downloads, trial generation or subprocess execution occurs.
By default this script only prints aggregate checks. --output-dir must name a
new directory under this repository's ignored private/ directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from openbreath_id.data import discover_recordings, read_signal_metadata
from openbreath_id.multifold_audit import build_validation_only_job_plan
from openbreath_id.protocol import make_outer_splits

SPEC = ROOT / "configs/primary.json"
DURATION_SPEC = SPEC
MODULES = {
    "openbreath-neural": "openbreath_id.neural_train",
    "openbreath-quality": "openbreath_id.quality",
}


def split_bytes(subjects, *, seed=2027):
    """Match the historical writer's role order, UTF-8 and CRLF exactly."""
    splits = make_outer_splits(subjects, n_folds=5, validation_subjects=15, seed=seed)
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=["fold", "subject_id", "split"],
                            lineterminator="\r\n")
    writer.writeheader()
    for split in splits:
        for role in ("train", "validation", "test"):
            for subject in getattr(split, role):
                writer.writerow({"fold": split.fold, "subject_id": subject, "split": role})
    return stream.getvalue().encode("utf-8"), splits


def check_cohort(subjects, specification):
    registry = specification["development_identity_registry"]
    if len(subjects) != len(set(subjects)):
        raise ValueError("Duplicate primary Wake recordings; cohort is ambiguous.")
    if len(subjects) != registry["identities"]:
        raise ValueError("Primary Wake cohort size differs from the frozen registry.")
    content, splits = split_bytes(subjects, seed=registry["split_seed"])
    digest = hashlib.sha256(content).hexdigest().upper()
    if digest != registry["sha256"].upper():
        raise ValueError("Reconstructed registry hash differs; do not claim matched partitions.")
    return content, splits, digest


def check_output_directory(output_dir, dataset_root):
    output = Path(output_dir).resolve()
    private = (ROOT / "private").resolve()
    dataset = Path(dataset_root).resolve()
    if output == private or not output.is_relative_to(private):
        raise ValueError("Output must be a new subdirectory under repository private/.")
    if output.exists():
        raise ValueError("Output already exists; choose a new name. No overwrite is allowed.")
    if output.is_relative_to(dataset) or dataset.is_relative_to(output):
        raise ValueError("Output must not overlap the dataset directory.")
    return output


def make_job_preview(specification, dataset_root, output):
    jobs = build_validation_only_job_plan(specification, dataset_root_token=str(dataset_root))
    run_root = output / "run_artifacts"
    for job in jobs:
        legacy = job["argv"]
        # Remap generated artifacts only, without changing scientific arguments.
        mapped = [str(run_root / value) if value.startswith("reports/") else value
                  for value in legacy[1:]]
        job["argv"] = [sys.executable, "-m", MODULES[legacy[0]], *mapped]
        job["output"] = str(run_root / job["output"])
        job["depends_on"] = [str(run_root / name) for name in job["depends_on"]]
        job["optional_historical_quality_prototype"] = job["stage"] == "quality_validation_only"
    return {
        "status": "preview_only_not_executed",
        "purpose": "Validation-only training recipe; not an EER reproduction runner.",
        "configuration_sha256": hashlib.sha256(SPEC.read_bytes()).hexdigest(),
        "encoder_jobs": sum(job["stage"] == "encoder_validation_only" for job in jobs),
        "optional_quality_jobs": sum(job["stage"] == "quality_validation_only" for job in jobs),
        "required_working_directory": str(ROOT),
        "warning": "Explicit later execution costs compute and creates private biometric artifacts. "
                   "Checkpoints, frozen trials and separate scoring/uncertainty steps are still needed.",
        "jobs": jobs,
    }


def prepare(dataset_root, output_dir=None):
    dataset = Path(dataset_root).resolve()
    if not dataset.is_dir():
        raise ValueError("Dataset root is not a directory.")
    output = check_output_directory(output_dir, dataset) if output_dir is not None else None
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    duration_spec = json.loads(DURATION_SPEC.read_text(encoding="utf-8"))
    records = [record for record in discover_recordings(dataset, states=("wake",))
               if record.session_id == "primary"]
    content, splits, digest = check_cohort([record.subject_id for record in records], spec)
    recipe = duration_spec["duration_experiment"]
    minimum_samples = round(duration_spec["data"]["sampling_hz"] * (
        recipe["enrollment_anchor_seconds"] + 60 * recipe["guard_minutes"]
        + recipe["probe_anchor_seconds"]))
    for record in records:
        metadata = read_signal_metadata(record.path, duration_spec["data"]["signal_variable"])
        if metadata.n_samples < minimum_samples:
            raise ValueError("At least one primary Wake file is too short for a complete nested anchor.")
    summary = {
        "status": "PASS",
        "primary_wake_recordings": len(records),
        "mat_headers_valid": True,
        "registry_sha256_matches_frozen": True,
        "registry_sha256": digest,
        "fold_sizes": [{"fold": split.fold, "train": len(split.train),
                        "validation": len(split.validation), "test": len(split.test)} for split in splits],
        "source_signal_bytes_verified_against_historical_data": False,
        "training_or_scoring_executed": False,
        "trial_manifests_generated": False,
        "files_written": output is not None,
        "interpretation": "Header/cohort checks only, not signal-quality, source-content or EER validation.",
    }
    if output is not None:
        preview = make_job_preview(spec, dataset, output)
        output.mkdir(parents=True, exist_ok=False)
        (output / "subject_splits.csv").write_bytes(content)
        (output / "PREFLIGHT.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        (output / "VALIDATION_JOB_PREVIEW.json").write_text(
            json.dumps(preview, indent=2) + "\n", encoding="utf-8")
        (output / "DO_NOT_PUBLISH.txt").write_text(
            "Local participant split assignments and machine-specific paths. Do not publish.\n"
            "This is newly generated preparation metadata, not the historical trial-manifest freeze.\n",
            encoding="utf-8")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, help="Optional new directory under repository private/.")
    args = parser.parse_args(argv)
    try:
        result = prepare(args.dataset_root, args.output_dir)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Preflight stopped: {exc}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
