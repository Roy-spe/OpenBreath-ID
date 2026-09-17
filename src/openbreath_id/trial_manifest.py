"""Freeze reproducible subject-disjoint enrollment-probe trial manifests."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import gzip
import hashlib
import io
import json
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np

from .data import Recording, discover_recordings, read_signal_metadata
from .protocol import IdentitySplit


TRIAL_FIELDS = (
    "trial_id",
    "fold",
    "role",
    "condition",
    "endpoint",
    "enroll_subject",
    "probe_subject",
    "enroll_session",
    "probe_session",
    "enroll_state",
    "probe_state",
    "enroll_start_sample",
    "enroll_end_sample",
    "probe_start_sample",
    "probe_end_sample",
    "genuine_or_impostor",
    "temporal_guard_semantics",
    "enroll_source",
    "probe_source",
    "enroll_source_sha256",
    "probe_source_sha256",
)


@dataclass(frozen=True, slots=True)
class TrialCondition:
    name: str
    endpoint: str
    enrollment_seconds: int
    probe_seconds: int
    guard_minutes: int
    max_probes_per_subject: int
    enrollment_anchor_seconds: int | None = None
    probe_anchor_seconds: int | None = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "TrialCondition":
        condition = cls(
            name=str(values["name"]),
            endpoint=str(values["endpoint"]),
            enrollment_seconds=int(values["enrollment_seconds"]),
            probe_seconds=int(values["probe_seconds"]),
            guard_minutes=int(values["guard_minutes"]),
            max_probes_per_subject=int(values["max_probes_per_subject"]),
            enrollment_anchor_seconds=(
                int(values["enrollment_anchor_seconds"])
                if "enrollment_anchor_seconds" in values
                else None
            ),
            probe_anchor_seconds=(
                int(values["probe_anchor_seconds"])
                if "probe_anchor_seconds" in values
                else None
            ),
        )
        condition.validate()
        return condition

    def validate(self) -> None:
        if not self.name or any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_" for character in self.name):
            raise ValueError(f"invalid condition name: {self.name!r}")
        if self.enrollment_seconds < 30 or self.enrollment_seconds % 30:
            raise ValueError("enrollment_seconds must be a positive multiple of 30")
        if self.probe_seconds < 30 or self.probe_seconds % 30:
            raise ValueError("probe_seconds must be a positive multiple of 30")
        if self.guard_minutes < 0:
            raise ValueError("guard_minutes must be nonnegative")
        if self.max_probes_per_subject < 1:
            raise ValueError("max_probes_per_subject must be positive")
        enrollment_anchor = self.enrollment_anchor_seconds or self.enrollment_seconds
        probe_anchor = self.probe_anchor_seconds or self.probe_seconds
        if enrollment_anchor < self.enrollment_seconds or enrollment_anchor % 30:
            raise ValueError(
                "enrollment_anchor_seconds must be a multiple of 30 and at least enrollment_seconds"
            )
        if probe_anchor < self.probe_seconds or probe_anchor % 30:
            raise ValueError(
                "probe_anchor_seconds must be a multiple of 30 and at least probe_seconds"
            )


@dataclass(frozen=True, slots=True)
class SourceRecord:
    subject_id: str
    session_id: str
    state: str
    relative_path: str
    path: Path
    n_samples: int
    sha256: str


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _resolved_project_file(project_root: Path, value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else project_root / path


def load_specification(path: str | Path, *, project_root: str | Path) -> dict[str, object]:
    specification_path = Path(path).resolve()
    specification = json.loads(specification_path.read_text(encoding="utf-8"))
    root = Path(project_root).resolve()

    registry = specification["development_identity_registry"]
    registry_path = _resolved_project_file(root, registry["path"])
    observed_registry_hash = sha256_file(registry_path)
    expected_registry_hash = str(registry["sha256"]).upper()
    if observed_registry_hash != expected_registry_hash:
        raise ValueError(
            "development identity registry hash mismatch: "
            f"expected {expected_registry_hash}, observed {observed_registry_hash}"
        )

    proposal = specification.get("proposal_provenance")
    if proposal:
        proposal_path = _resolved_project_file(root, proposal["path"])
        observed_proposal_hash = sha256_file(proposal_path)
        expected_proposal_hash = str(proposal["sha256"]).upper()
        if observed_proposal_hash != expected_proposal_hash:
            raise ValueError(
                "proposal provenance hash mismatch: "
                f"expected {expected_proposal_hash}, observed {observed_proposal_hash}"
            )
    parent = specification.get("parent_protocol")
    if parent:
        parent_path = _resolved_project_file(root, parent["path"])
        observed_parent_hash = sha256_file(parent_path)
        expected_parent_hash = str(parent["sha256"]).upper()
        if observed_parent_hash != expected_parent_hash:
            raise ValueError(
                "parent protocol hash mismatch: "
                f"expected {expected_parent_hash}, observed {observed_parent_hash}"
            )
    return specification


def read_subject_splits(path: str | Path) -> list[IdentitySplit]:
    grouped: dict[int, dict[str, list[str]]] = {}
    with Path(path).open(newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream)
        required = {"fold", "subject_id", "split"}
        if reader.fieldnames is None or not required <= set(reader.fieldnames):
            raise ValueError("split registry must contain fold, subject_id, and split")
        for row in reader:
            fold = int(row["fold"])
            role = row["split"].strip()
            if role not in {"train", "validation", "test"}:
                raise ValueError(f"invalid split role {role!r}")
            grouped.setdefault(
                fold, {"train": [], "validation": [], "test": []}
            )[role].append(row["subject_id"].strip())

    if not grouped:
        raise ValueError("split registry is empty")
    splits: list[IdentitySplit] = []
    universe: set[str] | None = None
    for fold in sorted(grouped):
        roles = grouped[fold]
        split = IdentitySplit(
            fold=fold,
            train=tuple(sorted(roles["train"])),
            validation=tuple(sorted(roles["validation"])),
            test=tuple(sorted(roles["test"])),
        )
        split.validate()
        current = set(split.train) | set(split.validation) | set(split.test)
        if universe is None:
            universe = current
        elif current != universe:
            raise ValueError(f"fold {fold} does not contain the same identity universe")
        splits.append(split)
    if [split.fold for split in splits] != list(range(len(splits))):
        raise ValueError("fold numbers must be contiguous from zero")
    return splits


def discover_primary_sources(
    dataset_root: str | Path,
    *,
    state: str,
    signal_variable: str,
) -> dict[str, SourceRecord]:
    root = Path(dataset_root).resolve()
    recordings = [
        recording
        for recording in discover_recordings(root, states=(state,))
        if recording.session_id == "primary"
    ]
    sources: dict[str, SourceRecord] = {}
    for recording in recordings:
        if recording.subject_id in sources:
            raise ValueError(
                f"multiple primary {state} recordings for {recording.subject_id}"
            )
        metadata = read_signal_metadata(recording.path, variable=signal_variable)
        sources[recording.subject_id] = SourceRecord(
            subject_id=recording.subject_id,
            session_id=recording.session_id,
            state=recording.state,
            relative_path=recording.path.relative_to(root).as_posix(),
            path=recording.path,
            n_samples=metadata.n_samples,
            sha256=sha256_file(recording.path),
        )
    return sources


def select_probe_starts(
    n_samples: int,
    condition: TrialCondition,
    *,
    sample_rate_hz: int,
) -> np.ndarray:
    enrollment_anchor_seconds = (
        condition.enrollment_anchor_seconds or condition.enrollment_seconds
    )
    probe_anchor_seconds = condition.probe_anchor_seconds or condition.probe_seconds
    first_probe = (
        enrollment_anchor_seconds + condition.guard_minutes * 60
    ) * sample_rate_hz
    probe_anchor_samples = probe_anchor_seconds * sample_rate_hz
    groups = (n_samples - first_probe) // probe_anchor_samples
    if groups < 1:
        return np.empty(0, dtype=np.int64)
    if groups <= condition.max_probes_per_subject:
        selected = np.arange(groups, dtype=np.int64)
    else:
        selected = np.linspace(
            0,
            groups - 1,
            condition.max_probes_per_subject,
            dtype=np.int64,
        )
    return first_probe + selected * probe_anchor_samples


def _trial_id(
    fold: int,
    role: str,
    condition: str,
    enroll_subject: str,
    probe_subject: str,
    probe_start: int,
) -> str:
    key = "|".join(
        (
            str(fold),
            role,
            condition,
            enroll_subject,
            probe_subject,
            str(probe_start),
        )
    )
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def iter_trial_rows(
    sources: Mapping[str, SourceRecord],
    subjects: Iterable[str],
    condition: TrialCondition,
    *,
    fold: int,
    role: str,
    sample_rate_hz: int,
    temporal_guard_semantics: str,
) -> Iterator[dict[str, object]]:
    identities = tuple(sorted(set(subjects)))
    if len(identities) < 2:
        raise ValueError("at least two identities are required to construct trials")
    missing = sorted(set(identities) - set(sources))
    if missing:
        raise ValueError(f"missing primary source recordings for: {', '.join(missing)}")

    enrollment_samples = condition.enrollment_seconds * sample_rate_hz
    probe_samples = condition.probe_seconds * sample_rate_hz
    for probe_subject in identities:
        probe_source = sources[probe_subject]
        starts = select_probe_starts(
            probe_source.n_samples,
            condition,
            sample_rate_hz=sample_rate_hz,
        )
        if not len(starts):
            raise ValueError(
                f"{probe_subject} has no valid probes for condition {condition.name}"
            )
        for probe_start in starts:
            for enroll_subject in identities:
                enroll_source = sources[enroll_subject]
                yield {
                    "trial_id": _trial_id(
                        fold,
                        role,
                        condition.name,
                        enroll_subject,
                        probe_subject,
                        int(probe_start),
                    ),
                    "fold": fold,
                    "role": role,
                    "condition": condition.name,
                    "endpoint": condition.endpoint,
                    "enroll_subject": enroll_subject,
                    "probe_subject": probe_subject,
                    "enroll_session": enroll_source.session_id,
                    "probe_session": probe_source.session_id,
                    "enroll_state": enroll_source.state,
                    "probe_state": probe_source.state,
                    "enroll_start_sample": 0,
                    "enroll_end_sample": enrollment_samples,
                    "probe_start_sample": int(probe_start),
                    "probe_end_sample": int(probe_start) + probe_samples,
                    "genuine_or_impostor": (
                        "genuine" if enroll_subject == probe_subject else "impostor"
                    ),
                    "temporal_guard_semantics": temporal_guard_semantics,
                    "enroll_source": enroll_source.relative_path,
                    "probe_source": probe_source.relative_path,
                    "enroll_source_sha256": enroll_source.sha256,
                    "probe_source_sha256": probe_source.sha256,
                }


def write_gzip_trial_csv(
    path: str | Path, rows: Iterable[Mapping[str, object]]
) -> dict[str, int]:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    counts = {"rows": 0, "genuine": 0, "impostor": 0}
    with temporary.open("wb") as raw_stream:
        with gzip.GzipFile(
            filename="", mode="wb", fileobj=raw_stream, mtime=0
        ) as gzip_stream:
            with io.TextIOWrapper(
                gzip_stream, encoding="utf-8", newline=""
            ) as text_stream:
                writer = csv.DictWriter(
                    text_stream, fieldnames=TRIAL_FIELDS, lineterminator="\n"
                )
                writer.writeheader()
                for row in rows:
                    writer.writerow(row)
                    counts["rows"] += 1
                    label = str(row["genuine_or_impostor"])
                    counts[label] += 1
    temporary.replace(output)
    return counts


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def generate_manifests(
    *,
    dataset_root: str | Path,
    specification_path: str | Path,
    output_dir: str | Path,
    project_root: str | Path,
    selected_conditions: Sequence[str] = (),
) -> dict[str, object]:
    root = Path(project_root).resolve()
    specification_file = Path(specification_path).resolve()
    specification = load_specification(specification_file, project_root=root)
    registry = specification["development_identity_registry"]
    split_path = _resolved_project_file(root, registry["path"])
    splits = read_subject_splits(split_path)
    expected_folds = int(registry["folds"])
    if len(splits) != expected_folds:
        raise ValueError(f"expected {expected_folds} folds, found {len(splits)}")

    data = specification["data"]
    enrollment_state = str(data["enrollment_state"])
    probe_state = str(data["probe_state"])
    if enrollment_state != probe_state:
        raise ValueError("initial manifest generator supports same-state conditions only")
    sample_rate_hz = int(data["sampling_hz"])
    sources = discover_primary_sources(
        dataset_root,
        state=enrollment_state,
        signal_variable=str(data["signal_variable"]),
    )
    universe = set(splits[0].train) | set(splits[0].validation) | set(splits[0].test)
    if set(sources) != universe:
        missing = sorted(universe - set(sources))
        extra = sorted(set(sources) - universe)
        raise ValueError(f"source/registry identity mismatch; missing={missing}, extra={extra}")
    if len(universe) != int(registry["identity_count"]):
        raise ValueError("identity count does not match the specification")

    conditions = [
        TrialCondition.from_mapping(values) for values in specification["conditions"]
    ]
    condition_names = {condition.name for condition in conditions}
    if len(condition_names) != len(conditions):
        raise ValueError("condition names must be unique")
    if selected_conditions:
        requested = set(selected_conditions)
        unknown = requested - condition_names
        if unknown:
            raise ValueError(f"unknown conditions: {', '.join(sorted(unknown))}")
        conditions = [condition for condition in conditions if condition.name in requested]

    output = Path(output_dir).resolve()
    source_registry = {
        "dataset_root": str(Path(dataset_root).resolve()),
        "signal_variable": str(data["signal_variable"]),
        "state": enrollment_state,
        "sources": [
            {
                "subject_id": source.subject_id,
                "session_id": source.session_id,
                "state": source.state,
                "relative_path": source.relative_path,
                "n_samples": source.n_samples,
                "sha256": source.sha256,
            }
            for source in sorted(sources.values(), key=lambda value: value.subject_id)
        ],
    }
    source_registry_path = output / "source_files.json"
    _write_json(source_registry_path, source_registry)

    roles = tuple(str(role) for role in specification["trial_construction"]["roles"])
    if not roles or not set(roles) <= {"validation", "test"}:
        raise ValueError("trial roles must contain validation and/or test")
    outputs: list[dict[str, object]] = []
    for split in splits:
        for condition in conditions:
            for role in roles:
                subjects = getattr(split, role)
                filename = f"fold_{split.fold}_{condition.name}_{role}.csv.gz"
                path = output / filename
                counts = write_gzip_trial_csv(
                    path,
                    iter_trial_rows(
                        sources,
                        subjects,
                        condition,
                        fold=split.fold,
                        role=role,
                        sample_rate_hz=sample_rate_hz,
                        temporal_guard_semantics=str(data["temporal_guard_semantics"]),
                    ),
                )
                outputs.append(
                    {
                        "fold": split.fold,
                        "role": role,
                        "condition": condition.name,
                        "path": filename,
                        "sha256": sha256_file(path),
                        "identities": len(subjects),
                        **counts,
                    }
                )

    manifest = {
        "manifest_version": "1.0",
        "specification_version": specification["specification_version"],
        "specification_path": str(specification_file),
        "specification_sha256": sha256_file(specification_file),
        "identity_registry_path": str(split_path.resolve()),
        "identity_registry_sha256": sha256_file(split_path),
        "source_registry": source_registry_path.name,
        "source_registry_sha256": sha256_file(source_registry_path),
        "temporal_guard_semantics": data["temporal_guard_semantics"],
        "conditions": [condition.name for condition in conditions],
        "outputs": outputs,
    }
    _write_json(output / "trial_manifest.json", manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--specification",
        type=Path,
        default=Path("configs/proposal_v3_development.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("metadata/trials"))
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--condition", action="append", default=[])
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = generate_manifests(
        dataset_root=args.dataset_root,
        specification_path=args.specification,
        output_dir=args.output_dir,
        project_root=args.project_root,
        selected_conditions=args.condition,
    )
    print(
        f"wrote {len(manifest['outputs'])} trial files for "
        f"{len(manifest['conditions'])} conditions to {args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
