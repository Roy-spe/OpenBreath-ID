"""Synthetic-only tests for the public preparation adapter; no study IDs/data."""
import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.io import savemat

ROOT = Path(__file__).resolve().parents[1]
loader = importlib.util.spec_from_file_location("preparation", ROOT / "scripts/prepare_reproduction.py")
prep = importlib.util.module_from_spec(loader)
loader.loader.exec_module(prep)


def synthetic_cohort():
    return [f"synthetic_person_{i:03d}" for i in range(97)]


def synthetic_specification():
    content, _ = prep.split_bytes(synthetic_cohort())
    return {"development_identity_registry": {"identities": 97, "split_seed": 2027,
            "sha256": hashlib.sha256(content).hexdigest()}}


def test_splits_are_deterministic_and_role_disjoint():
    content, splits = prep.split_bytes(synthetic_cohort())
    reversed_content, _ = prep.split_bytes(reversed(synthetic_cohort()))
    assert content == reversed_content
    assert content.count(b"\r\n") == 486
    assert [len(split.test) for split in splits] == [20, 20, 19, 19, 19]
    assert all(len(split.validation) == 15 for split in splits)
    for split in splits:
        split.validate()
    assert len({person for split in splits for person in split.test}) == 97


def test_cohort_checks_reject_missing_duplicate_and_wrong_identity():
    people = synthetic_cohort()
    spec = synthetic_specification()
    prep.check_cohort(people, spec)
    for invalid in (people[:-1], people[:-1] + [people[0]], people[:-1] + ["synthetic_other"]):
        with pytest.raises(ValueError):
            prep.check_cohort(invalid, spec)


def test_output_must_be_new_private_nonoverlapping_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(prep, "ROOT", tmp_path)
    dataset = tmp_path / "data"
    valid = tmp_path / "private" / "run1"
    assert prep.check_output_directory(valid, dataset) == valid.resolve()
    for invalid in (tmp_path / "reports", tmp_path / "private", tmp_path / "private/../../outside"):
        with pytest.raises(ValueError):
            prep.check_output_directory(invalid, dataset)
    valid.mkdir(parents=True)
    with pytest.raises(ValueError):
        prep.check_output_directory(valid, dataset)
    with pytest.raises(ValueError):
        prep.check_output_directory(tmp_path / "private/data/run", tmp_path / "private/data")


def test_preview_is_complete_and_accepted_by_released_parsers(tmp_path):
    from openbreath_id.neural_train import parse_args as neural_args
    from openbreath_id.quality import parse_args as quality_args
    spec = json.loads(prep.SPEC.read_text(encoding="utf-8"))
    output = tmp_path / "private/run1"
    preview = prep.make_job_preview(spec, tmp_path / "data", output)
    assert not output.exists()
    assert preview["encoder_jobs"] == 30
    assert preview["optional_quality_jobs"] == 15
    assert len({(job["fold"], job["seed"], job["model"]) for job in preview["jobs"]}) == 45
    for job in preview["jobs"]:
        assert job["argv"][:2] == [sys.executable, "-m"]
        assert Path(job["output"]).is_relative_to(output)
        if job["stage"] == "encoder_validation_only":
            args = neural_args(job["argv"][3:])
            assert args.validation_only and args.epochs == 40
            assert args.base_channels == (72 if job["model"] == "v3_stacked" else 40)
            assert args.selection_probe_seconds == [60, 30]
            assert args.selection_probe_weights == [1.0, 0.5]
        else:
            args = quality_args(job["argv"][3:])
            assert args.epochs == 20
            assert Path(args.encoder_report).is_relative_to(output)


def test_header_preflight_default_writes_nothing_and_optional_output_is_private(tmp_path, monkeypatch):
    from openbreath_id.data import Recording
    dataset = tmp_path / "data"
    dataset.mkdir()
    signal = dataset / "synthetic.mat"
    savemat(signal, {"fieldValue": np.zeros((14400, 2))})
    records = [Recording(person, "primary", "wake", signal) for person in synthetic_cohort()]
    monkeypatch.setattr(prep, "discover_recordings", lambda *args, **kwargs: records)
    config = json.loads(prep.SPEC.read_text(encoding="utf-8"))
    config.update(synthetic_specification())
    spec_path = tmp_path / "synthetic_spec.json"
    spec_path.write_text(json.dumps(config), encoding="utf-8")
    monkeypatch.setattr(prep, "SPEC", spec_path)
    monkeypatch.setattr(prep, "ROOT", tmp_path)
    before = set(tmp_path.rglob("*"))
    result = prep.prepare(dataset)
    assert not result["files_written"]
    assert set(tmp_path.rglob("*")) == before
    output = tmp_path / "private/run1"
    assert prep.prepare(dataset, output)["files_written"]
    assert {p.name for p in output.iterdir()} == {
        "subject_splits.csv", "PREFLIGHT.json", "VALIDATION_JOB_PREVIEW.json", "DO_NOT_PUBLISH.txt"}
    assert not (output / "run_artifacts").exists()
    with pytest.raises(ValueError):
        prep.prepare(dataset, output)


def test_short_recording_stops_without_writing(tmp_path, monkeypatch):
    from openbreath_id.data import Recording
    dataset = tmp_path / "data"
    dataset.mkdir()
    signal = dataset / "synthetic.mat"
    savemat(signal, {"fieldValue": np.zeros((100, 2))})
    records = [Recording(person, "primary", "wake", signal) for person in synthetic_cohort()]
    monkeypatch.setattr(prep, "discover_recordings", lambda *args, **kwargs: records)
    config_path = tmp_path / "synthetic_spec.json"
    config_path.write_text(json.dumps(synthetic_specification()), encoding="utf-8")
    monkeypatch.setattr(prep, "SPEC", config_path)
    monkeypatch.setattr(prep, "ROOT", tmp_path)
    output = tmp_path / "private/run1"
    with pytest.raises(ValueError, match="too short"):
        prep.prepare(dataset, output)
    assert not output.exists()
