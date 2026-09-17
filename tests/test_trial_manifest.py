import csv
import gzip
from pathlib import Path

import numpy as np

from openbreath_id.trial_manifest import (
    SourceRecord,
    TrialCondition,
    iter_trial_rows,
    select_probe_starts,
    write_gzip_trial_csv,
)


def _condition() -> TrialCondition:
    return TrialCondition(
        name="wake_wake_e300_p60_g30",
        endpoint="primary",
        enrollment_seconds=300,
        probe_seconds=60,
        guard_minutes=30,
        max_probes_per_subject=2,
    )


def _sources() -> dict[str, SourceRecord]:
    samples = (300 + 30 * 60 + 10 * 60) * 6
    return {
        subject: SourceRecord(
            subject_id=subject,
            session_id="primary",
            state="wake",
            relative_path=f"{subject}/wake.mat",
            path=Path(f"{subject}/wake.mat"),
            n_samples=samples,
            sha256=(str(index) * 64)[:64],
        )
        for index, subject in enumerate(("subj_001", "subj_002", "subj_003"), 1)
    }


def test_probe_starts_respect_enrollment_guard_and_limit() -> None:
    condition = _condition()
    starts = select_probe_starts(
        (300 + 30 * 60 + 10 * 60) * 6,
        condition,
        sample_rate_hz=6,
    )
    assert starts.tolist() == [12600, 15840]
    assert np.diff(starts).min() >= condition.probe_seconds * 6


def test_trials_are_balanced_and_subject_disjoint_by_role() -> None:
    rows = list(
        iter_trial_rows(
            _sources(),
            ("subj_001", "subj_002", "subj_003"),
            _condition(),
            fold=0,
            role="test",
            sample_rate_hz=6,
            temporal_guard_semantics="sample-index separation",
        )
    )
    assert len(rows) == 18
    assert sum(row["genuine_or_impostor"] == "genuine" for row in rows) == 6
    assert sum(row["genuine_or_impostor"] == "impostor" for row in rows) == 12
    assert len({row["trial_id"] for row in rows}) == len(rows)
    assert {row["role"] for row in rows} == {"test"}
    assert min(int(row["probe_start_sample"]) for row in rows) == 12600


def test_compressed_trial_output_is_deterministic(tmp_path: Path) -> None:
    rows = list(
        iter_trial_rows(
            _sources(),
            ("subj_001", "subj_002", "subj_003"),
            _condition(),
            fold=0,
            role="validation",
            sample_rate_hz=6,
            temporal_guard_semantics="sample-index separation",
        )
    )
    first = tmp_path / "first.csv.gz"
    second = tmp_path / "second.csv.gz"
    first_counts = write_gzip_trial_csv(first, rows)
    second_counts = write_gzip_trial_csv(second, rows)
    assert first_counts == second_counts == {"rows": 18, "genuine": 6, "impostor": 12}
    assert first.read_bytes() == second.read_bytes()
    with gzip.open(first, mode="rt", encoding="utf-8", newline="") as stream:
        restored = list(csv.DictReader(stream))
    assert len(restored) == 18
    assert restored[0]["condition"] == _condition().name


def test_nested_duration_conditions_reuse_five_minute_probe_anchors() -> None:
    short = TrialCondition(
        name="wake_wake_e30_p30_g30_nested300",
        endpoint="duration_secondary",
        enrollment_seconds=30,
        probe_seconds=30,
        guard_minutes=30,
        max_probes_per_subject=3,
        enrollment_anchor_seconds=300,
        probe_anchor_seconds=300,
    )
    long = TrialCondition(
        name="wake_wake_e300_p300_g30_nested300",
        endpoint="duration_secondary",
        enrollment_seconds=300,
        probe_seconds=300,
        guard_minutes=30,
        max_probes_per_subject=3,
        enrollment_anchor_seconds=300,
        probe_anchor_seconds=300,
    )
    n_samples = (300 + 30 * 60 + 20 * 60) * 6
    short_starts = select_probe_starts(n_samples, short, sample_rate_hz=6)
    long_starts = select_probe_starts(n_samples, long, sample_rate_hz=6)
    assert short_starts.tolist() == long_starts.tolist()
    assert short_starts[0] == (300 + 30 * 60) * 6
    assert np.diff(short_starts).min() >= 300 * 6
