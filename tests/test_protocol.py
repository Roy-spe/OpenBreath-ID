from openbreath_id.protocol import make_outer_splits


def test_97_subject_fold_sizes_and_disjointness() -> None:
    subjects = [f"subj_{index:03d}" for index in range(1, 98)]
    splits = make_outer_splits(subjects)
    assert [(len(s.train), len(s.validation), len(s.test)) for s in splits] == [
        (62, 15, 20),
        (62, 15, 20),
        (63, 15, 19),
        (63, 15, 19),
        (63, 15, 19),
    ]
    assert sorted(subject for split in splits for subject in split.test) == subjects
    for split in splits:
        split.validate()


def test_split_is_deterministic() -> None:
    subjects = [f"subj_{index:03d}" for index in range(1, 98)]
    assert make_outer_splits(subjects, seed=7) == make_outer_splits(reversed(subjects), seed=7)

