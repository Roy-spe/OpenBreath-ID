"""Deterministic subject-disjoint outer-fold construction."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable


@dataclass(frozen=True, slots=True)
class IdentitySplit:
    fold: int
    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]

    def validate(self) -> None:
        train, validation, test = map(set, (self.train, self.validation, self.test))
        if train & validation or train & test or validation & test:
            raise ValueError(f"fold {self.fold} contains identity leakage")
        if len(train | validation | test) != len(self.train) + len(self.validation) + len(self.test):
            raise ValueError(f"fold {self.fold} contains duplicate identities")


def make_outer_splits(
    subjects: Iterable[str],
    *,
    n_folds: int = 5,
    validation_subjects: int = 15,
    seed: int = 2027,
) -> list[IdentitySplit]:
    """Make rotating test folds and fold-specific validation identities.

    For 97 identities this yields test sizes 20/20/19/19/19 and train sizes
    62/62/63/63/63, with 15 validation identities in every fold.
    """

    identities = sorted(set(subjects))
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2")
    if len(identities) < n_folds + validation_subjects + 1:
        raise ValueError("not enough subjects for the requested folds and validation set")

    shuffled = identities.copy()
    random.Random(seed).shuffle(shuffled)
    base, extra = divmod(len(shuffled), n_folds)
    test_folds: list[tuple[str, ...]] = []
    cursor = 0
    for fold in range(n_folds):
        size = base + (1 if fold < extra else 0)
        test_folds.append(tuple(sorted(shuffled[cursor : cursor + size])))
        cursor += size

    splits: list[IdentitySplit] = []
    universe = set(identities)
    for fold, test in enumerate(test_folds):
        development = sorted(universe - set(test))
        random.Random(seed + 10_000 + fold).shuffle(development)
        validation = tuple(sorted(development[:validation_subjects]))
        train = tuple(sorted(development[validation_subjects:]))
        split = IdentitySplit(fold=fold, train=train, validation=validation, test=test)
        split.validate()
        if set(split.train) | set(split.validation) | set(split.test) != universe:
            raise ValueError(f"fold {fold} does not cover the complete subject set")
        splits.append(split)
    return splits
