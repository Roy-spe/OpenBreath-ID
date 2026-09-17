"""Run the frozen primary EER multiplicity correction from scored archives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .multifold_aggregate import load_primary_tables
from .multifold_bootstrap import paired_primary_eer_holm
from .multifold_test import SYSTEMS
from .trial_manifest import sha256_file


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--specification", type=Path, default=Path("configs/proposal_v3_1_multifold_duration_evaluation.json"))
    parser.add_argument("--scoring-dir", type=Path, default=Path("reports/v3_multifold_scoring"))
    parser.add_argument("--authorization", type=Path, default=Path("reports/v3_multifold_test_authorization.json"))
    parser.add_argument("--output", type=Path, default=Path("reports/v3_multifold_primary_eer_holm.json"))
    parser.add_argument("--replicates", type=int, default=10_000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    specification = json.loads(args.specification.read_text(encoding="utf-8"))
    authorization = json.loads(args.authorization.read_text(encoding="utf-8"))
    if not authorization.get("authorized"):
        raise ValueError("frozen test scoring was not authorized")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(args.scoring_dir.glob("test_fold*_seed*.json"))]
    primary = list(map(str, specification["evaluation"]["primary_conditions"]))
    tables = load_primary_tables(reports, scoring_dir=args.scoring_dir, primary_conditions=primary)
    result = paired_primary_eer_holm(
        tables,
        comparisons=((SYSTEMS[1], SYSTEMS[0]), (SYSTEMS[2], SYSTEMS[1])),
        replicates=args.replicates,
        seed=int(specification["uncertainty"]["bootstrap_seed"]),
        workers=args.workers,
    )
    result["provenance"] = {
        "specification_sha256": sha256_file(args.specification),
        "authorization_sha256": sha256_file(args.authorization),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["hypotheses"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
