#!/usr/bin/env python3
"""Split public test videos into the fixed unlabeled M233/M244 routing domains."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np


LOW_MIDDLE_MAX_EVENTS = 200_000
EXTREME_H2_MIN_EVENTS = 500_000
H1_POLARITY_MINORITY_MAX = 0.20
THRESHOLDS = {
    "low_middle": 0.94,
    "h1": 0.62,
    "h2_other": 0.65,
    "h2_extreme": 0.63,
}


def minority_polarity_ratio(events: np.ndarray) -> float:
    if len(events) == 0:
        return 0.0
    _, counts = np.unique(events["p"], return_counts=True)
    return float(counts.min()) / float(len(events))


def classify(event_count: int, minority_ratio: float) -> str:
    if event_count <= LOW_MIDDLE_MAX_EVENTS:
        return "low_middle"
    if minority_ratio < H1_POLARITY_MINORITY_MAX:
        return "h1"
    if event_count >= EXTREME_H2_MIN_EVENTS:
        return "h2_extreme"
    return "h2_other"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--copy", action="store_true", help="copy instead of symlinking input NPZ files")
    args = parser.parse_args()

    test_root = args.test_root.resolve()
    output_root = args.output_root.resolve()
    sources = sorted(test_root.glob("test_*.npz"))
    if len(sources) != 31:
        raise AssertionError("expected 31 test NPZ files, found {}".format(len(sources)))
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError("output root is not empty: {}".format(output_root))

    rows = []
    for source in sources:
        with np.load(source, allow_pickle=False) as payload:
            events = payload["ev"]
            event_count = int(len(events))
            minority_ratio = minority_polarity_ratio(events)
        domain = classify(event_count, minority_ratio)
        target = output_root / domain / "test" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if args.copy:
            shutil.copy2(source, target)
        else:
            target.symlink_to(source)
        rows.append(
            {
                "name": source.name,
                "event_count": event_count,
                "minority_polarity_ratio": minority_ratio,
                "domain": domain,
                "m124_threshold": THRESHOLDS[domain],
            }
        )

    manifest = {
        "schema": "evsod-m244-domain-routing-v1",
        "uses_test_labels": False,
        "rules": {
            "low_middle_max_events": LOW_MIDDLE_MAX_EVENTS,
            "h1_minority_ratio_lt": H1_POLARITY_MINORITY_MAX,
            "extreme_h2_min_events": EXTREME_H2_MIN_EVENTS,
            "thresholds": THRESHOLDS,
        },
        "videos": rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    counts = {name: sum(row["domain"] == name for row in rows) for name in THRESHOLDS}
    print("domain_counts:", counts)
    print("manifest:", output_root / "manifest.json")


if __name__ == "__main__":
    main()
