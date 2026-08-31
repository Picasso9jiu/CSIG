#!/usr/bin/env python3
"""Recreate the label-free M243 empty-bin report from frozen raw scores.

The script is an audit tool, not a test-time optimisation routine. It reads
only public test events, the M233 prediction package and frozen model scores.
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
BASE_THRESHOLD = 0.724
CANDIDATE_THRESHOLD = 0.718
EMPTY_TOP_K = (1, 2, 4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", required=True, type=Path)
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--base-zip", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference-report", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    test_root = args.test_root.resolve()
    raw_root = args.raw_dir.resolve()
    base_zip = args.base_zip.resolve()
    output = args.output.resolve()
    for path in (test_root, raw_root, base_zip):
        if not path.exists():
            raise FileNotFoundError(path)

    # configs.configs parses argv during import. The audit intentionally uses
    # the immutable release YAML and sets the postprocessor explicitly below.
    sys.argv = [sys.argv[0], "--config", str(ROOT / "configs" / "evisseg_evuav.yaml")]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from configs.configs import cfg
    from utils.component_background_verifier import extract_components
    from utils.postprocess import ChallengePostprocessor
    from utils.track_quality_bonus import P32TrackQualityBonus, P32TrackQualityBonusConfig

    p32 = P32TrackQualityBonusConfig(
        enabled=True,
        candidate_floor=0.60,
        spatial_radius=2,
        temporal_bin_size=50,
        max_link_distance=8.0,
        max_gap_bins=2,
        min_track_bins=4,
        min_seed_components=3,
        bonus=0.010,
        max_score_cap=0.97,
        max_motion_residual=2.0,
        velocity_history_bins=2,
    )

    def base_labels(name: str) -> np.ndarray:
        with zipfile.ZipFile(base_zip) as archive:
            return np.loadtxt(archive.open(name), dtype=np.int8, usecols=(4,))

    def candidate_mask(raw: np.ndarray, locs: torch.Tensor) -> np.ndarray:
        scores = torch.from_numpy(raw.astype(np.float32, copy=False)).clone()
        post, _ = ChallengePostprocessor.from_cfg(cfg, CANDIDATE_THRESHOLD).apply(scores, locs)
        boosted, _ = P32TrackQualityBonus(p32, CANDIDATE_THRESHOLD).apply(post.clone(), locs)
        return boosted.reshape(-1).cpu().numpy() >= CANDIDATE_THRESHOLD

    rows = []
    aggregate = {"videos": 0, "candidate_components": 0, "empty_components": 0, "empty_new_events": 0}
    raw_paths = sorted(raw_root.glob("test_*.npy"))
    if [path.name for path in raw_paths] != ["test_022.npy", "test_023.npy"]:
        raise AssertionError("M243 must audit exactly test_022.npy and test_023.npy")

    for raw_path in raw_paths:
        name = raw_path.stem
        source = test_root / (name + ".npz")
        with np.load(source, allow_pickle=False) as payload:
            events = payload["ev"]
            loc_np = np.concatenate(
                (np.zeros((len(events), 1), dtype=np.int64), payload["ev_loc"].astype(np.int64)), axis=1
            )
        raw = np.load(raw_path, allow_pickle=False).reshape(-1)
        if len(raw) != len(events):
            raise AssertionError("{} raw-score length differs from event count".format(name))
        base = base_labels(name + ".txt").reshape(-1).astype(bool)
        if len(base) != len(raw):
            raise AssertionError("{} base-label length differs from raw score length".format(name))
        locs = torch.from_numpy(loc_np).contiguous()
        candidate = candidate_mask(raw, locs)
        base_bins = set(np.floor_divide(loc_np[:, 3][base], 50).tolist())
        components, _ = extract_components(raw, locs, candidate, len(events))
        empty_rows = []
        candidate_new_events = 0
        for order, component in enumerate(components):
            indices = np.asarray(component.event_indices, dtype=np.int64)
            new_indices = indices[~base[indices]]
            if not len(new_indices):
                continue
            candidate_new_events += int(len(new_indices))
            if int(component.time_bin) not in base_bins:
                empty_rows.append(
                    {
                        "order": int(order),
                        "time_bin": int(component.time_bin),
                        "event_indices": new_indices,
                        "raw_scores": raw[new_indices],
                    }
                )
        variants = {}
        for k in EMPTY_TOP_K:
            ordered = sorted(empty_rows, key=lambda row: (-float(np.max(row["raw_scores"])), row["time_bin"], row["order"]))[:k]
            variants["empty_topk_{}".format(k)] = {
                "selected_components": len(ordered),
                "selected_events": len(ordered),
                "event_indices": [int(row["event_indices"][int(np.argmax(row["raw_scores"]))]) for row in ordered],
                "max_raw_scores": [float(np.max(row["raw_scores"])) for row in ordered],
            }
        rows.append(
            {
                "name": name,
                "event_count": int(len(events)),
                "base_positive": int(base.sum()),
                "candidate_positive": int(candidate.sum()),
                "candidate_new_events": candidate_new_events,
                "empty_components": len(empty_rows),
                "empty_new_events": int(sum(len(row["event_indices"]) for row in empty_rows)),
                "variants": variants,
            }
        )
        aggregate["videos"] += 1
        aggregate["candidate_components"] += len(components)
        aggregate["empty_components"] += len(empty_rows)
        aggregate["empty_new_events"] += int(sum(len(row["event_indices"]) for row in empty_rows))

    report = {
        "schema": "ev-uav-m243-test-actual-v1",
        "base_package": str(base_zip),
        "raw_score_dir": str(raw_root),
        "candidate_raw_threshold": CANDIDATE_THRESHOLD,
        "base_threshold": BASE_THRESHOLD,
        "rule": "M243: admit only new candidates in 50-unit time bins with no M233 positive event; top-k is per video, ranked by max raw score.",
        "label_usage": "No test labels are read.",
        "aggregate": aggregate,
        "videos": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("output:", output)
    for row in rows:
        print(row["name"], "top4_indices:", row["variants"]["empty_topk_4"]["event_indices"])
    if args.reference_report:
        reference = json.loads(args.reference_report.resolve().read_text(encoding="utf-8"))
        got = {row["name"]: row["variants"]["empty_topk_4"]["event_indices"] for row in rows}
        expected = {row["name"]: row["variants"]["empty_topk_4"]["event_indices"] for row in reference["videos"]}
        if got != expected:
            raise AssertionError("raw-score audit does not match the released M243 selection")
        print("reference_top4_selection: ok")


if __name__ == "__main__":
    main()
