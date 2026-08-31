#!/usr/bin/env python3
"""Rebuild the M244 submission from the released M233 package and audit report.

M244 changes exactly eight negative labels: the top raw-score event in each
of the four selected empty 50-unit bins of test_022 and test_023.  The report
was produced without test labels.  This script validates every output row
against the public test event order before writing the ZIP.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
M244_SHA256 = "390fa26a200bb80f4729318011484621eb998269db616b313b7f022620f60e20"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def expected_names(test_root: Path) -> List[str]:
    names = sorted(path.with_suffix(".txt").name for path in test_root.glob("test_*.npz"))
    if len(names) != 31:
        raise AssertionError("expected 31 test NPZ files, found {} in {}".format(len(names), test_root))
    return names


def set_label(line: bytes) -> bytes:
    end = len(line.rstrip(b"\r\n"))
    start = end
    while start > 0 and line[start - 1] not in b" \t":
        start -= 1
    if line[start:end] != b"0":
        raise AssertionError("M244 can only restore an existing negative label")
    return line[:start] + b"1" + line[end:]


def validate_member(name: str, data: bytes, test_root: Path) -> int:
    lines = data.splitlines(keepends=True)
    with np.load(test_root / (Path(name).stem + ".npz"), allow_pickle=False) as payload:
        events = payload["ev"]
    if len(lines) != len(events):
        raise AssertionError("{}: {} rows for {} events".format(name, len(lines), len(events)))
    positives = 0
    for index, line in enumerate(lines):
        fields = line.split()
        if len(fields) != 5:
            raise AssertionError("{}:{}: expected five fields".format(name, index))
        if int(fields[0]) != int(events["x"][index]):
            raise AssertionError("{}:{}: x changed".format(name, index))
        if int(fields[1]) != int(events["y"][index]):
            raise AssertionError("{}:{}: y changed".format(name, index))
        if abs(float(fields[2]) - float(events["t"][index])) > 1e-8:
            raise AssertionError("{}:{}: t changed".format(name, index))
        if int(fields[3]) != int(events["p"][index]):
            raise AssertionError("{}:{}: p changed".format(name, index))
        label = int(fields[4])
        if label not in (0, 1):
            raise AssertionError("{}:{}: non-binary label {}".format(name, index, label))
        positives += label
    return positives


def selected_indices(report_path: Path) -> Dict[str, List[int]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("schema") != "ev-uav-m243-test-actual-v1":
        raise AssertionError("unexpected M243 report schema")
    selected: Dict[str, List[int]] = {}
    for video in report.get("videos", []):
        name = str(video["name"]) + ".txt"
        indices = [int(index) for index in video["variants"]["empty_topk_4"]["event_indices"]]
        if indices:
            selected[name] = indices
    if sorted(selected) != ["test_022.txt", "test_023.txt"]:
        raise AssertionError("M244 must touch only test_022 and test_023")
    if sum(len(indices) for indices in selected.values()) != 8:
        raise AssertionError("M244 must restore exactly eight events")
    return selected


def write_archive(
    output: Path,
    names: Iterable[str],
    payload: Dict[str, bytes],
    reference: Optional[Path],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    reference_info: Dict[str, zipfile.ZipInfo] = {}
    if reference is not None and reference.is_file():
        with zipfile.ZipFile(reference) as archive:
            reference_info = {info.filename: info for info in archive.infolist()}
        if sorted(reference_info) != sorted(names):
            raise AssertionError("reference ZIP has a different file list")

    with zipfile.ZipFile(
        output,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as archive:
        for name in names:
            if reference_info:
                # Reuse the original member metadata when it is available. The
                # payload remains independently rebuilt from M233 plus M243.
                info = copy.copy(reference_info[name])
                archive.writestr(info, payload[name], compress_type=info.compress_type, compresslevel=6)
            else:
                archive.writestr(name, payload[name])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", required=True, type=Path, help="directory containing test_*.npz")
    parser.add_argument("--base", type=Path, default=ROOT / "artifacts" / "m233_base_submission.zip")
    parser.add_argument("--report", type=Path, default=ROOT / "artifacts" / "m243_test_actual.json")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "m244_empty_top4_raw_test.zip")
    parser.add_argument(
        "--reference",
        type=Path,
        default=ROOT / "artifacts" / "m244_reference_submission.zip",
        help="optional official M244 ZIP used only to preserve ZIP member metadata",
    )
    parser.add_argument("--strict-hash", action="store_true", help="fail if ZIP SHA-256 differs from the official artifact")
    args = parser.parse_args()

    test_root = args.test_root.resolve()
    base = args.base.resolve()
    report = args.report.resolve()
    output = args.output.resolve()
    reference = args.reference.resolve() if args.reference else None
    for path in (test_root, base, report):
        if not path.exists():
            raise FileNotFoundError(path)

    names = expected_names(test_root)
    selected = selected_indices(report)
    with zipfile.ZipFile(base) as archive:
        if sorted(archive.namelist()) != names:
            raise AssertionError("M233 base package does not match the public test file list")
        payload = {name: archive.read(name) for name in names}

    for name, indices in selected.items():
        lines = payload[name].splitlines(keepends=True)
        for index in indices:
            if not 0 <= index < len(lines):
                raise AssertionError("{}: selected index {} is out of range".format(name, index))
            lines[index] = set_label(lines[index])
        payload[name] = b"".join(lines)

    base_payload: Dict[str, bytes]
    with zipfile.ZipFile(base) as archive:
        base_payload = {name: archive.read(name) for name in names}
    for name in names:
        if name not in selected and payload[name] != base_payload[name]:
            raise AssertionError("{} changed outside the M244 scope".format(name))

    write_archive(output, names, payload, reference)
    with zipfile.ZipFile(output) as archive:
        if sorted(archive.namelist()) != names:
            raise AssertionError("unexpected output ZIP members")
        counts = {name: validate_member(name, archive.read(name), test_root) for name in names}

    digest = sha256(output)
    print("output:", output)
    print("sha256:", digest)
    print("official_sha256:", M244_SHA256)
    print("total_positive_events:", sum(counts.values()))
    print("selected_events:", sum(len(indices) for indices in selected.values()))
    print("test_022_positive:", counts["test_022.txt"])
    print("test_023_positive:", counts["test_023.txt"])
    if digest != M244_SHA256:
        message = "ZIP bytes differ from the canonical archive; event-level validation passed"
        if args.strict_hash:
            raise AssertionError(message)
        print("warning:", message, file=sys.stderr)


if __name__ == "__main__":
    main()
