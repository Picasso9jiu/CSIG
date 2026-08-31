#!/usr/bin/env python3
"""Validate an EV-UAV Challenge 2 submission ZIP against public event files."""

from __future__ import annotations

import argparse
import hashlib
import zipfile
from pathlib import Path

import numpy as np


M244_SHA256 = "390fa26a200bb80f4729318011484621eb998269db616b313b7f022620f60e20"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def names_from_test_root(test_root: Path) -> list[str]:
    names = sorted(path.with_suffix(".txt").name for path in test_root.glob("test_*.npz"))
    if len(names) != 31:
        raise AssertionError("expected 31 test NPZ files, found {}".format(len(names)))
    return names


def validate_member(name: str, data: bytes, test_root: Path) -> tuple[int, int]:
    with np.load(test_root / (Path(name).stem + ".npz"), allow_pickle=False) as payload:
        events = payload["ev"]
    lines = data.splitlines()
    if len(lines) != len(events):
        raise AssertionError("{}: line count {} != {}".format(name, len(lines), len(events)))
    positives = 0
    for index, line in enumerate(lines):
        fields = line.split()
        if len(fields) != 5:
            raise AssertionError("{}:{}: expected five fields".format(name, index))
        if int(fields[0]) != int(events["x"][index]):
            raise AssertionError("{}:{}: x order differs".format(name, index))
        if int(fields[1]) != int(events["y"][index]):
            raise AssertionError("{}:{}: y order differs".format(name, index))
        if abs(float(fields[2]) - float(events["t"][index])) > 1e-8:
            raise AssertionError("{}:{}: t order differs".format(name, index))
        if int(fields[3]) != int(events["p"][index]):
            raise AssertionError("{}:{}: p order differs".format(name, index))
        label = int(fields[4])
        if label not in (0, 1):
            raise AssertionError("{}:{}: invalid label {}".format(name, index, label))
        positives += label
    return len(events), positives


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", dest="zip_path", required=True, type=Path)
    parser.add_argument("--test-root", required=True, type=Path)
    parser.add_argument("--expected-sha256", default="", help="optional lowercase SHA-256 assertion")
    parser.add_argument("--expected-positive", type=int, default=None)
    parser.add_argument("--m244", action="store_true", help="assert the released M244 fingerprint")
    args = parser.parse_args()

    zip_path = args.zip_path.resolve()
    test_root = args.test_root.resolve()
    if not zip_path.is_file():
        raise FileNotFoundError(zip_path)
    if not test_root.is_dir():
        raise FileNotFoundError(test_root)
    names = names_from_test_root(test_root)

    with zipfile.ZipFile(zip_path) as archive:
        if sorted(archive.namelist()) != names:
            raise AssertionError("ZIP member list differs from public test NPZ files")
        stats = [validate_member(name, archive.read(name), test_root) for name in names]
    total_events = sum(event_count for event_count, _ in stats)
    total_positive = sum(positive for _, positive in stats)
    digest = sha256(zip_path)

    expected_sha = args.expected_sha256.lower().strip()
    expected_positive = args.expected_positive
    if args.m244:
        expected_sha = M244_SHA256
        expected_positive = 87706
    if expected_sha and digest != expected_sha:
        raise AssertionError("SHA-256 mismatch: {} != {}".format(digest, expected_sha))
    if expected_positive is not None and total_positive != expected_positive:
        raise AssertionError("positive count {} != {}".format(total_positive, expected_positive))

    print("valid_txt_files:", len(names))
    print("total_events:", total_events)
    print("total_positive_events:", total_positive)
    print("sha256:", digest)
    if args.m244:
        print("m244_fingerprint: ok")


if __name__ == "__main__":
    main()
