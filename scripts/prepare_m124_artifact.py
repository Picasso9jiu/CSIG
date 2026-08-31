#!/usr/bin/env python3
"""Create a threshold-specific copy of the frozen M124 verifier artifact."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import pickle
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    destination = args.destination.resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists() and not args.force:
        raise FileExistsError("refusing to overwrite {}; pass --force".format(destination))

    with gzip.open(source, "rb") as stream:
        payload = pickle.load(stream)
    if not isinstance(payload, dict):
        raise TypeError("M124 artifact payload must be a mapping")
    if "verifier_threshold" not in payload or "schema" not in payload:
        raise ValueError("not a supported M124 verifier artifact")

    output = dict(payload)
    output["verifier_threshold"] = float(args.threshold)
    output["release_override"] = {
        "source": source.name,
        "source_sha256": sha256(source),
        "threshold": float(args.threshold),
        "uses_test_labels": False,
        "uses_video_name": False,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with gzip.open(temporary, "wb", compresslevel=1) as stream:
        pickle.dump(output, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(destination)

    with gzip.open(destination, "rb") as stream:
        check = pickle.load(stream)
    if abs(float(check["verifier_threshold"]) - float(args.threshold)) > 1e-12:
        raise AssertionError("threshold did not persist")
    print("source_sha256:", sha256(source))
    print("output:", destination)
    print("output_sha256:", sha256(destination))
    print("threshold:", check["verifier_threshold"])


if __name__ == "__main__":
    main()
