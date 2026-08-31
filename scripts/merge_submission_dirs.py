#!/usr/bin/env python3
"""Merge non-overlapping prediction directories and validate their file list."""

from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path


def test_names(test_root: Path) -> list[str]:
    names = sorted(path.with_suffix(".txt").name for path in test_root.glob("test_*.npz"))
    if len(names) != 31:
        raise AssertionError("expected 31 test NPZ files, found {}".format(len(names)))
    return names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", required=True, type=Path)
    parser.add_argument("--input-dir", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--zip", dest="zip_path", required=True, type=Path)
    args = parser.parse_args()

    expected = test_names(args.test_root.resolve())
    output_dir = args.output_dir.resolve()
    zip_path = args.zip_path.resolve()
    if output_dir.exists():
        raise FileExistsError("refusing to overwrite {}".format(output_dir))
    if zip_path.exists():
        raise FileExistsError("refusing to overwrite {}".format(zip_path))

    sources: dict[str, Path] = {}
    for directory in args.input_dir:
        directory = directory.resolve()
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for path in directory.glob("test_*.txt"):
            if path.name in sources:
                raise AssertionError("duplicate prediction {}".format(path.name))
            sources[path.name] = path
    if sorted(sources) != expected:
        missing = sorted(set(expected) - set(sources))
        unexpected = sorted(set(sources) - set(expected))
        raise AssertionError("input file mismatch; missing={}, unexpected={}".format(missing, unexpected))

    output_dir.mkdir(parents=True)
    for name in expected:
        shutil.copy2(sources[name], output_dir / name)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for name in expected:
            archive.write(output_dir / name, arcname=name)
    print("merged_txt_files:", len(expected))
    print("output_dir:", output_dir)
    print("zip:", zip_path)


if __name__ == "__main__":
    main()
