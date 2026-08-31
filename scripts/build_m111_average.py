#!/usr/bin/env python3
"""Build the fixed M111 phase-specialist checkpoint from three seed models."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", type=Path, default=None)
    args = parser.parse_args()
    if len(args.checkpoint) != 3:
        raise ValueError("M111 requires exactly three seed checkpoints")
    paths = [path.resolve() for path in args.checkpoint]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    payloads = [torch.load(path, map_location="cpu") for path in paths]
    if not all(isinstance(payload, dict) and "model_state_dict" in payload for payload in payloads):
        raise ValueError("all checkpoints must contain model_state_dict")
    states = [payload["model_state_dict"] for payload in payloads]
    keys = list(states[0])
    if any(list(state) != keys for state in states[1:]):
        raise AssertionError("model-state key order differs between seed checkpoints")

    averaged = {}
    for key in keys:
        tensors = [state[key] for state in states]
        first = tensors[0]
        if not torch.is_tensor(first) or any(not torch.is_tensor(value) or value.shape != first.shape for value in tensors[1:]):
            raise AssertionError("{} has incompatible tensors".format(key))
        if torch.is_floating_point(first):
            averaged[key] = torch.stack([value.to(dtype=torch.float64) for value in tensors], dim=0).mean(dim=0).to(dtype=first.dtype)
        else:
            if any(not torch.equal(first, value) for value in tensors[1:]):
                raise AssertionError("{} is non-floating and differs between seeds".format(key))
            averaged[key] = first.clone()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError("refusing to overwrite {}".format(output))
    result = dict(payloads[0])
    result["model_state_dict"] = averaged
    result["m111_phase_specialist_average"] = {
        "method": "equal arithmetic mean in float64, cast to source dtype",
        "source_seeds": [72, 73, 76],
        "sources": [
            {"path": str(path), "sha256": sha256(path)} for path in paths
        ],
        "non_floating_policy": "require exact equality",
        "metadata_source_seed": 72,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(result, output)
    digest = sha256(output)
    print("output:", output)
    print("sha256:", digest)
    if args.reference:
        reference_path = args.reference.resolve()
        reference = torch.load(reference_path, map_location="cpu")
        reference_state = reference.get("model_state_dict") if isinstance(reference, dict) else None
        if not isinstance(reference_state, dict) or list(reference_state) != keys:
            raise AssertionError("reference checkpoint has incompatible model_state_dict")
        if any(not torch.equal(averaged[key], reference_state[key]) for key in keys):
            raise AssertionError("averaged model tensors differ from the reference checkpoint")
        print("reference_model_state: ok")
        print("reference_sha256:", sha256(reference_path))


if __name__ == "__main__":
    main()
