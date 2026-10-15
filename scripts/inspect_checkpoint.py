"""Verify an official NeuroLM checkpoint before reading its tensor metadata."""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import torch


EXPECTED_SHA256 = {
    "NeuroLM-B.pt": "ffe098bc138b89f8817d3710a3604498d8ecd15135080e2ca27735d05c6d29ab",
    "VQ.pt": "e792c39a6a9e6d1bf4604cf63090730424f1d37f942597883d0c0a1375a2663a",
}


def inspect(path: Path, trusted_pickle: bool = False) -> dict:
    expected = EXPECTED_SHA256[path.name]
    with path.open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    if actual != expected:
        raise ValueError(f"SHA-256 mismatch for {path.name}: {actual}")

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except pickle.UnpicklingError:
        if not trusted_pickle:
            raise
        # PyTorch 2.2.2 cannot allowlist the NumPy scalar in this verified file.
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a dictionary in {path.name}")
    model = checkpoint.get("model")
    return {
        "file": path.name,
        "bytes": path.stat().st_size,
        "sha256": actual,
        "checkpoint_keys": sorted(checkpoint),
        "model_tensor_count": len(model) if isinstance(model, dict) else None,
        "model_args": checkpoint.get("model_args"),
        "encoder_args": checkpoint.get("encoder_args"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--trusted-pickle", action="store_true")
    args = parser.parse_args()
    if args.checkpoint.name not in EXPECTED_SHA256:
        parser.error("checkpoint must be NeuroLM-B.pt or VQ.pt")
    print(json.dumps(inspect(args.checkpoint, args.trusted_pickle), indent=2, default=str))


if __name__ == "__main__":
    main()
