"""Record size, token count and SHA-256 of completed OpenWebText bins."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def file_sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args()
    data = {"dataset": "Skylion007/openwebtext", "revision": "79d93d786212f7344586290adb811d4ae6a1762c", "source_rows": 8013769, "dtype": "uint16", "split_seed": 2357, "splits": {}}
    for split in ("train", "val"):
        path = args.text_dir / f"{split}.bin"
        size = path.stat().st_size
        assert size > 0 and size % 2 == 0, (path, size)
        values = np.memmap(path, dtype=np.uint16, mode="r")
        assert int(values[-1]) == 50256, "last document lacks GPT-2 end token"
        data["splits"][split] = {"bytes": size, "tokens": len(values), "sha256": file_sha256(path)}
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
