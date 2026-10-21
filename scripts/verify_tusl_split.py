"""Audit the fixed author's TUSL file split without publishing EEG contents."""

import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path


def digest(names):
    return hashlib.sha256("\n".join(names).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()

    edf = sorted(args.raw_dir.rglob("*.edf"))
    assert len({p.stem for p in edf}) == len(edf), "EDF stems are not unique"
    n = len(edf)
    groups = {
        "train": edf[: int(n * 0.6)],
        "test": edf[int(n * 0.6) : int(n * 0.8)],
        "eval": edf[int(n * 0.8) :],
    }
    manifest = {"raw_edf": n, "splits": {}}
    for split, source in groups.items():
        samples = sorted((args.out_dir / split).glob("*.pkl"))
        expected = {p.stem for p in source}
        stems = {p.stem.rsplit("-", 1)[0] for p in samples}
        assert stems <= expected, f"{split}: unexpected output stems {stems - expected}"
        assert all(p.stem.rsplit("-", 1)[-1].isdigit() for p in samples)
        skipped = sorted(expected - stems)
        labels = Counter()
        shapes = Counter()
        for path in samples:
            with path.open("rb") as stream:
                sample = pickle.load(stream)
            labels[str(int(sample["y"][0]))] += 1
            shapes[str(tuple(sample["X"].shape))] += 1
        assert sum(labels.values()) == len(samples)
        assert list(shapes) == ["(23, 2000)"], f"{split}: unexpected shapes {shapes}"
        manifest["splits"][split] = {
            "input_recordings": len(source),
            "output_recordings": len(stems),
            "samples": len(samples),
            "skipped_recordings": len(skipped),
            "input_stem_sha256": digest(sorted(expected)),
            "skipped_stem_sha256": digest(skipped),
            "labels": dict(sorted(labels.items())),
            "shapes": dict(shapes),
        }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: {x: v[x] for x in ("input_recordings", "output_recordings", "samples", "skipped_recordings")} for k, v in manifest["splits"].items()}, indent=2))


if __name__ == "__main__":
    main()
