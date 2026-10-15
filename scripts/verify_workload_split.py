"""Check the author Workload preprocessing output counts and subject split."""

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

NAME = re.compile(r"Subject(?P<subject>\d{2})_[12]-(?P<window>\d+)\.pkl")
EXPECTED = {
    "train": (set(range(26)), 1508),
    "eval": (set(range(26, 31)), 290),
    "test": (set(range(31, 36)), 290),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    summary = {}
    for split, (expected_subjects, expected_count) in EXPECTED.items():
        files = list((args.output_dir / split).glob("*.pkl"))
        subjects = Counter()
        recordings = Counter()
        for path in files:
            match = NAME.fullmatch(path.name)
            if not match:
                raise ValueError(f"Unexpected file name: {path.name}")
            subject = int(match["subject"])
            subjects[subject] += 1
            recordings[path.name.split("-")[0]] += 1
        if len(files) != expected_count or set(subjects) != expected_subjects:
            raise ValueError(f"Unexpected {split} count or subject split")
        if set(recordings.values()) != {29}:
            raise ValueError(f"Each {split} recording must yield 29 windows")
        names = "\n".join(sorted(path.name for path in files)) + "\n"
        summary[split] = {
            "samples": len(files),
            "subjects": sorted(subjects),
            "filenames_sha256": hashlib.sha256(names.encode()).hexdigest(),
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
