"""Check the HMC v1.1 preprocessing split and output filename coverage."""

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


NAME = re.compile(r"(?P<subject>SN\d{3})-(?P<window>\d+)\.pkl")
EXPECTED_COUNTS = {"train": 91248, "eval": 22124, "test": 23871}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_recordings_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    raw_files = sorted(args.raw_recordings_dir.glob("SN???.edf"))
    if len(raw_files) != 151:
        raise ValueError(f"Expected 151 HMC recordings; found {len(raw_files)}")
    raw_subjects = [path.stem for path in raw_files]
    if len(set(raw_subjects)) != len(raw_subjects):
        raise ValueError("Duplicate HMC recording subject")

    split_subjects = {
        "train": raw_subjects[:100],
        "eval": raw_subjects[100:125],
        "test": raw_subjects[125:],
    }
    summary = {}
    for split, expected_subjects in split_subjects.items():
        files = list((args.output_dir / split).glob("*.pkl"))
        windows = defaultdict(set)
        for path in files:
            match = NAME.fullmatch(path.name)
            if not match:
                raise ValueError(f"Unexpected HMC output name: {path.name}")
            subject = match["subject"]
            window = int(match["window"])
            if window in windows[subject]:
                raise ValueError(f"Duplicate window: {path.name}")
            windows[subject].add(window)
        if set(windows) != set(expected_subjects):
            raise ValueError(f"Unexpected subjects in {split}")
        if len(files) != EXPECTED_COUNTS[split]:
            raise ValueError(f"Unexpected {split} window count: {len(files)}")
        for subject, indexes in windows.items():
            if indexes != set(range(len(indexes))):
                raise ValueError(f"Missing window index for {subject}")
        names = "\n".join(sorted(path.name for path in files)) + "\n"
        summary[split] = {
            "samples": len(files),
            "subjects": expected_subjects,
            "samples_per_subject": {subject: len(windows[subject]) for subject in expected_subjects},
            "filenames_sha256": hashlib.sha256(names.encode()).hexdigest(),
        }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
