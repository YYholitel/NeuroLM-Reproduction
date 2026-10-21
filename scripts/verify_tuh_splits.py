"""Audit TUAB/TUEV output membership without publishing EEG or file names."""

import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


def sha(names):
    return hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()


def subject(stem):
    return stem.split("_", 1)[0]


def tuab_sources(raw, seed):
    np.random.seed(seed)
    result = {name: {} for name in ("train", "val", "test")}
    all_subjects = sorted({subject(p.stem) for category in ("abnormal", "normal")
                           for p in (raw / "train" / category / "01_tcp_ar").glob("*.edf")})
    np.random.shuffle(all_subjects)
    boundary = int(len(all_subjects) * 0.8)
    train_members = set(all_subjects[:boundary])
    val_members = set(all_subjects[boundary:])
    for label, category in ((1, "abnormal"), (0, "normal")):
        train = raw / "train" / category / "01_tcp_ar"
        test = raw / "eval" / category / "01_tcp_ar"
        groups = (("train", train, train_members),
                  ("val", train, val_members),
                  ("test", test, {subject(p.stem) for p in test.glob("*.edf")}))
        for split, directory, selected in groups:
            for path in directory.glob("*.edf"):
                if subject(path.stem) in selected:
                    assert path.stem not in result[split]
                    result[split][path.stem] = label
    return result


def tuev_sources(raw):
    train = {p.stem for p in (raw / "train").rglob("*.edf")}
    test = {p.stem for p in (raw / "eval").rglob("*.edf")}
    assert not train & test
    return {"train": train, "val": train, "test": test}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", required=True, choices=("tuab", "tuev"))
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=4523)
    args = parser.parse_args()
    source = tuab_sources(args.raw_dir, args.seed) if args.kind == "tuab" else tuev_sources(args.raw_dir)
    folders = {"train": "train", "val": "val", "test": "test"} if args.kind == "tuab" else {"train": "processed_train", "val": "processed_eval", "test": "processed_test"}
    result = {"kind": args.kind, "seed": args.seed if args.kind == "tuab" else 4523, "splits": {}}
    seen = {}
    for split, folder in folders.items():
        files = sorted((args.out_dir / folder).glob("*.pkl"))
        stems = [p.stem.rsplit("_" if args.kind == "tuab" else "-", 1)[0] for p in files]
        assert all(p.stem.rsplit("_" if args.kind == "tuab" else "-", 1)[-1].isdigit() for p in files)
        allowed = set(source[split])
        assert set(stems) <= allowed, f"{split}: output outside expected input"
        seen[split] = set(stems)
        people = {subject(name) for name in stems}
        subset = files[:3] + files[-3:]
        for path in subset:
            with path.open("rb") as handle:
                item = pickle.load(handle)
            if args.kind == "tuab":
                assert item["X"].shape == (23, 2000)
                assert int(item["y"]) == source[split][path.stem.rsplit("_", 1)[0]]
            else:
                assert item["signal"].shape[0] == 23
                assert "label" in item
        row = {"input_recordings": len(allowed), "output_recordings": len(seen[split]),
               "samples": len(files), "output_subjects": len(people),
               "output_stem_sha256": sha(seen[split]), "output_subject_sha256": sha(people)}
        if args.kind == "tuab":
            missing = allowed - seen[split]
            row["skipped_recordings"] = len(missing)
            row["skipped_stem_sha256"] = sha(missing)
            row["labels_from_source"] = dict(sorted(Counter(str(source[split][name]) for name in stems).items()))
        result["splits"][split] = row
    people = {split: {subject(name) for name in stems} for split, stems in seen.items()}
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        assert not (people[left] & people[right]), f"patient overlap: {left}/{right}"
    result["patient_overlap_all_splits"] = 0
    if args.kind == "tuev":
        result["train_input_recordings"] = len(source["train"])
        result["test_input_recordings"] = len(source["test"])
        result["unrepresented_train_recordings"] = len(source["train"] - seen["train"] - seen["val"])
        result["unrepresented_test_recordings"] = len(source["test"] - seen["test"])
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
