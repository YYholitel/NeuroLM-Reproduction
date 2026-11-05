"""Check stair-stepping visibility in the author's PickleLoader on one sample."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-dir", required=True, type=Path)
    parser.add_argument("--sample", required=True, type=Path, help="Trusted local pretraining pickle")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    sys.path.insert(0, str(args.author_dir))
    from dataset import PickleLoader

    X, chans, times, input_mask, gpt_mask, C, N = PickleLoader(
        [str(args.sample)], GPT_training=True
    )[0]
    assert N >= 2 * C and int(input_mask.sum()) == N
    mask = gpt_mask[0]
    blocks = {
        "t1_to_t1": mask[:C, :C],
        "t2_to_t1": mask[C:2*C, :C],
        "t1_to_t2": mask[:C, C:2*C],
        "t2_to_t2": mask[C:2*C, C:2*C],
    }
    checks = {
        "t1_to_t1_all_true": bool(blocks["t1_to_t1"].all()),
        "t2_to_t1_all_true": bool(blocks["t2_to_t1"].all()),
        "t1_to_t2_all_false": bool((~blocks["t1_to_t2"]).all()),
        "t2_to_t2_all_true": bool(blocks["t2_to_t2"].all()),
        "padding_keys_all_false": bool((~mask[:, N:]).all()),
    }
    # Independent time-order specification; rows are queries, columns are keys.
    positions = torch.arange(mask.size(0))
    expected = (positions[:, None] // C >= positions[None, :] // C) & (positions[None, :] < N)
    checks["full_mask_matches_time_order"] = torch.equal(mask, expected)
    summary = {
        "num_chans": C,
        "valid_tokens": N,
        "mask_shape": list(gpt_mask.shape),
        "slice_shape": list(mask[:2*C, :2*C].shape),
        "block_shape": [C, C],
        "block_true_counts": {name: int(block.sum()) for name, block in blocks.items()},
        "checks": checks,
        "model_forward_executed": False,
        "sample_sha256": hashlib.sha256(args.sample.read_bytes()).hexdigest(),
        "loader_sha256": hashlib.sha256((args.author_dir / "dataset.py").read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    assert all(checks.values()), "EEG attention mask does not match time-step visibility"
    print("EEG_MASK_CHECK_OK")


if __name__ == "__main__":
    main()
