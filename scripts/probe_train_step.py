"""Probe one NeuroLM-B instruction training step without claiming an experiment result.

Uses a real preprocessed Workload example and deterministic synthetic text.
No checkpoint is written. The reported loss is only a runtime diagnostic.
"""

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

EXPECTED_SHA256 = "ffe098bc138b89f8817d3710a3604498d8ecd15135080e2ca27735d05c6d29ab"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--workload-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--eeg-batch", type=int, default=1)
    parser.add_argument("--text-batch", type=int, default=1)
    args = parser.parse_args()
    if args.eeg_batch < 1 or args.text_batch < 1:
        parser.error("batch sizes must be positive")

    with args.checkpoint.open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    if actual != EXPECTED_SHA256:
        raise ValueError("NeuroLM-B.pt SHA-256 mismatch")

    sys.path.insert(0, str(args.author_dir))
    from downstream_dataset import WorkloadLoader  # noqa: E402
    from model.model import GPTConfig  # noqa: E402
    from model.model_neurolm import NeuroLM  # noqa: E402

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = NeuroLM(GPTConfig(**checkpoint["model_args"]), init_from="scratch")
    model.load_state_dict(
        {key.removeprefix("_orig_mod."): value for key, value in checkpoint["model"].items()},
        strict=True,
    )
    del checkpoint
    gc.collect()

    model = model.to(args.device).train()
    optimizer = model.configure_optimizers(0.1, 5e-4, (0.9, 0.95), "cuda")
    use_bf16 = torch.cuda.is_bf16_supported()
    dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=not use_bf16)

    sample_name = "Subject00_1-0.pkl"
    dataset = WorkloadLoader(
        str(args.workload_dir / "train"), [sample_name],
        is_instruct=True, is_val=False, eeg_max_len=276, text_max_len=80,
    )
    x_eeg, x_text, y_text, chans, times, mask, gpt_mask = dataset[0]
    def repeated(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.unsqueeze(0).repeat(args.eeg_batch, *([1] * tensor.ndim)).to(args.device)

    x_eeg = repeated(x_eeg)
    x_text = repeated(x_text)
    y_text = repeated(y_text)
    chans = repeated(chans)
    times = repeated(times)
    mask = repeated(mask)
    gpt_mask = repeated(gpt_mask)
    y_eeg = torch.full(
        (args.eeg_batch, x_eeg.size(1)), -1 - model.GPT2.config.vocab_size,
        dtype=torch.long, device=args.device,
    )
    text_tokens = torch.arange(1025, dtype=torch.long, device=args.device).unsqueeze(0).repeat(args.text_batch, 1)
    text_input = text_tokens[:, :-1].contiguous()
    text_target = text_tokens[:, 1:].contiguous()

    torch.cuda.reset_peak_memory_stats(args.device)
    start = time.monotonic()
    with torch.autocast(device_type="cuda", dtype=dtype):
        instruction_loss, _, _ = model(
            x_eeg, y_eeg, x_text, y_text, chans, times, mask,
            eeg_text_mask=gpt_mask,
        )
        text_loss, _, _ = model(x_text=text_input, y_text=text_target)
        loss = instruction_loss + text_loss
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize(args.device)

    print(json.dumps({
        "purpose": "runtime_and_memory_probe_only",
        "checkpoint_sha256": actual,
        "sample": sample_name,
        "repeats_single_eeg_sample": True,
        "eeg_batch": args.eeg_batch,
        "text_batch": args.text_batch,
        "text_length": text_input.size(1),
        "dtype": str(dtype),
        "instruction_loss_diagnostic": instruction_loss.item(),
        "synthetic_text_loss_diagnostic": text_loss.item(),
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated(args.device) / 2**30, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved(args.device) / 2**30, 3),
        "seconds": round(time.monotonic() - start, 3),
        "torch": torch.__version__,
    }, indent=2))


if __name__ == "__main__":
    main()
