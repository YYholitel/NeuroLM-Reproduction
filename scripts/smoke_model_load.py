"""Load NeuroLM-B and run text-only plus optional Workload EEG smoke passes."""

import argparse
import gc
import hashlib
import json
import sys
from pathlib import Path

import torch

EXPECTED_SHA256 = "ffe098bc138b89f8817d3710a3604498d8ecd15135080e2ca27735d05c6d29ab"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--author-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workload-dir", type=Path)
    args = parser.parse_args()

    with args.checkpoint.open("rb") as source:
        actual = hashlib.file_digest(source, "sha256").hexdigest()
    if actual != EXPECTED_SHA256:
        raise ValueError("NeuroLM-B.pt SHA-256 mismatch")

    sys.path.insert(0, str(args.author_dir))
    from model.model import GPTConfig  # noqa: E402
    from model.model_neurolm import NeuroLM  # noqa: E402

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = NeuroLM(GPTConfig(**checkpoint["model_args"]), init_from="scratch")
    state_dict = {
        key.removeprefix("_orig_mod."): value
        for key, value in checkpoint["model"].items()
    }
    model.load_state_dict(state_dict, strict=True)
    del checkpoint, state_dict
    gc.collect()

    model = model.to(args.device).eval()
    tokens = torch.zeros((1, 8), dtype=torch.long, device=args.device)
    with torch.inference_mode():
        _, _, logits = model(x_text=tokens)
    summary = {
        "checkpoint_sha256": actual,
        "parameters": sum(param.numel() for param in model.parameters()),
        "input_shape": list(tokens.shape),
        "logits_shape": list(logits.shape),
        "device": args.device,
        "torch": torch.__version__,
    }

    if args.workload_dir:
        import tiktoken  # noqa: E402
        from downstream_dataset import WorkloadLoader  # noqa: E402
        from train_instruction import get_pred  # noqa: E402

        test_dir = args.workload_dir / "test"
        names = ["Subject31_1-0.pkl", "Subject31_2-0.pkl"]
        dataset = WorkloadLoader(
            str(test_dir), names, is_instruct=True, is_val=True,
            eeg_max_len=276, text_max_len=80,
        )
        decoder = tiktoken.get_encoding("gpt2")
        task = {"name": "Workload", "result_idx": 9, "label_dic": {"Yes": 1, "No": 0}}
        samples = []
        for index, name in enumerate(names):
            x_eeg, x_text, label, chans, times, mask, gpt_mask = dataset[index]
            with torch.inference_mode():
                generated = model.generate(
                    x_eeg.unsqueeze(0).to(args.device),
                    x_text.unsqueeze(0).to(args.device),
                    chans.unsqueeze(0).to(args.device),
                    times.unsqueeze(0).to(args.device),
                    mask.unsqueeze(0).to(args.device),
                    eeg_text_mask=gpt_mask.unsqueeze(0).to(args.device),
                    max_new_tokens=5,
                    top_k=1,
                )
            decoded = decoder.decode(generated[0, 1:].tolist())
            samples.append({
                "file": name,
                "label": label,
                "prediction": get_pred(decoded, task),
                "generated_text": decoded,
            })
        summary["workload_smoke_samples"] = samples

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
