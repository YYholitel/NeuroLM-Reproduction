"""Check one real EEG sample through the author's pretraining loader and model."""

import argparse
import ast
import gc
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

EXPECTED_SHA256 = "ffe098bc138b89f8817d3710a3604498d8ecd15135080e2ca27735d05c6d29ab"
VQ_SHA256 = "e792c39a6a9e6d1bf4604cf63090730424f1d37f942597883d0c0a1375a2663a"


def sha256(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def load_target_tokenizer(path, device):
    """Use the same VQ construction and state-dict mapping as train_pretrain.py."""
    from model.model_neural_transformer import NTConfig
    from model.model_vq import VQ

    assert sha256(path) == VQ_SHA256, "VQ.pt SHA-256 mismatch"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    encoder = dict(n_layer=12, n_head=12, n_embd=768, block_size=1024,
                   bias=False, dropout=0., num_classes=0, in_chans=1, out_chans=16)
    decoder = dict(n_layer=4, n_head=12, n_embd=768, block_size=1024,
                   bias=False, dropout=0., num_classes=0, in_chans=128)
    for key in ["n_layer", "n_head", "n_embd", "block_size", "bias"]:
        encoder[key] = checkpoint["encoder_args"][key]
        decoder[key] = checkpoint["decoder_args"][key]
    tokenizer = VQ(NTConfig(**encoder), NTConfig(**decoder))
    weights = {k.removeprefix("_orig_mod."): v for k, v in checkpoint["model"].items()}
    tokenizer.load_state_dict({k[3:]: v for k, v in weights.items() if k.startswith("VQ.")}, strict=True)
    return tokenizer.requires_grad_(False).to(device).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--sample-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--vq-checkpoint", type=Path,
                        help="Generate real next-time-step EEG targets and return full logits plus loss.")
    args = parser.parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)

    # This script is for trusted, locally prepared pickle files only.
    sample_path = min(args.sample_dir.glob("*.pkl"))
    with sample_path.open("rb") as source:
        sample = pickle.load(source)
    raw_x = np.asarray(sample["X"])
    assert raw_x.ndim == 2 and np.isfinite(raw_x).all()
    adaptation = "none"
    loader_path = sample_path
    ch_names = sample.get("ch_names")
    if ch_names is None:
        # TUAB preprocessing enforces this order before writing {X, y} files.
        tree = ast.parse((args.author_dir / "dataset_maker/prepare_TUAB.py").read_text())
        order = next(
            ast.literal_eval(node.value)
            for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "chOrder_standard" for t in node.targets)
        )
        assert raw_x.shape[0] == len(order)
        ch_names = [name.removeprefix("EEG ").removesuffix("-REF") for name in order]
        loader_path = args.run_dir / "tuab_pretraining_adapter.pkl"
        with loader_path.open("wb") as output:
            pickle.dump({"X": raw_x, "ch_names": ch_names}, output)
        adaptation = "added_ch_names_from_author_TUAB_enforced_order; EEG_values_unchanged"
    assert len(ch_names) == raw_x.shape[0]
    assert raw_x.shape[1] % 200 == 0
    expected_tokens = raw_x.shape[0] * (raw_x.shape[1] // 200)
    assert 0 < expected_tokens <= 1024

    sys.path.insert(0, str(args.author_dir))
    from dataset import PickleLoader
    from model.model import GPTConfig
    from model.model_neurolm import NeuroLM

    loader = PickleLoader([str(loader_path)], GPT_training=True)
    X, input_chans, input_time, input_mask, gpt_mask, num_chans, num_tokens = loader[0]
    assert X.shape == (1024, 200) and gpt_mask.shape == (1, 1024, 1024)
    assert num_chans == raw_x.shape[0] and num_tokens == expected_tokens
    assert int(input_mask.sum()) == num_tokens
    assert not input_mask[num_tokens:].any()
    assert not gpt_mask[:, :, num_tokens:].any()
    assert not X[num_tokens:].any()
    assert torch.isfinite(X).all()
    # Verify that patch order is time-major, then channel, as in PickleLoader.
    expected = torch.tensor(raw_x / 100, dtype=torch.float32)
    expected = expected.reshape(num_chans, -1, 200).permute(1, 0, 2).reshape(-1, 200)
    assert torch.equal(X[:num_tokens], expected)
    summary = {
        "check": "PickleLoader_GPT_training_EEG_only_forward_no_training",
        "source_sample_sha256": sha256(sample_path),
        "source_keys": sorted(sample.keys()),
        "source_X_shape": list(raw_x.shape),
        "ch_names": list(ch_names),
        "adapter": adaptation,
        "loader_output": {
            "X": {"shape": list(X.shape), "dtype": str(X.dtype)},
            "input_chans": {"shape": list(input_chans.shape), "dtype": str(input_chans.dtype)},
            "input_time": {"shape": list(input_time.shape), "dtype": str(input_time.dtype)},
            "input_mask": {"shape": list(input_mask.shape), "dtype": str(input_mask.dtype)},
            "gpt_mask": {"shape": list(gpt_mask.shape), "dtype": str(gpt_mask.dtype)},
            "num_chans": num_chans,
            "data.size(0)": num_tokens,
        },
        "valid_tokens": int(input_mask.sum()),
        "padding_tokens": 1024 - num_tokens,
        "valid_time_range": [int(input_time[:num_tokens].min()), int(input_time[:num_tokens].max())],
    }
    print(json.dumps(summary, indent=2), flush=True)
    actual = sha256(args.checkpoint)
    assert actual == EXPECTED_SHA256, "NeuroLM-B.pt SHA-256 mismatch"
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = NeuroLM(GPTConfig(**checkpoint["model_args"]), init_from="scratch")
    weights = {k.removeprefix("_orig_mod."): v for k, v in checkpoint["model"].items()}
    model.load_state_dict(weights, strict=True)
    del checkpoint, weights
    gc.collect()
    model = model.to(args.device).eval()

    intermediates = {}

    def capture(name):
        def hook(module, inputs, output):
            intermediates[name] = {
                "shape": list(output.shape),
                "finite": bool(torch.isfinite(output).all().item()),
            }
        return hook

    handles = [
        model.tokenizer.register_forward_hook(capture("tokenizer_features")),
        model.GPT2.transformer.ln_f.register_forward_hook(capture("gpt_hidden_states")),
    ]
    X_batch = X.unsqueeze(0).to(args.device)
    chans_batch = input_chans.unsqueeze(0).to(args.device)
    times_batch = input_time.unsqueeze(0).to(args.device)
    mask_batch = input_mask.unsqueeze(0).to(args.device)
    Y_eeg = None
    target_tokenizer = load_target_tokenizer(args.vq_checkpoint, args.device) if args.vq_checkpoint else None
    with torch.inference_mode():
        if target_tokenizer is not None:
            assert num_tokens > num_chans, "Need at least two time steps for EEG prediction targets"
            ids = target_tokenizer.get_codebook_indices(X_batch, chans_batch, times_batch, mask_batch)
            vocab_offset = model.GPT2.config.vocab_size
            Y_eeg = torch.full((1, 1024), -1 - vocab_offset, dtype=torch.long, device=args.device)
            Y_eeg[0, :num_tokens - num_chans] = ids[0, num_chans:num_tokens]
            effective_targets = Y_eeg + vocab_offset
            assert ids.shape == (1, 1024)
            codebook_size = target_tokenizer.quantize.num_tokens
            assert int(ids.min()) >= 0 and int(ids.max()) < codebook_size
            assert int((effective_targets != -1).sum()) == num_tokens - num_chans
            summary["pretraining_targets"] = {
                "vq_checkpoint_sha256": VQ_SHA256,
                "codebook_indices_shape": list(ids.shape),
                "codebook_size": codebook_size,
                "Y_eeg_shape": list(Y_eeg.shape),
                "Y_eeg_dtype": str(Y_eeg.dtype),
                "shift_patches": num_chans,
                "supervised_positions": num_tokens - num_chans,
                "ignored_positions": 1024 - (num_tokens - num_chans),
                "vocab_offset": vocab_offset,
                "ignore_value_before_offset": -1 - vocab_offset,
                "ignore_value_in_cross_entropy": -1,
                "requires_grad": Y_eeg.requires_grad,
            }
        loss, log, logits = model(
            x_eeg=X_batch,
            y_eeg=Y_eeg,
            input_chans=chans_batch,
            input_time=times_batch,
            input_mask=mask_batch,
            eeg_mask=gpt_mask.unsqueeze(0).to(args.device),
        )
    for handle in handles:
        handle.remove()
    assert logits.shape == (1, 1024 if args.vq_checkpoint else 1, 58496)
    if args.vq_checkpoint:
        assert loss is not None and torch.isfinite(loss)
    assert torch.isfinite(logits).all() and all(v["finite"] for v in intermediates.values())
    summary.update({
        "checkpoint_sha256": actual,
        "parameters": sum(p.numel() for p in model.parameters()),
        "torch": torch.__version__,
        "device": args.device,
        "precision": "float32_without_autocast",
        "forward_intermediates": intermediates,
        "logits_shape": list(logits.shape),
        "logits_finite": True,
        "loss": None if loss is None else float(loss),
        "logits_position": "all_1024_positions_with_VQ_targets" if args.vq_checkpoint else "last_of_1024_positions_is_padding; upstream_no_target_forward_returns_last_position_only",
        "model_forward_calls": 1,
        "backward_or_optimizer_step": False,
        "num_chans_and_num_tokens_role": "loader_metadata_for_target_construction; not_model_forward_arguments",
        "source_code_sha256": {
            name: sha256(args.author_dir / name)
            for name in ["dataset.py", "model/model_neurolm.py", "model/model.py", "model/model_vq.py", "model/norm_ema_quantizer.py"]
        },
    })
    (args.run_dir / "eeg_path_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)
    print("EEG_PATH_SMOKE_OK", flush=True)


if __name__ == "__main__":
    main()
