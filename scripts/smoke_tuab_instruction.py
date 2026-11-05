"""One real TUAB instruction forward; verify answer-only cross-entropy."""

import argparse
import gc
import hashlib
import json
import pickle
import sys
from pathlib import Path

import tiktoken
import torch
import torch.nn.functional as F

CHECKPOINT_SHA256 = "ffe098bc138b89f8817d3710a3604498d8ecd15135080e2ca27735d05c6d29ab"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--author-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--sample-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    sys.path.insert(0, str(args.author_dir))
    from downstream_dataset import TUABLoader
    from model.model import GPTConfig
    from model.model_neurolm import NeuroLM

    sample_path = min(args.sample_dir.glob("*.pkl"))
    with sample_path.open("rb") as source:
        sample = pickle.load(source)  # trusted, locally preprocessed TUAB
    loader = TUABLoader(str(args.sample_dir), [sample_path.name],
                        is_instruct=True, is_val=False, eeg_max_len=276, text_max_len=80)
    X_eeg, X_text, Y_text, chans, times, input_mask, gpt_mask = loader[0]
    eeg_len, text_len = X_eeg.size(0), X_text.size(0)
    total_len = eeg_len + text_len
    valid_eeg = int(input_mask.sum())
    valid_text = loader.text[int(sample["y"])].numel()
    prompt_len = loader.prompt.numel()
    assert valid_eeg == sample["X"].shape[0] * (sample["X"].shape[1] // 200)
    assert X_text[0] == 50257
    assert torch.equal(X_text[:prompt_len], loader.prompt)
    assert gpt_mask.shape == (1, total_len, total_len)
    assert not gpt_mask[:, :, valid_eeg:eeg_len].any()
    text_scored = torch.where(Y_text != -1)[0]
    assert text_scored.tolist() == list(range(prompt_len - 1, valid_text - 1))
    assert torch.equal(Y_text[text_scored], X_text[text_scored + 1])
    assert bool((Y_text[valid_text - 1:] == -1).all())
    assert X_text[valid_text - 1] == 50256
    enc = tiktoken.get_encoding("gpt2")

    actual = digest(args.checkpoint)
    assert actual == CHECKPOINT_SHA256
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = NeuroLM(GPTConfig(**checkpoint["model_args"]), init_from="scratch")
    model.load_state_dict({k.removeprefix("_orig_mod."): v for k, v in checkpoint["model"].items()}, strict=True)
    del checkpoint
    gc.collect()
    model = model.to(args.device).eval()
    assert total_len <= model.GPT2.config.block_size
    batch = [t.unsqueeze(0).to(args.device) for t in
             [X_eeg, X_text, Y_text, chans, times, input_mask, gpt_mask]]
    x_eeg, x_text, y_text, input_chans, input_time, mask, eeg_text_mask = batch
    offset = model.GPT2.config.vocab_size
    y_eeg = torch.full((1, eeg_len), -1 - offset, dtype=torch.long, device=args.device)
    targets = torch.cat((y_eeg + offset, y_text), dim=-1)
    scored = targets != -1
    assert not scored[:, :eeg_len].any()
    assert int(scored.sum()) == valid_text - prompt_len

    shapes = {}

    def capture(name):
        def hook(module, inputs, output):
            shapes[name] = list(output.shape)
            assert torch.isfinite(output).all()
        return hook

    handles = [model.tokenizer.register_forward_hook(capture("eeg_encoder")),
               model.GPT2.transformer.ln_f.register_forward_hook(capture("gpt_hidden"))]
    with torch.inference_mode():
        loss, log, logits = model(
            x_eeg=x_eeg, y_eeg=y_eeg, x_text=x_text, y_text=y_text,
            input_chans=input_chans, input_time=input_time, input_mask=mask,
            eeg_text_mask=eeg_text_mask,
        )
        assert logits.shape == (1, total_len, 58496)
        assert torch.isfinite(logits).all() and torch.isfinite(loss)
        answer_only_loss = F.cross_entropy(logits[scored], targets[scored])
        per_position = F.cross_entropy(logits.reshape(-1, logits.size(-1)),
                                      targets.reshape(-1), ignore_index=-1, reduction="none").view_as(targets)
        assert (per_position[~scored] == 0).all()
        assert torch.allclose(loss, answer_only_loss, atol=1e-6, rtol=1e-6)
    for handle in handles:
        handle.remove()

    result = {
        "check": "TUAB_single_instruction_batch_forward_and_answer_loss",
        "source_sample_sha256": digest(sample_path), "checkpoint_sha256": actual,
        "batch_size": 1, "model_mode": "eval", "precision": "float32_without_autocast",
        "source_X_shape": list(sample["X"].shape),
        "eeg_input_shape": list(x_eeg.shape), "text_input_shape": list(x_text.shape),
        "eeg_text_mask_shape": list(eeg_text_mask.shape),
        "valid_eeg_tokens": valid_eeg, "eeg_padding_tokens": eeg_len - valid_eeg,
        "valid_text_tokens_including_sep": valid_text, "prompt_tokens_including_sep": prompt_len,
        "text_padding_tokens": text_len - valid_text,
        "total_sequence_length": total_len, "nonpadding_sequence_length": valid_eeg + valid_text,
        "sep_global_index": eeg_len, "sep_token_id": int(X_text[0]),
        "vocab_offset": offset, "Y_eeg_fill_value": -1 - offset,
        "targets_shape": list(targets.shape), "nonignored_target_count": int(scored.sum()),
        "supervised_global_logit_positions": torch.where(scored[0])[0].tolist(),
        "supervised_global_source_token_positions": (text_scored + 1 + eeg_len).tolist(),
        "supervised_token_ids": Y_text[text_scored].tolist(),
        "supervised_token_pieces": [enc.decode([int(i)]) for i in Y_text[text_scored]],
        "decoded_supervised_text": enc.decode(Y_text[text_scored].tolist()),
        "loss": float(loss), "answer_only_recomputed_loss": float(answer_only_loss),
        "loss_absolute_difference": float((loss - answer_only_loss).abs()),
        "ignored_position_loss_all_zero": True, "logits_finite": True,
        "logits_shape": list(logits.shape), "forward_intermediate_shapes": shapes,
        "model_forward_calls": 1, "backward_or_optimizer_step": False,
        "external_vq_target_model_loaded": False,
        "source_code_sha256": {name: digest(args.author_dir / name) for name in
                               ["downstream_dataset.py", "model/model_neurolm.py", "model/model.py"]},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    print("TUAB_INSTRUCTION_FORWARD_OK", flush=True)


if __name__ == "__main__":
    main()
