"""Strict official-checkpoint initialization, with frozen aligned VQ targets."""
import hashlib
from pathlib import Path
import torch
from model.model import GPTConfig
from model.model_neurolm import NeuroLM
from model.model_neural_transformer import NTConfig
from model.model_vq import VQ

HASHES = {'NeuroLM-B.pt': 'ffe098bc138b89f8817d3710a3604498d8ecd15135080e2ca27735d05c6d29ab',
          'VQ.pt': 'e792c39a6a9e6d1bf4604cf63090730424f1d37f942597883d0c0a1375a2663a'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_official(neurolm_path, vq_path):
    # These exact files are checked before submission and on each worker node.
    saved = torch.load(str(vq_path), map_location='cpu', weights_only=False, mmap=True)
    state = {k.removeprefix('_orig_mod.'): v for k, v in saved['model'].items()}
    state = {k[3:]: v for k, v in state.items() if k.startswith('VQ.')}
    shape = list(state['quantize.embedding.weight'].shape)
    assert shape == [8192, 128]
    target = VQ(NTConfig(**saved['encoder_args']), NTConfig(**saved['decoder_args']),
                n_embed=shape[0], embed_dim=shape[1], dead_code_patience=0).eval()
    # Only training bookkeeping introduced by the previous runtime is absent in
    # the author's old checkpoint. No missing learned parameter is permitted.
    counters = {'quantize.code_age', 'quantize.num_updates', 'quantize.total_replacements'}
    missing = set(target.state_dict()) - set(state)
    assert missing == counters and not (set(state) - set(target.state_dict()))
    for key in sorted(counters):
        state[key] = torch.zeros_like(target.state_dict()[key])
    target.load_state_dict(state, strict=True)
    target.requires_grad_(False)
    assert bool(target.quantize.embedding.initted)
    saved = torch.load(str(neurolm_path), map_location='cpu', weights_only=False, mmap=True)
    state = {k.removeprefix('_orig_mod.'): v for k, v in saved['model'].items()}
    network = NeuroLM(GPTConfig(**saved['model_args']), init_from='scratch', eeg_vocab_size=shape[0])
    network.load_state_dict(state, strict=True)
    # 'scratch' constructs the shape only; every tensor is then overwritten.
    assert all(torch.equal(value.cpu(), state[key]) for key, value in network.state_dict().items())
    a, b = network.tokenizer.state_dict(), target.encoder.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a)
    assert all(not p.requires_grad for p in network.tokenizer.parameters())
    return network, target, dict(source='author released NeuroLM-B.pt', continued_pretraining=True,
        initial_neurolm_path=str(neurolm_path), tokenizer_path=str(vq_path), expected_sha256=HASHES,
        strict_full_checkpoint_load=True, fresh_optimizer=True, official_encoder_target_exact_match=True,
        codebook_shape=shape, decoder_layers=saved.get('decoder_args', {}).get('n_layer', 4),
        initialized_nonlearned_counters=sorted(counters))
