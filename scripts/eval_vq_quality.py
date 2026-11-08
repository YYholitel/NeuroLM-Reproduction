"""Measure VQ utilization/perplexity on a frozen batch, without training.

Use only trusted project pickle/checkpoint files. An independently initialized
codebook has arbitrary IDs: utilization alone is not downstream model quality.
"""
import argparse
import json
from pathlib import Path
import sys

from checkpoint_eval_io import clean_state, sha256, tensor_digest, write_json
from downstream_eval_core import entropy_statistics


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--author-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--batch', type=Path, help='Frozen tensor batch produced with --save-batch')
    inputs.add_argument('--sample-manifest', type=Path, help='JSON list of trusted pretraining pickle paths or indexed-tar entries, in fixed order')
    parser.add_argument('--save-batch', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--indices-output', type=Path)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--block-size', type=int, default=1024)
    args = parser.parse_args()
    for path in [args.output, args.indices_output, args.save_batch]:
        if path is not None and path.exists():
            raise FileExistsError(path)
    import torch
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    sys.path.insert(0, str(args.author_dir.resolve()))
    from dataset import PickleLoader
    from model.model_vq import VQ
    from model.model_neural_transformer import NTConfig
    if args.batch:
        batch = torch.load(args.batch, map_location='cpu', weights_only=False)
        input_hash = sha256(args.batch)
        source = 'frozen_tensor_batch'
    else:
        entries = json.loads(args.sample_manifest.read_text(encoding='utf-8'))
        if not isinstance(entries, list) or not entries:
            raise ValueError('Manifest must be a nonempty fixed list')
        loader = PickleLoader(entries, block_size=args.block_size, GPT_training=False)
        samples = [loader[i] for i in range(len(loader))]
        batch = {name: torch.stack([sample[idx] for sample in samples]) for name, idx in
                 [('x', 0), ('input_chans', 3), ('input_time', 4), ('valid_mask', 5)]}
        input_hash = sha256(args.sample_manifest)
        source = 'fixed_manifest_via_author_PickleLoader'
    names = ['x', 'input_chans', 'input_time', 'valid_mask']
    if set(batch) != set(names):
        raise ValueError('Frozen batch must contain exactly ' + ', '.join(names))
    x, chans, times, mask = (batch[name] for name in names)
    if (x.ndim != 3 or x.shape[-1] != 200 or x.shape[:2] != mask.shape
            or chans.shape != mask.shape or times.shape != mask.shape
            or mask.dtype != torch.bool or not mask.any(1).all() or not torch.isfinite(x).all()):
        raise ValueError('Invalid fixed EEG batch')
    tensor_hashes = {name: tensor_digest(batch[name]) for name in names}
    if args.save_batch:
        args.save_batch.parent.mkdir(parents=True, exist_ok=True)
        with args.save_batch.open('xb') as stream:
            torch.save(batch, stream)
    checkpoint_hash = sha256(args.checkpoint)
    checkpoint = torch.load(str(args.checkpoint), map_location='cpu', weights_only=False, mmap=True)
    state = clean_state(checkpoint['model'])
    if any(key.startswith('VQ.') for key in state):
        state = {key[3:]: value for key, value in state.items() if key.startswith('VQ.')}
    codebook = state['quantize.embedding.weight']
    size, width = codebook.shape
    options = dict(checkpoint.get('vq_options', {}))
    options.update(n_embed=size, embed_dim=width)
    model = VQ(NTConfig(**checkpoint['encoder_args']), NTConfig(**checkpoint['decoder_args']), **options)
    model.load_state_dict(state, strict=True)
    if hasattr(model.quantize.embedding, 'initted') and not bool(model.quantize.embedding.initted):
        raise ValueError('Uninitialized codebook: refusing evaluation-time k-means')
    model = model.to(args.device).eval()
    before = {name: tensor_digest(tensor) for name, tensor in model.named_parameters()}
    buffers_before = {name: tensor_digest(tensor) for name, tensor in model.named_buffers()}
    before_codebook = tensor_digest(model.quantize.embedding.weight)
    with torch.inference_mode():
        indices = model.get_codebook_indices(*(batch[name].to(args.device) for name in names)).cpu()
    after = {name: tensor_digest(tensor) for name, tensor in model.named_parameters()}
    if before != after or before_codebook != tensor_digest(model.quantize.embedding.weight):
        raise AssertionError('Evaluation changed parameters/codebook')
    report = {
        'status': 'completed', 'checkpoint_sha256': checkpoint_hash,
        'checkpoint_epoch_zero_based': checkpoint.get('epoch'),
        'input_kind': source, 'input_source_sha256': input_hash,
        'input_tensor_sha256': tensor_hashes, 'input_shape': list(x.shape),
        'precision': 'float32_no_autocast', 'device': args.device,
        'torch_version': torch.__version__, 'evaluation_parameters_unchanged': True,
        'changed_runtime_buffers': [name for name, tensor in model.named_buffers()
                                    if buffers_before.get(name) != tensor_digest(tensor)],
        'statistics': entropy_statistics(indices, mask, size),
        'per_sample': [entropy_statistics(ids[None], valid[None], size)
                       for ids, valid in zip(indices, mask)],
        'code_sha256': {name: sha256(args.author_dir / name) for name in
                        ['dataset.py', 'model/model_vq.py', 'model/norm_ema_quantizer.py']},
        'script_sha256': sha256(__file__),
        'interpretation': 'Valid EEG tokens only; exp(-sum(p*ln(p))). Fixed-batch utilization is not accuracy or proof of codebook health.',
    }
    for sample in report['per_sample']:
        sample.pop('histogram')
    if args.indices_output:
        args.indices_output.parent.mkdir(parents=True, exist_ok=True)
        with args.indices_output.open('xb') as stream:
            torch.save({'indices': indices, 'valid_mask': mask}, stream)
    if sha256(args.checkpoint) != checkpoint_hash:
        raise RuntimeError('Checkpoint changed while evaluating; use an immutable snapshot')
    write_json(args.output, report)
    print(json.dumps({key: value for key, value in report['statistics'].items() if key != 'histogram'}, allow_nan=False))


if __name__ == '__main__':
    main()
