"""Evaluate a repaired checkpoint on exactly the previous collapse audit's 32 samples.

This is a fixed training-sample diagnostic, not held-out validation. Evaluation
must leave every quantizer state tensor unchanged. No training is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch


def stats(counts):
    counts = counts.double()
    probabilities = counts[counts > 0] / counts.sum()
    return dict(tokens=int(counts.sum()), active_codes=int((counts > 0).sum()),
                perplexity=float(torch.exp(-(probabilities * probabilities.log()).sum())),
                top1_fraction=float(counts.max() / counts.sum()),
                top5_fraction=float(counts.topk(5).values.sum() / counts.sum()))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(8)
    root = args.root.resolve()
    manifest = json.loads((root / 'code/bundle_manifest.json').read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((root / 'code' / name).read_bytes()).hexdigest() == digest, name
    sys.path.insert(0, str(root / 'code'))
    from dataset import PickleLoader
    from model.model_vq import VQ
    from model.model_neural_transformer import NTConfig

    # Own, trusted training checkpoint, containing optimizer/NumPy state.
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert ckpt['loss_normalization'] == 'global_valid_elements'
    model = VQ(NTConfig(**ckpt['encoder_args']), NTConfig(**ckpt['decoder_args']),
               **ckpt['vq_options']).eval()
    state = {k.removeprefix('_orig_mod.'): v for k, v in ckpt['model'].items()}
    model.load_state_dict({k[3:]: v for k, v in state.items() if k.startswith('VQ.')}, strict=True)
    quantizer_before = {k: v.clone() for k, v in model.quantize.state_dict().items()}
    reference = json.loads(args.reference.read_text())
    names = reference['sample_names']
    assert len(names) == len(set(names)) == 32
    chosen = {}
    with (root / 'assets/train_files.txt').open() as stream:
        for line in stream:
            entry = json.loads(line)
            if entry['name'] in names:
                assert entry['name'] not in chosen
                chosen[entry['name']] = dict(entry, shard=str(root / 'assets' / entry['shard']))
    assert set(chosen) == set(names)
    loader = PickleLoader([chosen[name] for name in names])
    capacity = model.quantize.num_tokens
    counts = torch.zeros(capacity, dtype=torch.int64)
    by_dataset = {}
    indices, masks, losses = [], [], []
    with torch.inference_mode():
        for index in range(len(loader)):
            batch = [tensor.unsqueeze(0) for tensor in loader[index]]
            x, freq, raw, chans, times, mask = batch
            tokens = model.get_codebook_indices(x, chans, times, mask)
            _, _, log = model(*batch)
            assert ((tokens[mask] >= 0) & (tokens[mask] < capacity)).all()
            histogram = torch.bincount(tokens[mask], minlength=capacity)
            counts += histogram
            dataset = names[index].split('/')[1]
            by_dataset.setdefault(dataset, torch.zeros_like(counts)).add_(histogram)
            losses.append(dict(name=names[index], tokens=int(mask.sum()),
                               reconstruction=float(log['val/rec_freq_loss'] + log['val/rec_raw_loss']),
                               commitment=float(log['val/quant_loss'])))
            indices.append(tokens[0].numpy()); masks.append(mask[0].numpy())
            print(json.dumps(dict(audited=index + 1, total=32)), flush=True)
    assert all(torch.equal(value, model.quantize.state_dict()[name])
               for name, value in quantizer_before.items()), 'Evaluation mutated quantizer state'
    assert int(counts.sum()) == reference['valid']['tokens'] == 4944
    total_tokens = sum(item['tokens'] for item in losses)
    report = dict(checkpoint=str(args.checkpoint), checkpoint_epoch=ckpt['epoch'] + 1,
                  checkpoint_updates=ckpt['iter_num'], sample_names=names,
                  diagnostic_split='fixed train samples; not held-out validation',
                  matching_previous_samples=True, evaluation_quantizer_unchanged=True,
                  codebook_size=capacity, valid=stats(counts), counts=counts.tolist(),
                  previous_valid={k: reference['valid'][k] for k in
                                  ['tokens', 'active_codes', 'perplexity', 'top1_fraction', 'top5_fraction']},
                  per_dataset={name: stats(histogram) for name, histogram in by_dataset.items()},
                  reconstruction=sum(item['tokens'] * item['reconstruction'] for item in losses) / total_tokens,
                  commitment=sum(item['tokens'] * item['commitment'] for item in losses) / total_tokens,
                  sample_losses=losses)
    assert all(np.isfinite(item[key]) for item in losses for key in ['reconstruction', 'commitment'])
    args.output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output / 'token_indices.npz', indices=indices, valid_mask=masks)
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2))
    print('FIXED_AUDIT=' + json.dumps({k: v for k, v in report.items()
                                      if k not in ['sample_names', 'counts', 'sample_losses']}), flush=True)


if __name__ == '__main__':
    main()
