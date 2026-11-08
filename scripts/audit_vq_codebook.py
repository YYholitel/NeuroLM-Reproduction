"""Audit the completed VQ checkpoint on real EEG, without changing any checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch


def histogram(indices, size):
    counts = torch.bincount(indices.reshape(-1), minlength=size).double()
    total = int(counts.sum())
    active = int((counts > 0).sum())
    probabilities = counts[counts > 0] / max(total, 1)
    values, ids = counts.topk(min(10, size))
    return dict(tokens=total, active_codes=active,
                perplexity=float(torch.exp(-(probabilities * probabilities.log()).sum())),
                top1_fraction=float(values[0] / max(total, 1)),
                top5_fraction=float(values[:5].sum() / max(total, 1)),
                top10=[dict(code=int(i), count=int(n)) for i, n in zip(ids, values)],
                counts=counts.long().tolist())


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    sys.path.insert(0, str(root / 'code'))
    from dataset import PickleLoader
    from model.model_vq import VQ
    from model.model_neural_transformer import NTConfig

    manifest = json.loads((root / 'code/bundle_manifest.json').read_text())
    for name in ['model/norm_ema_quantizer.py', 'model/model_vq.py', 'dataset.py']:
        assert hashlib.sha256((root / 'code' / name).read_bytes()).hexdigest() == manifest[name]
    # These are our own run's checkpoints; optimizer state contains NumPy scalars.
    ckpt = torch.load(root / 'run/checkpoints/VQ/ckpt.pt', map_location='cpu', weights_only=False, mmap=True)
    model = VQ(NTConfig(**ckpt['encoder_args']), NTConfig(**ckpt['decoder_args'])).eval()
    model.load_state_dict({k[3:]: v for k, v in ckpt['model'].items() if k.startswith('VQ.')})
    quantizer = model.quantize
    weight = quantizer.embedding.weight.detach().clone()
    old = torch.load(root / 'run/checkpoints/VQ/ckpt-9.pt', map_location='cpu', weights_only=False, mmap=True)
    delta = (weight - old['model']['VQ.quantize.embedding.weight']).norm(dim=1)
    initial_bins = quantizer.embedding.cluster_size.detach().double()
    actual_ema = quantizer.cluster_size.detach().double()
    optimizer = model.configure_optimizers(1e-4, 5e-5, (0.9, 0.999), 'cpu')
    in_optimizer = any(p is quantizer.embedding.weight for group in optimizer.param_groups for p in group['params'])

    all_entries = [json.loads(line) for line in (root / 'assets/train_files.txt').read_text().splitlines()]
    sample_counts = {'TUAB': 8, 'TUEV': 8, 'HMC': 8, 'Workload': 4, 'TUSL': 4}
    rng = random.Random(20261108)
    entries = []
    datasets = []
    for name, count in sample_counts.items():
        candidates = [e for e in all_entries if e['name'].split('/')[1] == name]
        for entry in rng.sample(candidates, count):
            entries.append(dict(entry, shard=str(root / 'assets' / entry['shard'])))
            datasets.append(name)
    loader = PickleLoader(entries, GPT_training=False)
    indices = []
    masks = []
    for start in range(0, len(entries), 2):
        batch = [loader[i] for i in range(start, min(start + 2, len(entries)))]
        x, _, _, chans, times, mask = [torch.stack([sample[k] for sample in batch]) for k in range(6)]
        with torch.inference_mode():
            tokens = model.get_codebook_indices(x, chans, times, mask)
        indices.append(tokens.cpu())
        masks.append(mask)
        print(json.dumps({'audited_samples': min(start + 2, len(entries)), 'total': len(entries)}), flush=True)
    indices = torch.cat(indices)
    masks = torch.cat(masks)
    assert torch.equal(weight, quantizer.embedding.weight), 'Evaluation changed the codebook weights'
    np.savez_compressed(args.output / 'token_indices.npz', indices=indices.numpy(), valid_mask=masks.numpy())
    by_dataset = {}
    for name in sample_counts:
        selected = torch.tensor([d == name for d in datasets])
        subset = histogram(indices[selected][masks[selected]], quantizer.num_tokens)
        subset.pop('counts')
        by_dataset[name] = subset
    report = dict(checkpoint_epoch=ckpt['epoch'] + 1, checked_runtime_hashes=True,
                  beta=quantizer.beta, decay=quantizer.decay, declared_eps=quantizer.embedding.eps,
                  codebook_requires_grad=quantizer.embedding.weight.requires_grad,
                  codebook_in_optimizer=in_optimizer,
                  initial_rank0_kmeans_tokens=int(initial_bins.sum()),
                  initially_nonempty_clusters=int((initial_bins > 0).sum()),
                  initial_top32_cluster_fraction=float(initial_bins.topk(32).values.sum() / initial_bins.sum()),
                  actual_ema_total=float(actual_ema.sum()), actual_ema_top1_fraction=float(actual_ema.max() / actual_ema.sum()),
                  rows_changed_epoch10_to50=int((delta > 1e-6).sum()),
                  rows_unchanged_epoch10_to50=int((delta <= 1e-6).sum()),
                  unique_codebook_rows=int(torch.unique(weight, dim=0).size(0)),
                  codebook_l2_norm_range=[float(weight.norm(dim=1).min()), float(weight.norm(dim=1).max())],
                  sample_counts=sample_counts, batch_padding_fraction=1-float(masks.float().mean()),
                  valid=histogram(indices[masks], quantizer.num_tokens),
                  padding=histogram(indices[~masks], quantizer.num_tokens),
                  per_dataset=by_dataset,
                  first_sample=dict(name=entries[0]['name'], valid_token_indices=indices[0][masks[0]].tolist()),
                  sample_names=[e['name'] for e in entries],
                  evaluation_codebook_unchanged=True,
                  evaluation_usage_buffer_changed=not torch.equal(actual_ema, quantizer.cluster_size.double()))
    (args.output / 'audit.json').write_text(json.dumps(report, indent=2))
    compact = {k: v for k, v in report.items() if k not in ['valid', 'padding', 'sample_names', 'first_sample']}
    compact.update({k: {a: b for a, b in report[k].items() if a != 'counts'} for k in ['valid', 'padding']})
    compact['first_sample_first_32_valid_indices'] = report['first_sample']['valid_token_indices'][:32]
    print('AUDIT_RESULT=' + json.dumps(compact), flush=True)


if __name__ == '__main__':
    main()
