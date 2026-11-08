"""Inference-only TUSL/TUAB evaluation with strict parsing and token-trie decoding.

Loads an entire NeuroLM checkpoint strictly. An optional VQ encoder override is
an explicitly labelled, unaligned encoder-swap diagnostic, never retraining.
No text train.bin, optimizer, backward call, or train_vq.py is used.
"""
import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import sys

from checkpoint_eval_io import clean_state, sha256, tensor_digest, write_json
from downstream_eval_core import build_candidates, classification_metrics, generate_constrained, record_prediction

EXPECTED = {'TUAB': {'val': 73627, 'test': 36945}, 'TUSL': {'val': 66, 'test': 60}}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--author-dir', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--expected-checkpoint-sha256')
    p.add_argument('--vq-encoder-override', type=Path)
    p.add_argument('--dataset-dir', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--tasks', nargs='+', choices=list(EXPECTED), default=['TUSL', 'TUAB'])
    p.add_argument('--splits', nargs='+', choices=['val', 'test'], default=['val', 'test'])
    p.add_argument('--mode', choices=['constrained', 'author_greedy_strict_parser'], default='constrained')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--num-workers', type=int, default=2)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--device', default='cpu')
    p.add_argument('--precision', choices=['float32', 'bfloat16'], default='float32')
    p.add_argument('--limit', type=int, help='Smoke subset only; never reported as a full split')
    args = p.parse_args()
    if args.limit is not None and args.limit < 1:
        p.error('--limit must be positive')
    if len(set(args.tasks)) != len(args.tasks) or len(set(args.splits)) != len(args.splits):
        p.error('Duplicate task or split')
    import torch
    import torch.distributed as dist
    import tiktoken
    torch.set_num_threads(args.threads)
    torch.manual_seed(42)
    rank, world = int(os.getenv('RANK', 0)), int(os.getenv('WORLD_SIZE', 1))
    if args.device.startswith('cuda'):
        torch.cuda.set_device(int(os.getenv('LOCAL_RANK', 0)))
        device = torch.device('cuda', int(os.getenv('LOCAL_RANK', 0)))
    else:
        device = torch.device(args.device)
    if world > 1:
        dist.init_process_group('nccl' if device.type == 'cuda' else 'gloo')
    if rank == 0:
        args.out_dir.mkdir(parents=True, exist_ok=False)
    if world > 1:
        dist.barrier()
    sys.path.insert(0, str(args.author_dir.resolve()))
    from downstream_dataset import TUABLoader, TUSLLoader
    from model.model import GPTConfig
    from model.model_neurolm import NeuroLM
    checkpoint_hash = sha256(args.checkpoint)
    if args.expected_checkpoint_sha256 and checkpoint_hash != args.expected_checkpoint_sha256:
        raise ValueError('NeuroLM checkpoint SHA-256 does not match the declared input')
    checkpoint = torch.load(str(args.checkpoint), map_location='cpu', weights_only=False, mmap=True)
    model = NeuroLM(GPTConfig(**checkpoint['model_args']), init_from='scratch')
    model.load_state_dict(clean_state(checkpoint['model']), strict=True)
    provenance = {'neurolm_checkpoint_sha256': checkpoint_hash,
                  'checkpoint_epoch_zero_based': checkpoint.get('epoch'),
                  'checkpoint_iter_num': checkpoint.get('iter_num'),
                  'tokenizer_source': 'embedded_in_neurolm_checkpoint'}
    del checkpoint
    gc.collect()
    if args.vq_encoder_override:
        override_hash = sha256(args.vq_encoder_override)
        override = torch.load(str(args.vq_encoder_override), map_location='cpu', weights_only=False, mmap=True)
        state = clean_state(override['model'])
        prefix = 'VQ.encoder.' if any(k.startswith('VQ.encoder.') for k in state) else 'encoder.'
        encoder = {key[len(prefix):]: value for key, value in state.items() if key.startswith(prefix)}
        if not encoder:
            raise ValueError('No encoder found in VQ override')
        model.tokenizer.load_state_dict(encoder, strict=True)
        provenance.update(tokenizer_source='unaligned_vq_encoder_swap_diagnostic',
                          vq_override_sha256=override_hash,
                          vq_override_epoch_zero_based=override.get('epoch'))
        del override, state, encoder
        gc.collect()
    provenance['embedded_encoder_tensor_sha256'] = {
        name: tensor_digest(value) for name, value in model.tokenizer.state_dict().items()}
    model = model.to(device).eval()
    enc = tiktoken.get_encoding('gpt2')
    reports = []
    for task in args.tasks:
        for split in args.splits:
            root = args.dataset_dir / ('TUAB/processed' if task == 'TUAB' else 'TUSL')
            root = root / ('eval' if task == 'TUSL' and split == 'val' else split)
            with os.scandir(root) as entries:
                # Do not stat every CFS entry before reading it. Every selected
                # path must subsequently load successfully; errors are fatal.
                files = sorted(entry.name for entry in entries if entry.name.endswith(('.pkl', '.pickle')))
            if len(files) != EXPECTED[task][split]:
                raise ValueError(f'{task}/{split}: expected {EXPECTED[task][split]} files, got {len(files)}')
            selected = files[:args.limit] if args.limit else files
            loader_cls = TUABLoader if task == 'TUAB' else TUSLLoader
            dataset = loader_cls(str(root), selected, is_instruct=True, is_val=True, eeg_max_len=276, text_max_len=80)
            candidates, answers = build_candidates(dataset, enc.decode)
            indices = list(range(rank, len(dataset), world))
            dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=indices,
                                                     num_workers=args.num_workers, drop_last=False)
            cursor = 0
            pred_path = args.out_dir / f'{task}-{split}-rank{rank}.jsonl'
            fail_path = args.out_dir / f'{task}-{split}-rank{rank}-invalid.jsonl'
            with pred_path.open('x', encoding='utf-8') as output, fail_path.open('x', encoding='utf-8') as failures:
                for batch in dataloader:
                    x, text, labels, chans, times, valid, mask = batch
                    x, text, chans, times, valid, mask = [t.to(device) for t in (x, text, chans, times, valid, mask)]
                    context = torch.autocast(device.type, dtype=torch.bfloat16) if args.precision == 'bfloat16' else nullcontext()
                    with torch.inference_mode(), context:
                        if args.mode == 'constrained':
                            generated, probabilities = generate_constrained(model, x.float(), text, chans, times, valid, mask, candidates)
                        else:
                            generated = model.generate(x.float(), text, chans, times, valid, eeg_text_mask=mask, max_new_tokens=5)
                            probabilities = [None] * len(labels)
                    for i, tokens in enumerate(generated):
                        record_prediction(output, failures, index=indices[cursor], label=labels[i],
                                          text=enc.decode(tokens[1:].tolist()), answers=answers,
                                          probabilities=probabilities[i], context={'task': task, 'split': split, 'mode': args.mode})
                        cursor += 1
                    if cursor % (args.batch_size * 20) == 0:
                        print(json.dumps({'task': task, 'split': split, 'rank': rank, 'processed': cursor, 'assigned': len(indices)}), flush=True)
            if cursor != len(indices):
                raise AssertionError('Rank lost evaluation samples')
            if world > 1:
                dist.barrier()
            if rank == 0:
                rows, artifacts = {}, {}
                for worker in range(world):
                    path = args.out_dir / f'{task}-{split}-rank{worker}.jsonl'
                    artifacts[path.name] = sha256(path)
                    for line in path.read_text(encoding='utf-8').splitlines():
                        row = json.loads(line)
                        if row['index'] in rows:
                            raise AssertionError('Duplicate prediction index')
                        rows[row['index']] = row
                if set(rows) != set(range(len(dataset))):
                    raise AssertionError('Incomplete split coverage')
                rows = [rows[i] for i in range(len(dataset))]
                metrics = classification_metrics([r['label'] for r in rows], [r['prediction'] for r in rows], len(candidates))
                metrics['roc_auc'] = None
                metrics['roc_auc_note'] = 'Unavailable without valid scores; never computed from generated hard labels or a filtered subset.'
                if task == 'TUAB' and all('conditional_first_token_probabilities' in r for r in rows) and len(set(r['label'] for r in rows)) == 2:
                    from sklearn.metrics import roc_auc_score
                    metrics['roc_auc'] = float(roc_auc_score([r['label'] for r in rows], [r['conditional_first_token_probabilities'][1] for r in rows]))
                    metrics['roc_auc_note'] = 'Conditional probability of Yes among legal first tokens; modified constrained-decoding protocol.'
                report = {'task': task, 'split': split, 'full_split': len(selected) == len(files),
                          'available_samples': len(files), 'selected_file_list_sha256': hashlib.sha256('\n'.join(selected).encode()).hexdigest(),
                          'metrics': metrics, 'candidate_token_ids': candidates, 'answers': answers,
                          'prediction_sha256': artifacts}
                reports.append(report)
                write_json(args.out_dir / f'{task}-{split}-metrics.json', report)
                print('SPLIT_RESULT=' + json.dumps(report, allow_nan=False), flush=True)
            if world > 1:
                dist.barrier()
    if sha256(args.checkpoint) != checkpoint_hash:
        raise RuntimeError('Checkpoint changed during evaluation')
    if args.vq_encoder_override and sha256(args.vq_encoder_override) != override_hash:
        raise RuntimeError('VQ override changed during evaluation')
    if rank == 0:
        write_json(args.out_dir / 'summary.json', {
            'status': 'completed', 'training_performed': False, 'mode': args.mode,
            'paper_comparable': False, 'provenance': provenance,
            'precision': args.precision, 'world_size': world, 'batch_size_per_rank': args.batch_size,
            'torch': torch.__version__, 'device': str(device), 'results': reports,
            'constrained_attention': 'preserve padded EEG key mask on appended queries',
            'code_sha256': {name: sha256(args.author_dir / name) for name in ['downstream_dataset.py', 'model/model_neurolm.py', 'model/model.py']},
            'script_sha256': {name: sha256(Path(__file__).with_name(name)) for name in ['eval_downstream.py', 'downstream_eval_core.py', 'checkpoint_eval_io.py']},
        })
    if world > 1:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
