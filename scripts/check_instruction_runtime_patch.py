"""CPU regression checks for the patched evaluation's uneven distributed shards."""
import argparse
import ast
from contextlib import nullcontext
import json
from pathlib import Path
import tempfile

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


class Samples(torch.utils.data.Dataset):
    def __init__(self, count):
        self.count = count

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        x = torch.tensor([index])
        return x, x, index % 2, x, x, x, x


class Generator(torch.nn.Module):
    def generate(self, eeg, text, *args, **kwargs):
        return torch.cat((torch.zeros_like(text), eeg.long()), dim=1)


def decoded(tokens):
    return 'invalid' if tokens[0] == 2 else ('Yes' if tokens[0] % 2 else 'No')


def metric(preds, labels, metrics, is_binary):
    return {'accuracy': float((preds == labels).mean())}


def worker(rank, source, rendezvous, output):
    dist.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)
    try:
        tree = ast.parse(Path(source).read_text(encoding='utf-8'))
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)
                     and node.name in ('evaluate', 'get_pred')]
        ns = dict(torch=torch, np=np, Path=Path, json=json, ddp=True, ddp_rank=rank,
                  ddp_world_size=2, device='cpu', ctx=nullcontext(),
                  all_gather_object=dist.all_gather_object, get_metrics=metric)
        exec(compile(ast.Module(body=functions, type_ignores=[]), source, 'exec'), ns)
        for count in (5, 1):
            ds = Samples(count)
            loader = torch.utils.data.DataLoader(ds, batch_size=2, sampler=range(rank, count, 2))
            info = dict(name='TUAB', result_idx=0, label_dic={'Yes': 1, 'No': 0},
                        is_binary=True, metrics=['accuracy'])
            result = ns['evaluate'](Generator(), info, loader, decoded, Path(output) / str(count), 0, 'test')
            assert result['samples'] == count
            assert result['parse_failures'] == (1 if count == 5 else 0)
            assert result['accuracy'] == (0.8 if count == 5 else 1.0)
            if rank == 0:
                rows = []
                for file in (Path(output) / str(count) / 'predictions').glob('*.jsonl'):
                    rows.extend(json.loads(line) for line in file.read_text().splitlines())
                assert sorted(row['index'] for row in rows) == list(range(count))
        # Match the author's two forwards and one combined backward under DDP.
        class TwoBranches(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.shared = torch.nn.Linear(2, 1)
                self.eeg_position = torch.nn.Parameter(torch.ones(1))

            def forward(self, x, eeg):
                value = self.shared(x)
                return value + self.eeg_position if eeg else value

        model = torch.nn.parallel.DistributedDataParallel(TwoBranches())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        for _ in range(2):
            x = torch.ones(2, 2) * (rank + 1)
            loss1 = model(x, True).square().mean()
            loss2 = model(x, False).square().mean()
            (loss1 + loss2).backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        print(f'PASS rank {rank}: uneven/empty evaluation shards and combined DDP backward', flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--author-file', type=Path, required=True)
    args = parser.parse_args()
    work = Path.cwd() / '.local'
    work.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='neurolm-patch-check-', dir=work) as folder:
        rendezvous = (Path(folder) / 'rendezvous').as_uri()
        mp.spawn(worker, args=(str(args.author_file.resolve()), rendezvous, folder), nprocs=2, join=True)
