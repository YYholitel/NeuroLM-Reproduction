"""Isolate native LM-head crashes by batch size in separate GPU processes."""
import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import subprocess
import sys


def one_case(batch):
    import torch
    import sympy  # Match the runtime loaded by the author optimizer.
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = True
    layer = torch.nn.Linear(768, 58496, bias=False).cuda().eval()
    x = torch.randn(batch, 1, 768, device='cuda')
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        for _ in range(5):
            y = layer(x)
            torch.cuda.synchronize()
            assert torch.isfinite(y).all()
    print(json.dumps({'batch': batch, 'shape': list(y.shape), 'finite': True,
                      'torch': torch.__version__, 'cuda': torch.version.cuda,
                      'gpu': torch.cuda.get_device_name()}), flush=True)


def main(output):
    assert os.environ['WORLD_SIZE'] == '1' and os.environ['RANK'] == '0'
    assert os.environ['GPU_NUM_PER_NODE'] == '2'
    free = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.free', '--format=csv,noheader,nounits'], text=True)
    assert len(free.splitlines()) == 2
    assert all('H20' in line and int(line.split(',')[-1]) >= 90000 for line in free.splitlines())
    pids = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True)
    assert not pids.strip(), 'Existing GPU jobs'
    output.mkdir(exist_ok=False)
    def group(gpu, batches):
        results = []
        for batch in batches:
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONFAULTHANDLER='1')
            result = subprocess.run([sys.executable, '-X', 'faulthandler', __file__, '--batch', str(batch)],
                                    env=env, capture_output=True, text=True, timeout=120)
            (output / f'batch-{batch}.log').write_text(result.stdout + result.stderr)
            row = {'batch': batch, 'gpu': gpu, 'exit_code': result.returncode,
                   'gmp_overflow': 'gmp: overflow in mpz type' in result.stdout + result.stderr}
            if result.returncode == 0:
                row['result'] = json.loads(result.stdout)
            print(json.dumps(row), flush=True)
            results.append(row)
        return results
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(group, 0, [7, 1, 16])
        b = pool.submit(group, 1, [48, 8, 33])
        results = sorted(a.result() + b.result(), key=lambda x: x['batch'])
    (output / 'summary.json').write_text(json.dumps({'diagnostic_only': True, 'cases': results}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--batch', type=int)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    one_case(args.batch) if args.batch is not None else main(args.output)
