"""Verify the isolated A/B runtime, frozen asset manifest and requested hardware."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import torch

ROOT=Path('/mnt/cfs-omni/lhy/neurolm-reproduction/vq-compact-e5-20261108-r1')


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    manifest=json.loads((ROOT/'code/bundle_manifest.json').read_text())
    for name,digest in manifest.items():
        assert hashlib.sha256((ROOT/'code'/name).read_bytes()).hexdigest()==digest,name
    ready=json.loads((ROOT/'assets/data_ready.json').read_text())
    assert ready['ready'] and ready['total_samples']==459656 and not ready['diagnostic']
    entries=(ROOT/'assets/train_files.txt').read_bytes()
    assert len(entries.splitlines())==459656
    out=ROOT/'run'
    assert not (out/'checkpoints/VQ/ckpt.pt').exists()
    assert not (out/'metrics.jsonl').exists()
    assert (out/'text/train.bin').stat().st_size==18071164978
    assert (out/'text/val.bin').stat().st_size==8869212
    assert json.loads((ROOT/'cpu_test.json').read_text())['ok']
    if args.worker:
        assert torch.cuda.device_count()==8
        assert all('H20' in torch.cuda.get_device_name(i) for i in range(8))
    report=dict(ok=True,runtime_files=len(manifest),train_samples=459656,
                train_manifest_sha256=hashlib.sha256(entries).hexdigest(),
                plan_sha256=ready['plan_sha256'],torch_version=torch.__version__,
                worker=args.worker,worker_rank=os.environ.get('RANK'),
                gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if args.worker else [])
    path=ROOT/(f'preflight_node_{os.environ["RANK"]}.json' if args.worker else 'preflight.json')
    path.write_text(json.dumps(report,indent=2))
    print('VQ_E5_PREFLIGHT='+json.dumps(report),flush=True)


if __name__=='__main__': main()
