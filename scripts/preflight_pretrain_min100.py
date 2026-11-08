"""Check the exact VQ/GPT-2/data identities before the stage-two minimal run."""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
import torch

ROOT=Path('/mnt/cfs-omni/lhy/neurolm-reproduction/pretrain-min100-20261108-r2')
VQ=Path('/mnt/cfs-omni/lhy/neurolm-reproduction/vq-fixed-e5-20261108-r1')


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--worker',action='store_true')
    args=parser.parse_args()
    runtime=json.loads((ROOT/'code/bundle_manifest.json').read_text())
    for name,expected in runtime.items():assert digest(ROOT/'code'/name)==expected,name
    subprocess.run([sys.executable,'-c','import train_pretrain; import pretrain_minimal'],cwd=ROOT/'code',check=True)
    done=json.loads((VQ/'run/completed.json').read_text());assert done['steps']==4485 and done['epochs_completed']==5
    assert all((VQ/f'node-{rank}/exit_code.txt').read_text().strip()=='0' for rank in [0,1])
    weights=ROOT/'gpt2/model.safetensors'
    assert digest(weights)=='248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707'
    gpt_verified=json.loads((ROOT/'gpt2/verified.json').read_text())
    for name,expected in gpt_verified['files'].items():assert digest(ROOT/'gpt2'/name)==expected
    validation=json.loads((ROOT/'probe/validation_manifest.json').read_text());assert validation['samples']==160
    for entry in validation['files']:assert digest(ROOT/'probe'/entry['name'])==entry['sha256']
    assert digest(VQ/'assets/train_files.txt')=='b3a2af72fd7a835286fd664fe9f791d4e31a4d434d04d15550aed8c42df18158'
    assert json.loads((ROOT/'cpu_test.json').read_text())['ok']
    checkpoint=VQ/'run/checkpoints/VQ/ckpt-4.pt'
    checkpoint_sha=digest(checkpoint)
    assert not (ROOT/'run/checkpoints').exists() and not (ROOT/'run/metrics.jsonl').exists()
    if args.worker:
        assert torch.cuda.device_count()==8 and torch.cuda.is_bf16_supported()
        assert all('H20' in torch.cuda.get_device_name(i) for i in range(8))
    report=dict(ok=True,worker=args.worker,runtime_files=len(runtime),vq_checkpoint_sha256=checkpoint_sha,
                vq_checkpoint_epoch=5,gpt2_revision=gpt_verified['revision'],train_samples=459656,validation_samples=160,
                torch_version=torch.__version__,gpu_names=[torch.cuda.get_device_name(i) for i in range(8)] if args.worker else [])
    path=ROOT/(f'preflight_node_{os.environ["RANK"]}.json' if args.worker else 'preflight.json')
    path.write_text(json.dumps(report,indent=2));print('PRETRAIN_PREFLIGHT='+json.dumps(report),flush=True)


if __name__=='__main__':main()
