"""Verify immutable assets and exact model identity before full pretraining."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import torch

ROOT=Path('/mnt/cfs-omni/lhy/neurolm-reproduction/pretrain-full20-20261109-r1')
VQ=ROOT.parent/'vq-fixed-e5-20261108-r1/run/checkpoints/VQ/ckpt-4.pt'
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--worker',action='store_true');a=p.parse_args()
    runtime=json.loads((ROOT/'code/bundle_manifest.json').read_text())
    source=json.loads((ROOT/'source_manifest.json').read_text())
    for name,expected in runtime.items():assert sha(ROOT/'code'/name)==expected,name
    assert set(runtime)==set(source)
    assert [name for name in runtime if runtime[name]!=source[name]]==['pretrain_minimal.py']
    subprocess.run([sys.executable,'-c','import train_pretrain; import pretrain_minimal'],cwd=ROOT/'code',check=True)
    assert sha(VQ)=='546e07e0b455b15ef9374a96bc77dead226bba8d00afa6e738a625fbb266de07'
    assert sha(ROOT/'gpt2/model.safetensors')=='248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707'
    assert sha(ROOT/'assets/train_files.txt')=='b3a2af72fd7a835286fd664fe9f791d4e31a4d434d04d15550aed8c42df18158'
    ready=json.loads((ROOT/'assets/data_ready.json').read_text())
    assert ready['ready'] and not ready['diagnostic'] and ready['total_samples']==459656
    entries=[json.loads(line) for line in (ROOT/'assets/train_files.txt').read_text().splitlines()]
    assert len(entries)==459656
    for shard in {entry['shard'] for entry in entries}:assert (ROOT/'assets'/shard).is_file(),shard
    assert (ROOT/'assets/text/train.bin').stat().st_size==18071164978
    assert json.loads((ROOT/'cpu_test.json').read_text())['ok']
    assert not (ROOT/'run/checkpoints').exists() and not (ROOT/'run/metrics.jsonl').exists()
    if a.worker:
        assert torch.cuda.device_count()==8 and torch.cuda.is_bf16_supported()
        assert all('H20' in torch.cuda.get_device_name(i) for i in range(8))
    report=dict(ok=True,worker=a.worker,runtime_files=len(runtime),model_files_and_train_pretrain_unchanged=True,
                vq_sha256=sha(VQ),frozen_vq_epoch=5,train_samples=len(entries),total_epochs=20,total_updates=19140,
                global_eeg_batch=480,global_text_batch=32,downstream_data_accessed=False,
                gpu_names=[torch.cuda.get_device_name(i) for i in range(8)] if a.worker else [])
    path=ROOT/(f'preflight_node_{os.environ["RANK"]}.json' if a.worker else 'preflight.json')
    path.write_text(json.dumps(report,indent=2));print('PRETRAIN_PREFLIGHT='+json.dumps(report),flush=True)
if __name__=='__main__':main()
