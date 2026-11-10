"""Gate stage three on its seed-matched audited stage-two checkpoint."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch


def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--job',type=Path,required=True);p.add_argument('--worker',action='store_true')
    a=p.parse_args();j=a.job;r=j.parent.parent;torch.set_num_threads(4)
    audit=json.loads((j.parent/'stage2/checkpoint_audit.json').read_text());assert audit['ok'] and audit['epochs_completed']==20
    ckpt=Path(audit['checkpoint_path']);assert sha(ckpt)==audit['checkpoint_sha256']
    for name,h in json.loads((r/'author/bundle_manifest.json').read_text()).items():assert sha(r/'author'/name)==h,name
    if a.worker:
        assert torch.__version__.split('+')[0]=='2.2.2' and torch.cuda.device_count()==2 and torch.cuda.is_bf16_supported()
        assert all('H20' in torch.cuda.get_device_name(i) for i in range(2))
        assert json.loads((j/'preflight.json').read_text())['ok']
        print('WORKER_PREFLIGHT='+json.dumps(dict(ok=True,gpus=2,torch=torch.__version__)),flush=True);return
    sys.path.insert(0,str(r/'author'))
    from model.model import GPTConfig
    from model.model_neurolm import NeuroLM
    saved=torch.load(str(ckpt),map_location='cpu',mmap=True,weights_only=False)
    model=NeuroLM(GPTConfig(**saved['model_args']),init_from='scratch')
    model.load_state_dict(saved['model'],strict=True)
    assert all(not x.requires_grad for x in model.tokenizer.parameters())
    counts=json.loads((r/'assets/data_ready.json').read_text())['datasets']
    assert counts['TUSL']==174 and sum(counts.values())==459711
    from torch.utils.data import DistributedSampler
    sampler=DistributedSampler(range(459711),num_replicas=2,rank=0,drop_last=True,seed=int(j.parent.name.split('-')[1]))
    assert len(sampler)//32==459711//64==7182
    result=dict(ok=True,initial_checkpoint=str(ckpt),initial_checkpoint_sha256=audit['checkpoint_sha256'],
                frozen_encoder=True,strict_load=True,epochs=5,counts=counts,microsteps_per_epoch=7182,
                sampler_drop_last=True,seed=int(j.parent.name.split('-')[1]))
    (j/'preflight.json').write_text(json.dumps(result,indent=2));print('STAGE3_PREFLIGHT='+json.dumps(result),flush=True)


if __name__=='__main__':main()
