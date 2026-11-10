"""Independently check the final epoch, optimizer count, finiteness and frozen encoder."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def digest(state,prefix):
    h=hashlib.sha256();items=sorted((k[len(prefix):],v) for k,v in state.items() if k.startswith(prefix));assert items
    for name,v in items:
        x=v.detach().cpu().contiguous().numpy();h.update(name.encode());h.update(str(x.shape).encode());h.update(x.tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--job',type=Path,required=True);p.add_argument('--stage',type=int,choices=[2,3],required=True)
    a=p.parse_args();j=a.job;r=j.parent.parent;torch.set_num_threads(4)
    if a.stage==2:
        path=j/'run/checkpoints/NeuroLM-B/ckpt.pt';expected_epoch=19;microsteps=19140;steps=19140
        proof=json.loads((j/'run/completed.json').read_text());assert proof['epochs_completed']==20 and proof['steps']==19140
        assert proof['frozen_state_unchanged'] and proof['gpt_weights_updated']
        initial=json.loads((r/'preflight.json').read_text())
        encoder=initial['official_encoder_sha256'];gpt=initial['initial_gpt_block_sha256']
        assert json.loads((j/'run/initialization.json').read_text())['strict_full_checkpoint_load']
    else:
        path=j/'train/checkpoints/instruction-B/ckpt.pt';expected_epoch=4;microsteps=7182*5;steps=898*5
        initial=json.loads((j.parent/'stage2/run/completed.json').read_text())['final']
        encoder=initial['frozen_encoder_sha256'];gpt=initial['gpt_block_sha256']
        manifest=json.loads((j/'train/run_manifest.json').read_text())
        assert manifest['microsteps_per_epoch']==7182 and manifest['dataset_counts']['TUSL']['train']==174
        assert manifest['args']['epochs']==5 and manifest['args']['seed']==int(j.parent.name.split('-')[1])
    saved=torch.load(str(path),map_location='cpu',mmap=True,weights_only=False)
    assert saved['epoch']==expected_epoch and saved['iter_num']==microsteps
    observed={int(v['step']) for v in saved['optimizer']['state'].values() if 'step' in v}
    assert observed=={steps},observed
    assert all(torch.isfinite(v).all() for v in saved['model'].values() if v.is_floating_point())
    assert digest(saved['model'],'tokenizer.')==encoder
    assert digest(saved['model'],'GPT2.transformer.h.0.')!=gpt
    with path.open('rb') as f:h=hashlib.file_digest(f,'sha256').hexdigest()
    report=dict(ok=True,stage=a.stage,epochs_completed=expected_epoch+1,microsteps=microsteps,optimizer_steps=steps,
                frozen_encoder_unchanged=True,gpt_weights_updated=True,all_model_tensors_finite=True,
                checkpoint_path=str(path),checkpoint_sha256=h,checkpoint_bytes=path.stat().st_size,encoder_sha256=encoder)
    (j/'checkpoint_audit.json').write_text(json.dumps(report,indent=2))
    print('CHECKPOINT_AUDIT='+json.dumps(report),flush=True)


if __name__=='__main__':main()
