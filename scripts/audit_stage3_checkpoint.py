"""Verify completed instruction training and the embedded frozen stage-two encoder."""
from pathlib import Path
import hashlib
import json
import torch
ROOT=Path('/mnt/cfs-omni/lhy/neurolm-reproduction/stage3-new-pretrain-20261109-r1')
p=ROOT/'train/checkpoints/instruction-B/ckpt.pt'
torch.set_num_threads(4)
saved=torch.load(str(p),map_location='cpu',weights_only=False,mmap=True)
assert saved['epoch']==0 and saved['iter_num']==7182
steps={int(state['step']) for state in saved['optimizer']['state'].values() if 'step' in state}
assert steps=={898},steps
assert all(torch.isfinite(value).all() for value in saved['model'].values() if value.is_floating_point())
def digest(prefix):
    result=hashlib.sha256();items=sorted((k[len(prefix):],v) for k,v in saved['model'].items() if k.startswith(prefix));assert items
    for name,value in items:
        array=value.detach().cpu().contiguous().numpy()
        result.update(name.encode());result.update(str(array.shape).encode());result.update(array.tobytes())
    return result.hexdigest()
initial=json.loads((ROOT.parent/'pretrain-full20-20261109-r1/run/completed.json').read_text())['final']
assert digest('tokenizer.')==initial['frozen_encoder_sha256']
assert digest('GPT2.transformer.h.0.')!=initial['gpt_block_sha256']
with p.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
report=dict(ok=True,epochs_completed=1,microsteps=7182,optimizer_steps=898,all_model_tensors_finite=True,
            frozen_encoder_unchanged=True,gpt_weights_updated=True,checkpoint_sha256=sha,checkpoint_bytes=p.stat().st_size,
            checkpoint_path=str(p),encoder_sha256=digest('tokenizer.'))
(ROOT/'checkpoint_audit.json').write_text(json.dumps(report,indent=2));print('CHECKPOINT_AUDIT='+json.dumps(report),flush=True)
