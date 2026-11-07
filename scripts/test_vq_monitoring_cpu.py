"""Two CPU ranks: unchanged gradients, padding-excluded counts, pure evaluation."""
import importlib.util,json,os,sys
from pathlib import Path
from contextlib import nullcontext
import torch
import torch.distributed as dist

root=Path('/mnt/cfs-omni/lhy/neurolm-reproduction/vq-five-full-20261107-r1')
sys.path.insert(0,str(root/'code'))
from model.model_vq import VQ
from model.model_neural_transformer import NTConfig
from vq_monitor import fixed_probe,usage_summary,epoch_summary
torch.set_num_threads(1);dist.init_process_group('gloo');rank=dist.get_rank()
spec=importlib.util.spec_from_file_location('old_vq','/mnt/cfs-omni/lhy/neurolm-reproduction/vq-smoke-20261106-r1/code/model/model_vq.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
encoder=NTConfig(n_layer=1,n_head=12,n_embd=768,block_size=8,out_chans=16,in_chans=1)
decoder=NTConfig(n_layer=1,n_head=12,n_embd=768,block_size=8,in_chans=8)
torch.manual_seed(11)
before=old.VQ(encoder,decoder,n_embed=16,embed_dim=8)
after=VQ(encoder,decoder,n_embed=16,embed_dim=8);after.load_state_dict(before.state_dict())
torch.manual_seed(50+rank)
x=torch.randn(2,8,200);mask=torch.zeros(2,8,dtype=torch.bool);mask[:,:4]=True
chans=torch.zeros(2,8,dtype=torch.int32);times=chans.clone()
yf=torch.randn(2,8,100)*mask.unsqueeze(-1);yr=torch.randn(2,8,200)*mask.unsqueeze(-1)
batch=(x,yf,yr,chans,times,mask)
torch.manual_seed(70+rank);loss_before,_,_=before(*batch);loss_before.backward()
torch.manual_seed(70+rank);loss_after,_,log=after(*batch);loss_after.backward()
assert torch.equal(loss_before,loss_after)
for (name,a),(other,b) in zip(before.named_parameters(),after.named_parameters()):
    assert name==other
    if a.grad is not None: assert torch.equal(a.grad,b.grad),name
counts=log['train/valid_code_counts'].clone();dist.all_reduce(counts)
assert int(counts.sum())==16
state={name:value.clone() for name,value in after.state_dict().items()};rng=torch.get_rng_state().clone()
probe=fixed_probe(after,batch,'cpu',nullcontext())
assert torch.equal(rng,torch.get_rng_state()) and after.training
assert all(torch.equal(value,after.state_dict()[name]) for name,value in state.items())
rows=[dict(reconstruction=2.0,frequency=1.0,raw=1.0,codebook=.001,adversarial=1.4,
           adversarial_eeg=.7,adversarial_text=.7,lr=5e-5,alpha=.46) for _ in range(5)]
summary=epoch_summary(5,rows,counts,counts,probe,probe,[])
assert summary['lr_last']==5e-5 and summary['code_usage']['valid_eeg_tokens']==16
assert summary['automatic_stop'] is False
if rank==0:
    result=dict(ok=True,world_size=dist.get_world_size(),device='cpu',unchanged_loss_and_gradients=True,
                excluded_padding=True,valid_tokens=16,evaluation_preserves_state_and_rng=True,
                probe=probe,code_usage=usage_summary(counts))
    (root/'monitor_test.json').write_text(json.dumps(result,indent=2));print('MONITOR_TEST='+json.dumps(result),flush=True)
dist.destroy_process_group()
