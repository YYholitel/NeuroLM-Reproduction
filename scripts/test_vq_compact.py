"""Exercise the requested codebook geometry through a real VQ optimization and reload."""
import argparse
import copy
import io
import json
from pathlib import Path
import sys
import torch

parser=argparse.ArgumentParser(__doc__);parser.add_argument('--runtime',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args();sys.path.insert(0,str(args.runtime));torch.set_num_threads(2);torch.manual_seed(42)
import train_vq
from model.model_neural_transformer import NTConfig
from model.model_vq import VQ
from vq_monitor import usage_summary
saved_argv=sys.argv
sys.argv=['train_vq.py','--train_manifest','unused','--data_ready_manifest','unused','--vq_codebook_size','1024','--vq_embed_dim','64','--vq_dead_code_patience','50','--vq_dead_code_interval','50']
options=train_vq.vq_options(train_vq.get_args());sys.argv=saved_argv
assert options['n_embed']==1024 and options['embed_dim']==64 and options['dead_code_patience']==50
enc=NTConfig(n_layer=1,n_head=12,n_embd=768,block_size=8,out_chans=16,in_chans=1)
dec=NTConfig(n_layer=1,n_head=12,n_embd=768,block_size=8,in_chans=64)
model=VQ(enc,dec,**options)
assert list(model.quantize.embedding.weight.shape)==[1024,64]
assert model.encode_task_layer[-1].out_features==64
x=torch.randn(2,8,200);freq=torch.randn(2,8,100);raw=torch.randn_like(x)
channels=torch.zeros(2,8,dtype=torch.long);times=channels.clone()
mask=torch.arange(8).expand(2,-1)<torch.tensor([[3],[5]])
optimizer=model.configure_optimizers(.01,5e-5,(.9,.999),'cpu')
loss,_,log=model(x,freq,raw,channels,times,mask);loss.backward()
assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
optimizer.step();optimizer.zero_grad();model.eval()
counts=log['train/valid_code_counts'];assert counts.shape==(1024,) and int(counts.sum())==8
usage=usage_summary(counts);assert usage['active_fraction']==usage['active_codes']/1024
initial=copy.deepcopy(model.state_dict());indices=model.get_codebook_indices(x,channels,times,mask)
assert not indices[~mask].any() and indices[mask].max()<1024
assert all(torch.equal(value,model.state_dict()[name]) for name,value in initial.items())
buffer=io.BytesIO();torch.save(dict(state=model.state_dict(),options=options,enc=vars(enc),dec=vars(dec)),buffer);buffer.seek(0)
saved=torch.load(buffer,weights_only=True)
restored=VQ(NTConfig(**saved['enc']),NTConfig(**saved['dec']),**saved['options']).eval()
restored.load_state_dict(saved['state'],strict=True)
assert torch.equal(indices,restored.get_codebook_indices(x,channels,times,mask))
report=dict(ok=True,codebook_shape=[1024,64],cli_geometry_applied=True,actual_optimizer_step=True,finite_gradients=True,
            checkpoint_strict_roundtrip=True,evaluation_unchanged=True,valid_tokens_counted=8,active_fraction_denominator=1024,
            dead_code_patience=50,dead_code_interval=50,test_encoder_decoder_layers=[1,1],training_encoder_decoder_layers=[12,3])
args.output.write_text(json.dumps(report,indent=2));print('COMPACT_TEST='+json.dumps(report),flush=True)
