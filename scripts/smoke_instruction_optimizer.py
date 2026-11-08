"""Diagnostic only: all five instruction loaders, two-rank accumulation, one optimizer update.

Uses the author's prompt/labels and official B checkpoint. This is not an epoch,
validation/test evaluation, or a paper metric. The full run must start afresh.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--author-dir',type=Path,required=True)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--text-bin',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    a=parser.parse_args()
    sys.path.insert(0,str(a.author_dir.resolve()))
    import numpy as np
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel
    import train_instruction as author
    from types import SimpleNamespace
    args=SimpleNamespace(seed=42,dataset_dir=str(a.data_dir),eeg_batch_size=32,num_workers=0)
    author.init(args)
    rank=author.ddp_rank;device=author.device
    assert author.ddp_world_size==2 and torch.cuda.device_count()==2
    assert sys.version_info[:3]==(3,11,8)
    assert torch.__version__.split('+')[0]=='2.2.2' and torch.version.cuda=='12.1'
    assert np.__version__=='1.26.4' and author.dtype=='bfloat16'
    start=time.monotonic()
    checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    model_args=dict(n_layer=12,n_head=12,n_embd=768,block_size=1024,bias=False,vocab_size=50257,dropout=0.)
    for key in ('n_layer','n_head','n_embd','block_size','bias','vocab_size'):
        model_args[key]=checkpoint['model_args'][key]
    model=author.NeuroLM(author.GPTConfig(**model_args),init_from='scratch')
    state={key.removeprefix('_orig_mod.'):value for key,value in checkpoint['model'].items()}
    model.load_state_dict(state,strict=True)
    del state,checkpoint
    model.to(device)
    optimizer=model.configure_optimizers(.1,5e-4,(.9,.95),'cuda')
    raw=model
    model=DistributedDataParallel(model,device_ids=[author.ddp_local_rank])
    names=['TUAB','TUEV','HMC','Workload','TUSL']
    datasets={name:author.get_instruct_datasets(args,name,eeg_max_len=276,text_max_len=80) for name in names}
    iterators={name:iter(info['data_loader_train']) for name,info in datasets.items()}
    before=raw.pos_embed.weight.detach().clone()
    rows=[]
    optimizer.zero_grad(set_to_none=True)
    for microstep in range(8):
        name=names[microstep%len(names)]
        try:batch=next(iterators[name])
        except StopIteration:
            iterators[name]=iter(datasets[name]['data_loader_train'])
            batch=next(iterators[name])
        eeg,text,labels,chans,times,input_mask,mask=[x.to(device) for x in batch]
        eeg=eeg.float()
        y_eeg=torch.full(eeg.shape[:2],-1-raw.GPT2.config.vocab_size,device=device)
        assert (y_eeg+raw.GPT2.config.vocab_size==-1).all()
        assert (labels!=-1).sum()>0
        data=np.memmap(a.text_bin,dtype=np.uint16,mode='r')
        ix=torch.randint(len(data)-1024,(8,))
        tx=torch.stack([torch.from_numpy(data[i:i+1024].astype(np.int64)) for i in ix]).to(device)
        ty=torch.stack([torch.from_numpy(data[i+1:i+1025].astype(np.int64)) for i in ix]).to(device)
        model.require_backward_grad_sync=microstep==7
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss1,_,logits=model(eeg,y_eeg,text,labels,chans,times,input_mask,eeg_text_mask=mask)
            loss2,_,_=model(None,None,tx,ty)
            loss=(loss1+loss2)/8
        assert torch.isfinite(loss),'Non-finite diagnostic loss'
        loss.backward()
        row={'rank':rank,'microstep':microstep,'dataset':name,
             'instruction_loss':float(loss1.detach()),'text_loss':float(loss2.detach()),
             'answer_targets':int((labels!=-1).sum()),'logits_shape':list(logits.shape)}
        rows.append(row)
        print(json.dumps(row),flush=True)
        del logits,loss,loss1,loss2
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    optimizer.step()
    delta=float((raw.pos_embed.weight.detach()-before).abs().max())
    assert delta>0 and np.isfinite(delta),'No observed optimizer parameter change'
    gathered=[None]*2
    dist.all_gather_object(gathered,{'rank':rank,'rows':rows,'gradient_norm':float(norm),
                                  'position_embedding_max_update':delta,
                                  'peak_allocated_gib':torch.cuda.max_memory_allocated()/1024**3})
    if rank==0:
        a.out_dir.mkdir(parents=True,exist_ok=True)
        path=a.out_dir/'diagnostic_checkpoint.pt'
        torch.save({'model':raw.state_dict(),'optimizer':optimizer.state_dict(),
                    'model_args':model_args,'diagnostic_only':True,'optimizer_steps':1},path)
        with path.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
        report={'status':'passed','diagnostic_only':True,'paper_comparable':False,
                'evaluation_run':False,'complete_epoch':False,'seed':42,
                'world_size':2,'eeg_batch_per_rank':32,'text_batch_per_rank':8,
                'gradient_accumulation':8,'optimizer_steps':1,
                'global_eeg_samples':512,'global_text_samples':128,
                'sampling':'fixed 64 original train samples per task; loaders cycled across eight microsteps',
                'learning_rate':5e-4,'scheduler':'disabled for this one-update diagnostic only',
                'seconds':time.monotonic()-start,'ranks':gathered,
                'checkpoint_sha256':sha,'checkpoint_bytes':path.stat().st_size}
        (a.out_dir/'diagnostic_summary.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps({'diagnostic':'passed','checkpoint_sha256':sha}),flush=True)
    dist.barrier()
    dist.destroy_process_group()


if __name__=='__main__':main()
