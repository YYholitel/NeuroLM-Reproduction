"""Validate real 21/23-channel samples, official initialization and frozen updates."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import pickle
import sys
import torch


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True)
    p.add_argument('--job',type=Path);p.add_argument('--worker',action='store_true')
    a=p.parse_args();r=a.root;b=r.parent
    sys.path.insert(0,str(r/'code'))
    from official_checkpoint_init import load_official, sha, HASHES
    from pretrain_minimal import state_hash, build_targets
    for folder in ('code','author'):
        for name,h in json.loads((r/folder/'bundle_manifest.json').read_text()).items():
            assert sha(r/folder/name)==h,name
    for name,h in HASHES.items():assert sha(b/'checkpoints'/name)==h,name
    ready=json.loads((r/'assets/data_ready.json').read_text())
    assert ready['ready'] and ready['total_samples']==459711 and ready['tusl_skipped_records']==0
    assert sha(r/'assets/train_files.txt')==ready['train_manifest_sha256']
    if a.worker:
        assert torch.cuda.device_count()==8 and torch.cuda.is_bf16_supported()
        assert all('H20' in torch.cuda.get_device_name(i) for i in range(8))
        assert json.loads((r/'preflight.json').read_text())['ok']
        print('WORKER_PREFLIGHT='+json.dumps(dict(ok=True,gpu_count=8,torch=torch.__version__)),flush=True)
        return
    torch.set_num_threads(4);torch.manual_seed(42)
    network,target,initialization=load_official(b/'checkpoints/NeuroLM-B.pt',b/'checkpoints/VQ.pt')
    from dataset import PickleLoader
    spec=importlib.util.spec_from_file_location('tusl3_downstream',r/'author/downstream_dataset.py')
    loaders=importlib.util.module_from_spec(spec);spec.loader.exec_module(loaders)
    examples={};covered=0
    for path in sorted((r/'data/TUSL/train').glob('*.pkl')):
        with path.open('rb') as f:sample=pickle.load(f)
        ch=len(sample['ch_names']); examples.setdefault(ch,path)
        ds=PickleLoader([str(path)],GPT_training=True)
        x,c,t,m,g,n,k=ds[0]
        assert n==ch and k==ch*10 and m.sum()==ch*10 and torch.isfinite(x).all()
        dl=loaders.TUSLLoader(str(path.parent),[path.name],is_instruct=True,eeg_max_len=276,text_max_len=80)
        ix,txt,y,ic,it,im,att=dl[0]
        assert im.sum()==ch*10 and ic.shape[0]==276 and torch.isfinite(ix).all()
        assert torch.allclose(ix[:ch*10],x[:ch*10])
        covered+=1
    assert covered==174 and set(examples)=={21,23}
    initial=state_hash(network.tokenizer);gpt_before=state_hash(network.GPT2.transformer.h[0])
    quant_before=state_hash(target.quantize)
    optimizer=network.configure_optimizers(.1,5e-5,(.9,.95),'cpu')
    frozen={id(p) for p in network.tokenizer.parameters()}
    assert not any(id(p) in frozen for g in optimizer.param_groups for p in g['params'])
    cases=[];network.train();network.tokenizer.eval();optimizer.zero_grad(set_to_none=True)
    for ch,path in examples.items():
        batch=next(iter(torch.utils.data.DataLoader(PickleLoader([str(path)],GPT_training=True),batch_size=1)))
        x,c,t,m,g,n,k=batch
        y=build_targets(target,x,c,t,m,n,k,network.GPT2.config.vocab_size)
        loss,_,_=network(x,y,None,None,c,t,m,eeg_mask=g)
        assert torch.isfinite(loss);(loss/7).backward()
        cases.append(dict(stage=2,channels=ch,valid_tokens=int(m.sum()),loss=float(loss.detach())))
    tasks=[('TUAB','TUAB/processed/train',loaders.TUABLoader),('TUEV','TUEV/processed_train',loaders.TUEVLoader),
           ('HMC','HMC/train',loaders.HMCLoader),('Workload','EEGWorkload/train',loaders.WorkloadLoader),
           ('TUSL','TUSL/train',loaders.TUSLLoader)]
    for name,relative,cls in tasks:
        folder=r/'data'/relative
        if name=='TUSL':filename=examples[21].name
        else:
            with os.scandir(folder) as it:filename=next(x.name for x in it if x.name.endswith('.pkl'))
        ds=cls(str(folder),[filename],is_instruct=True,eeg_max_len=276,text_max_len=80)
        x,txt,y,c,t,m,g=next(iter(torch.utils.data.DataLoader(ds,batch_size=1)))
        ey=torch.full(x.shape[:2],-1-network.GPT2.config.vocab_size)
        loss,_,_=network(x,ey,txt,y,c,t,m,eeg_text_mask=g)
        assert torch.isfinite(loss);(loss/7).backward()
        cases.append(dict(stage=3,task=name,valid_tokens=int(m.sum()),loss=float(loss.detach())))
    assert all(p.grad is None for p in network.tokenizer.parameters())
    norm=torch.nn.utils.clip_grad_norm_(network.parameters(),1.,error_if_nonfinite=True);optimizer.step()
    assert state_hash(network.tokenizer)==initial and state_hash(target.quantize)==quant_before
    assert state_hash(network.GPT2.transformer.h[0])!=gpt_before
    result=dict(ok=True,diagnostic_only=True,initialization=initialization,tusl_train_samples_checked=covered,
                channel_counts=sorted(examples),cases=cases,gradient_norm=float(norm),
                frozen_encoder_unchanged=True,frozen_quantizer_unchanged=True,gpt_update_verified=True,
                train_samples=459711,official_encoder_sha256=initial,initial_gpt_block_sha256=gpt_before)
    (r/'preflight.json').write_text(json.dumps(result,indent=2))
    for seed in (42,43,44):(r/f'seed-{seed}/stage2/preflight.json').write_text(json.dumps(result,indent=2))
    print('PREFLIGHT='+json.dumps(result),flush=True)


if __name__=='__main__':main()
