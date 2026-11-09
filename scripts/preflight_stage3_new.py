"""CPU compatibility check on five actual training examples before matched stage three."""
import hashlib
import json
import os
from pathlib import Path
import sys
import torch

ROOT=Path('/mnt/cfs-omni/lhy/neurolm-reproduction/stage3-new-pretrain-20261109-r1')
BASE=ROOT.parent
NEW=BASE/'pretrain-full20-20261109-r1/run/checkpoints/NeuroLM-B/ckpt.pt'
OLD=BASE/'outputs/neurolm-instruction-five-e1-2gpu-20261108-r2/checkpoints/instruction-B/ckpt.pt'
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def digest(module):
    h=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        array=value.detach().cpu().contiguous().numpy()
        h.update(name.encode());h.update(str(array.shape).encode());h.update(array.tobytes())
    return h.hexdigest()
def main():
    torch.set_num_threads(4);torch.manual_seed(42)
    assert torch.__version__.split('+')[0]=='2.2.2'
    assert sys.version_info[:3]==(3,11,8)
    manifest=json.loads((ROOT/'author/bundle_manifest.json').read_text())
    for name,expected in manifest.items():assert sha(ROOT/'author'/name)==expected,name
    assert sha(NEW)=='b13d1acdea4ed32fa1a3e6c7163c6fde6a03857eda131e36b10cc1380a3c5353'
    assert sha(OLD)=='a10a23f6f4a69058c996c01096d47b0806dafd83a0d0ca6506e274a5c6372174'
    sys.path.insert(0,str(ROOT/'author'));sys.path.insert(0,str(ROOT/'scripts'))
    import downstream_dataset as loaders
    from model.model import GPTConfig
    from model.model_neurolm import NeuroLM
    from downstream_eval_core import build_candidates,generate_constrained
    from preflight_instruction_run import SPLITS
    import tiktoken
    counts={}
    for relative,expected in SPLITS.items():
        with os.scandir(BASE/'data'/relative) as entries:actual=sum(x.name.endswith('.pkl') for x in entries)
        assert actual==expected,(relative,actual,expected)
        counts[relative]=actual
    saved=torch.load(str(NEW),map_location='cpu',weights_only=False,mmap=True)
    assert saved['epoch']==19 and saved['iter_num']==19140 and saved['epoch_complete']
    model=NeuroLM(GPTConfig(**saved['model_args']),init_from='scratch')
    model.load_state_dict(saved['model'],strict=True)
    official=torch.load(str(BASE/'checkpoints/NeuroLM-B.pt'),map_location='cpu',weights_only=False,mmap=True)
    assert official['model_args']==saved['model_args']
    assert {k.removeprefix('_orig_mod.'):list(v.shape) for k,v in official['model'].items()}=={k:list(v.shape) for k,v in saved['model'].items()}
    del official,saved
    before=digest(model.tokenizer);gpt_before=digest(model.GPT2.transformer.h[0])
    assert before=='c7cdbc311544d5cc72003611788cea28c98a52778922b7a8e7d76710b314876a'
    assert all(not p.requires_grad for p in model.tokenizer.parameters())
    optimizer=model.configure_optimizers(.1,5e-5,(.9,.95),'cpu')
    frozen={id(p) for p in model.tokenizer.parameters()}
    assert not any(id(p) in frozen for group in optimizer.param_groups for p in group['params'])
    cases=[('TUAB','TUAB/processed/train',loaders.TUABLoader),('TUEV','TUEV/processed_train',loaders.TUEVLoader),
           ('HMC','HMC/train',loaders.HMCLoader),('Workload','EEGWorkload/train',loaders.WorkloadLoader),('TUSL','TUSL/train',loaders.TUSLLoader)]
    enc=tiktoken.get_encoding('gpt2');checks=[];optimizer.zero_grad(set_to_none=True)
    for name,relative,loader in cases:
        folder=BASE/'data'/relative
        with os.scandir(folder) as entries:filename=next(e.name for e in entries if e.name.endswith('.pkl'))
        dataset=loader(str(folder),[filename],is_instruct=True,eeg_max_len=276,text_max_len=80)
        x,text,y,chans,times,valid,mask=next(iter(torch.utils.data.DataLoader(dataset,batch_size=1,num_workers=0)))
        assert torch.isfinite(x).all() and (y!=-1).sum()>0
        model.train();eeg_y=torch.full(x.shape[:2],-1-model.GPT2.config.vocab_size)
        loss,_,_=model(x.float(),eeg_y,text,y,chans,times,valid,eeg_text_mask=mask)
        assert torch.isfinite(loss);(loss/len(cases)).backward()
        validation=loader(str(folder),[filename],is_instruct=True,is_val=True,eeg_max_len=276,text_max_len=80)
        candidates,answers=build_candidates(validation,enc.decode)
        vx,vt,vy,vc,vtime,vm,vatt=next(iter(torch.utils.data.DataLoader(validation,batch_size=1,num_workers=0)))
        model.eval()
        with torch.inference_mode():generated,_=generate_constrained(model,vx.float(),vt,vc,vtime,vm,vatt,candidates)
        decoded=enc.decode(generated[0][1:].tolist());answer=decoded.split('Answer:',1)[1].split('<|endoftext|>',1)[0].strip()
        assert answer in answers
        checks.append(dict(task=name,classes=len(candidates),answer_targets=int((y!=-1).sum()),
                           instruction_loss=float(loss.detach()),legal_constrained_answer=True))
    assert all(p.grad is None for p in model.tokenizer.parameters())
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
    assert before==digest(model.tokenizer) and gpt_before!=digest(model.GPT2.transformer.h[0])
    report=dict(ok=True,diagnostic_only=True,training_samples_only=True,model_architecture_matches_official=True,
                input_checkpoint=str(NEW),input_checkpoint_sha256=sha(NEW),counts=counts,cases=checks,
                frozen_encoder_unchanged=True,gpt_update_verified=True,gradient_norm=float(norm),
                runtime_files=len(manifest),note='Disposable CPU optimizer update; production starts from untouched full-20 checkpoint.')
    (ROOT/'preflight.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
