"""Compare stratified source samples with actual CFS files and real stage-3 loaders."""
import argparse
import collections
import datetime
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np
import torch
import tiktoken


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--samples',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    torch.set_num_threads(1)
    os.environ['TIKTOKEN_CACHE_DIR']=str(args.root.parent/'cache/tiktoken')
    sys.path.insert(0,str(args.root/'author'))
    sys.path.insert(0,str(args.root/'scripts'))
    import downstream_dataset as ds
    from downstream_eval_core import build_candidates
    enc=tiktoken.get_encoding('gpt2')
    folders={
        'TUAB':['TUAB/processed/train','TUAB/processed/val','TUAB/processed/test'],
        'TUEV':['TUEV/processed_train','TUEV/processed_eval','TUEV/processed_test'],
        'HMC':['HMC/train','HMC/eval','HMC/test'],
        'Workload':['EEGWorkload/train','EEGWorkload/eval','EEGWorkload/test'],
        'TUSL':['TUSL/train','TUSL/eval','TUSL/test']}
    loaders={'TUAB':ds.TUABLoader,'TUEV':ds.TUEVLoader,'HMC':ds.HMCLoader,'Workload':ds.WorkloadLoader,'TUSL':ds.TUSLLoader}
    expected_answers={'TUAB':['No','Yes'],'TUEV':['(A)','(B)','(C)','(D)','(E)','(F)'],
                      'HMC':['(A)','(B)','(C)','(D)','(E)'],'Workload':['No','Yes'],'TUSL':['(A)','(B)','(C)']}
    samples=json.loads(args.samples.read_text())
    counts=collections.Counter();classes=collections.Counter();errors=[];digest=hashlib.sha256();extra_prefix={}
    for row in samples:
        task,split=row['task'],row['split']
        folder=args.root/'data'/folders[task][('train','val','test').index(split)]
        path=folder/row['file']
        try:
            body=path.read_bytes();file_hash=hashlib.sha256(body).hexdigest()
            assert file_hash==row['source_sha256'],'Cloud/source file SHA mismatch'
            sample=pickle.loads(body)
            observed=int(np.asarray(sample['label'] if task=='TUEV' else sample['y']).item())
            assert observed==row['expected']==row['label'],'Cloud/source/raw label mismatch'
            expected=observed-1 if task=='TUEV' else observed
            loader=loaders[task](str(folder),[path.name],is_instruct=True,is_val=False,eeg_max_len=276,text_max_len=80)
            train=loader[0]
            loader.is_val=True;validation=loader[0]
            assert int(np.asarray(validation[2]).item())==expected,'Evaluation label mismatch'
            text,target=train[1],train[2]
            positions=torch.where(target!=-1)[0]
            assert positions.numel()>0 and torch.equal(target[positions],text[positions+1]),'Next-token targets mismatch'
            candidates,answers=build_candidates(loader,enc.decode)
            supervised=target[positions].tolist();continuation=list(candidates[expected])
            assert supervised[-len(continuation):]==continuation,'Supervised answer mismatch'
            extra=supervised[:-len(continuation)]
            # TUEV/HMC additionally supervise the opening parenthesis supplied
            # by the evaluation prompt. That is a known template boundary,
            # not a class-label shift. Verify this token explicitly.
            assert enc.decode(extra)==(' (' if task in ('TUEV','HMC') else ''),'Unexpected supervised prompt prefix'
            extra_prefix[task]=enc.decode(extra)
            assert answers[expected_answers[task][expected]]==expected,'Decoded answer/label mapping mismatch'
            valid_text=len(loader.text[expected])
            assert bool((target[valid_text-1:]==-1).all()),'Padding contributes to label loss'
            assert torch.equal(validation[1],loader.prompt),'Evaluation prompt mismatch'
            counts[(task,split)]+=1;classes[(task,split,observed)]+=1
            digest.update((task+'\t'+split+'\t'+path.name+'\t'+file_hash+'\t'+str(expected)+'\n').encode())
        except Exception as exc:
            errors.append({'task':task,'split':split,'file':row['file'],'error':repr(exc)})
    report={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'scope':'Stratified sample of actual CFS stage-3 inputs: source SHA, raw label, eval integer, training answer tokens and decoding mapping.',
        'sampling':'Up to 20 examples per source-annotation class per split; uniformly sampled within each stratum without replacement, random seed 20261111. Full source audit is reported separately.',
        'requested':len(samples),'passed':sum(counts.values()),'error_count':len(errors),'errors':errors,
        'training_extra_supervised_prompt_suffix':extra_prefix,
        'row_digest_sha256':digest.hexdigest(),'sample_manifest_sha256':hashlib.sha256(args.samples.read_bytes()).hexdigest(),
        'code_sha256':{str(p.relative_to(args.root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (args.root/'author/downstream_dataset.py',args.root/'scripts/downstream_eval_core.py')},
        'coverage':[{'task':task,'split':split,'passed':count} for (task,split),count in sorted(counts.items())],
        'classes':[{'task':task,'split':split,'raw_label':label,'passed':count} for (task,split,label),count in sorted(classes.items())]}
    args.out.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    assert not errors and len(samples)==sum(counts.values())


if __name__=='__main__':main()
