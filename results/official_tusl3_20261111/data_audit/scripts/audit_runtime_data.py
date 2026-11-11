import collections
import datetime
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import pickle
import random
import statistics
import sys

import numpy as np
from scipy.signal import welch
import torch
import tiktoken

ROOT=Path('/mnt/cfs-omni/lhy/neurolm-reproduction/official-tusl3-20261111-r1')
OUT=ROOT/'monitor/data-audit-20261111-1714'
OUT.mkdir(parents=True,exist_ok=True)
torch.set_num_threads(1)
os.environ['TIKTOKEN_CACHE_DIR']=str(ROOT.parent/'cache/tiktoken')
sys.path.insert(0,str(ROOT/'author'))
import downstream_dataset as downstream
sys.path.insert(0,str(ROOT/'scripts'))
from downstream_eval_core import build_candidates
spec=importlib.util.spec_from_file_location('actual_stage2_dataset',ROOT/'code/dataset.py')
pretrain=importlib.util.module_from_spec(spec);spec.loader.exec_module(pretrain)
enc=tiktoken.get_encoding('gpt2')
gold=json.loads((OUT/'tusl_raw_gold.json').read_text())
manifest=json.loads((ROOT/'data/TUSL/manifest.json').read_text())
expected_hash={e['file']:e['sha256'] for r in manifest['records'] for e in r['events']}

def sha_bytes(data): return hashlib.sha256(data).hexdigest()

def signal_hash(x): return sha_bytes(np.ascontiguousarray(x,dtype='<f4').tobytes())

def psd(x):
    f,p=welch(x,fs=200,nperseg=min(400,x.shape[1]),axis=1)
    result={}
    for hz in (50,60):
        peak=p[:,abs(f-hz)<=.5].mean(axis=1)
        background=p[:,(abs(f-hz)>=2)&(abs(f-hz)<=5)].mean(axis=1)
        ratio=peak/np.maximum(background,np.finfo(float).tiny)
        result[str(hz)]=dict(median_peak_psd=float(np.median(peak)),median_ratio=float(np.median(ratio)),
            first_channel_peak_psd=float(peak[0]),first_channel_ratio=float(ratio[0]))
    return result

def stats(x):
    return dict(shape=list(x.shape),dtype=str(x.dtype),finite=bool(np.isfinite(x).all()),
        min=float(x.min()),max=float(x.max()),mean=float(x.mean()),std=float(x.std()),
        abs_p99=float(np.quantile(abs(x),.99)),zero_channels=int((np.ptp(x,axis=1)==0).sum()))

index=[]
with (ROOT/'assets/train_files.txt').open() as stream:
    for line in stream: index.append(json.loads(line))
tusl_index={Path(r['name']).name:r for r in index if r['name'].startswith('train/TUSL/')}
checks=[];errors=[];groups=collections.Counter();signal_groups=collections.defaultdict(list)
subjects=collections.defaultdict(set);recordings=collections.defaultdict(set)
for row in gold['records']:
    file=row['file'];path=ROOT/'data/TUSL'/file
    try:
        body=path.read_bytes();assert sha_bytes(body)==expected_hash[file]
        sample=pickle.loads(body);x=sample['X'];ch=len(row['channels']);valid=ch*10
        assert x.shape==(ch,2000) and np.isfinite(x).all()
        assert list(sample['ch_names'])==row['channels']
        assert list(sample['original_ch_names'])==row['original_ch_names']
        assert sample['reference']==row['reference']
        assert int(np.asarray(sample['y']).item())==row['label']
        assert sample['event_seconds']==[row['start'],row['end']]
        assert sample['units']=='uV' and sample['sampling_rate']==200
        expected=torch.from_numpy(np.asarray(x/100,dtype=np.float32).reshape(ch,10,200).transpose(1,0,2).reshape(valid,200))
        location=tusl_index[path.name] if row['split']=='train' else str(path)
        if isinstance(location,dict):
            shard=Path(location['shard'])
            with shard.open('rb') as f:f.seek(location['offset']);payload=f.read(location['size'])
            assert sha_bytes(payload)==location['sha256']
            packed=pickle.loads(payload)
            assert np.array_equal(packed['X'],x) and packed['ch_names']==sample['ch_names']
        s2=pretrain.PickleLoader([location],GPT_training=True)[0]
        assert torch.equal(s2[0][:valid],expected) and not s2[0][valid:].any()
        assert int(s2[3].sum())==valid and s2[5]==ch and s2[6]==valid
        expected_ids=torch.tensor([pretrain.standard_1020.index(name) for name in row['channels']]*10)
        expected_times=torch.arange(10).repeat_interleave(ch)
        assert torch.equal(s2[1][:valid],expected_ids) and torch.equal(s2[2][:valid],expected_times)
        assert not s2[4][:,:,valid:].any()
        ds=downstream.TUSLLoader(str(path.parent),[path.name],is_instruct=True,eeg_max_len=276,text_max_len=80)
        train=ds[0]
        for is_val in (False,True):
            dl=downstream.TUSLLoader(str(path.parent),[path.name],is_instruct=True,is_val=is_val,eeg_max_len=276,text_max_len=80)
            xx,txt,target,ids,times,mask,attention=dl[0]
            assert torch.equal(xx[:valid],expected) and not xx[valid:].any()
            assert int(mask.sum())==valid and torch.equal(ids[:valid],expected_ids) and torch.equal(times[:valid],expected_times)
            want=torch.tril(torch.ones(attention.shape[-2:],dtype=torch.bool))
            for t in range(10):want[t*ch:(t+1)*ch,t*ch:(t+1)*ch]=True
            want[:,valid:276]=False
            assert torch.equal(attention[0],want)
            if is_val:
                assert target==row['label'] and torch.equal(txt,dl.prompt)
                candidates,answers=build_candidates(dl,enc.decode)
                assert answers=={'(A)':0,'(B)':1,'(C)':2}
            else:
                positions=torch.where(target!=-1)[0]
                assert torch.equal(target[positions],txt[positions+1])
                supervised=enc.decode(target[positions].tolist())
                assert supervised=='ABC'[row['label']]+') <|endoftext|>'
                text_len=dl.text[row['label']].numel()
                assert (target[:dl.prompt.numel()-1]==-1).all() and (target[text_len-1:]==-1).all()
        record=dict(file=file,split=row['split'],label=row['label'],channels=ch,reference=row['reference'],
            valid_tokens=valid,supervised_answer=supervised,loader_max_abs_error=0.0,
            signal_sha256_float32=signal_hash(x),**stats(x))
        checks.append(record);groups[(row['split'],ch,row['reference'],row['label'])]+=1
        signal_groups[record['signal_sha256_float32']].append(file)
        subject=Path(row['source']).stem.split('_',1)[0]
        subjects[row['split']].add(subject);recordings[row['split']].add(row['source'])
    except Exception as exc:
        errors.append(dict(file=file,error=repr(exc)))
overlap={f'{a}/{b}':dict(filename_patient_id_overlap=len(subjects[a]&subjects[b]),
    recording_overlap=len(recordings[a]&recordings[b])) for a,b in [('train','eval'),('train','test'),('eval','test')]}
duplicates=[v for v in signal_groups.values() if len(v)>1]
report=dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),checked=len(checks),expected=len(gold['records']),
    errors=errors,raw_annotation_errors=gold['errors'],train_shard_checks=len(tusl_index),
    split_overlap=overlap,patient_identifier_definition='EDF filename prefix before first underscore',
    duplicate_signal_groups=duplicates,groups=[dict(split=k[0],channels=k[1],reference=k[2],label=k[3],count=v) for k,v in sorted(groups.items())],samples=checks)
(OUT/'tusl_runtime_audit.json').write_text(json.dumps(report,indent=2))
print('TUSL='+json.dumps({k:v for k,v in report.items() if k not in ('samples','groups','duplicate_signal_groups')}),flush=True)

paired=[]
for task in ('TUAB','TUEV'):
    by_record=collections.defaultdict(list)
    for row in index:
        if row['name'].startswith('train/'+task+'/'):
            name=Path(row['name']).stem;record=name.rsplit('_' if task=='TUAB' else '-',1)[0];by_record[record].append(row)
    rng=random.Random(20261111)
    chosen=sorted(rng.sample(sorted(by_record),min(60,len(by_record))))
    for record in chosen:
        row=rng.choice(by_record[record]);name=Path(row['name']).name
        location=dict(row,shard=str(ROOT/'assets'/row['shard']))
        with Path(location['shard']).open('rb') as f:f.seek(row['offset']);payload=f.read(row['size'])
        assert sha_bytes(payload)==row['sha256']
        stage2=pickle.loads(payload)
        directory=ROOT/'data'/('TUAB/processed/train' if task=='TUAB' else 'TUEV/processed_train')
        path=directory/name;body=path.read_bytes();stage3=pickle.loads(body)
        x2=np.asarray(stage2['X']);x3=np.asarray(stage3['X'] if task=='TUAB' else stage3['signal'])
        assert x2.shape==x3.shape and np.isfinite(x2).all() and np.isfinite(x3).all()
        delta=x2.astype(float)-x3.astype(float)
        paired.append(dict(task=task,file=name,record=record,index=int(Path(name).stem.rsplit('_' if task=='TUAB' else '-',1)[1]),
            label=int(np.asarray(stage3['y'] if task=='TUAB' else stage3['label']).ravel()[0]),
            stage2_payload_sha256=row['sha256'],stage3_payload_sha256=sha_bytes(body),
            stage2_signal_sha256_float32=signal_hash(x2),stage3_signal_sha256_float32=signal_hash(x3),
            equal_after_float32=bool(np.array_equal(x2.astype(np.float32),x3.astype(np.float32))),
            rmse=float(np.sqrt(np.mean(delta**2))),max_abs_difference=float(abs(delta).max()),
            stage2_stats=stats(x2),stage3_stats=stats(x3),stage2_psd=psd(x2),stage3_psd=psd(x3)))
summary=[]
for task in ('TUAB','TUEV'):
    rows=[r for r in paired if r['task']==task]
    summary.append(dict(task=task,samples=len(rows),distinct_records=len(rows),
        equal_float32=sum(r['equal_after_float32'] for r in rows),median_rmse=statistics.median(r['rmse'] for r in rows),
        median_60hz_ratio_stage2=statistics.median(r['stage2_psd']['60']['median_ratio'] for r in rows),
        median_60hz_ratio_stage3=statistics.median(r['stage3_psd']['60']['median_ratio'] for r in rows)))
(OUT/'stage2_stage3_pairs.json').write_text(json.dumps(dict(sampling='seed 20261111; 60 independent EDF records per task; one random existing train window per record',summary=summary,samples=paired),indent=2))
print('PAIRS='+json.dumps(summary),flush=True)
selected=[]
for task in ('TUAB','TUEV'):
    rows=[r for r in paired if r['task']==task]
    order=sorted(rows,key=lambda r:r['rmse'])
    for ix in [0,len(order)//2,len(order)-1]:selected.append(order[ix])
(OUT/'raw_reconstruction_plan.json').write_text(json.dumps(selected,indent=2))
