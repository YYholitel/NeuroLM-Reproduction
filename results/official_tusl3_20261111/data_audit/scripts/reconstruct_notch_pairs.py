import datetime
import hashlib
import json
from pathlib import Path
import pickle

import mne
import numpy as np

OUT=Path('/data/lihy/NeuroLM/outputs/data-audit-20261111-1714')
DATA=Path('/data/lihy/NeuroLM/datasets')
plan=json.loads((OUT/'raw_reconstruction_plan.json').read_text())
channels='FP1 FP2 F3 F4 C3 C4 P3 P4 O1 O2 F7 F8 T3 T4 T5 T6 A1 A2 FZ CZ PZ T1 T2'.split()
order=['EEG '+name+'-REF' for name in channels]

def sha(x): return hashlib.sha256(np.ascontiguousarray(x,dtype='<f4').tobytes()).hexdigest()

raw_files={}
for task,version in [('TUAB','v3.0.1'),('TUEV','v2.0.1')]:
    raw_files[task]={p.stem:p for p in (DATA/'raw/tuh'/task.lower()/version/'edf/train').rglob('*.edf')}
results=[]
for row in plan:
    task=row['task'];name=row['file'];idx=row['index']
    path=raw_files[task][row['record']]
    downstream=DATA/('TUAB/processed/train' if task=='TUAB' else 'TUEV/processed_train')/name
    body=downstream.read_bytes();sample=pickle.loads(body)
    x3=np.asarray(sample['X'] if task=='TUAB' else sample['signal'])
    result=dict(task=task,file=name,index=idx,source_edf=str(path),
        ncc_downstream_payload_matches_tencent=hashlib.sha256(body).hexdigest()==row['stage3_payload_sha256'],
        ncc_downstream_signal_matches_tencent=sha(x3)==row['stage3_signal_sha256_float32'],
        routes=[])
    assert result['ncc_downstream_signal_matches_tencent'],name
    raw=mne.io.read_raw_edf(str(path),preload=True,verbose='ERROR')
    raw.pick(order).reorder_channels(order)
    raw.filter(l_freq=.1,h_freq=75.,n_jobs=1,verbose='ERROR')
    for frequency in (50.,60.):
        filtered=raw.copy()
        filtered.notch_filter(frequency,n_jobs=1,verbose='ERROR')
        filtered.resample(200,n_jobs=1,verbose='ERROR')
        signal=filtered.get_data(units='uV')
        if task=='TUAB':
            x=signal[:,idx*2000:(idx+1)*2000]
        else:
            events=np.atleast_2d(np.genfromtxt(path.with_suffix('.rec'),delimiter=','))
            start=np.where(filtered.times>=events[idx,1])[0][0]
            end=np.where(filtered.times>=events[idx,2])[0][0]
            offset=signal.shape[1]
            tiled=np.concatenate([signal,signal,signal],axis=1)
            x=tiled[:,offset+start-400:offset+end+400]
        assert x.shape==x3.shape
        delta=x.astype(float)-x3.astype(float)
        route=dict(notch_hz=frequency,shape=list(x.shape),
            reconstructed_signal_sha256_float32=sha(x),
            matches_stage2_float32=sha(x)==row['stage2_signal_sha256_float32'],
            matches_stage3_float32=sha(x)==row['stage3_signal_sha256_float32'],
            downstream_rmse=float(np.sqrt(np.mean(delta**2))),
            downstream_max_abs_error=float(np.abs(delta).max()))
        result['routes'].append(route)
        filtered.close()
        del filtered,signal,x,delta
        if task=='TUEV':del tiled
    raw.close();del raw
    results.append(result)
    print('RECONSTRUCTED='+json.dumps(result),flush=True)
    (OUT/'notch_reconstruction.json').write_text(json.dumps(dict(
        checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        mne_version=mne.__version__,numpy_version=np.__version__,samples=results),indent=2))
