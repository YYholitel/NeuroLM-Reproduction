"""Build and stream train-only VQ inputs without a second EEG copy on source disk.

TUAB/TUEV are rebuilt from EDF at 60 Hz notch using the existing train-window
membership. Other datasets retain their verified signals. All EEG is stored in
microvolts as float32, with explicit channel names and without class labels.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from collections import defaultdict
import contextlib
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import pickle
import struct
import sys
import tarfile
import time
import zlib

CHANNELS = ['FP1','FP2','F3','F4','C3','C4','P3','P4','O1','O2','F7','F8',
            'T3','T4','T5','T6','A1','A2','FZ','CZ','PZ','T1','T2']
SETS = {'TUAB':('TUAB/processed/train',298511),
        'TUEV':('TUEV/processed_train',68270), 'HMC':('HMC/train',91248),
        'Workload':('EEGWorkload/train',1508), 'TUSL':('TUSL/train',119)}
CHUNK = 32 * 1024 * 1024

def dump_atomic(path, obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.tmp-{os.getpid()}')
    tmp.write_text(json.dumps(obj,ensure_ascii=True,sort_keys=True),encoding='utf-8')
    os.replace(tmp,path)

def fingerprint(path):
    s=Path(path).stat()
    return {'path':str(path),'bytes':s.st_size,'mtime_ns':s.st_mtime_ns}

def make_plan(args):
    root=Path(args.source_root)
    groups=[]; counts={}; split_hashes={}
    # Small datasets first, followed by the expensive source-EDF conversion.
    for dataset in ['Workload','TUSL','HMC','TUEV','TUAB']:
        rel,expected=SETS[dataset]
        files=sorted((root/rel).glob('*.pkl'))
        assert len(files)==expected,(dataset,len(files),expected)
        names=[p.name for p in files]
        counts[dataset]=len(files)
        split_hashes[dataset]=hashlib.sha256(('\n'.join(names)+'\n').encode()).hexdigest()
        if dataset in ['TUAB','TUEV']:
            version='v3.0.1' if dataset=='TUAB' else 'v2.0.1'
            rawroot=root/'raw/tuh'/dataset.lower()/version/'edf/train'
            edfs={}
            for p in rawroot.rglob('*.edf'):
                assert p.stem not in edfs,('Duplicate EDF basename',p.stem)
                edfs[p.stem]=p
            members=defaultdict(list)
            sep='_' if dataset=='TUAB' else '-'
            for p in files:
                stem,index=p.stem.rsplit(sep,1)
                members[stem].append([p.name,int(index)])
            for stem,windows in sorted(members.items()):
                assert stem in edfs,('Missing original EDF',dataset,stem)
                g={'id':dataset+'-'+stem,'kind':'edf','dataset':dataset,
                   'input':fingerprint(edfs[stem]),'windows':windows,'count':len(windows)}
                if dataset=='TUEV': g['annotation']=fingerprint(edfs[stem].with_suffix('.rec'))
                groups.append(g)
        else:
            for start in range(0,len(files),256):
                selected=files[start:start+256]
                groups.append({'id':f'{dataset}-{start//256:05d}','kind':'pickles',
                    'dataset':dataset,'inputs':[fingerprint(p) for p in selected],
                    'count':len(selected)})
    text={}
    for name in ['train.bin','val.bin']:
        p=root/'text'/name
        text[name]=fingerprint(p)
        for offset in range(0,p.stat().st_size,CHUNK):
            groups.append({'id':f'text-{name}-{offset//CHUNK:05d}','kind':'text',
                'input':fingerprint(p),'offset':offset,'length':min(CHUNK,p.stat().st_size-offset),
                'target':'text/'+name,'count':0})
    plan={'schema':1,'exporter_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'counts':counts,'total_samples':sum(counts.values()),
        'source_split':'train only; existing validation and test windows excluded',
        'source_root':str(root),'split_membership_sha256':split_hashes,
        'notch_hz':{'TUAB':60,'TUEV':60,'HMC':50,'Workload':50,'TUSL':60},
        'sampling_rate':200,'stored_unit':'uV','stored_dtype':'float32',
        'normalization':'X/100 only in PickleLoader; per-sample z-score targets as upstream',
        'text':text,'groups':groups}
    identity=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    plan['plan_sha256']=identity
    dump_atomic(args.plan,plan)
    print(json.dumps({'plan_sha256':identity,'counts':counts,'groups':len(groups),
                      'total_samples':plan['total_samples'],'text_bytes':sum(v['bytes'] for v in text.values())}))

def validate_input(info):
    assert fingerprint(info['path'])==info,('Source changed after plan',info['path'])

def make_group(group):
    import numpy as np
    if group['kind']=='text':
        validate_input(group['input'])
        with open(group['input']['path'],'rb') as f:
            f.seek(group['offset']); body=f.read(group['length'])
        assert len(body)==group['length']
        return group, zlib.compress(body,1), {'uncompressed_bytes':len(body),'raw_sha256':hashlib.sha256(body).hexdigest()}
    samples=[]
    if group['kind']=='edf':
        import mne
        validate_input(group['input'])
        if 'annotation' in group: validate_input(group['annotation'])
        with contextlib.redirect_stdout(io.StringIO()):
            raw=mne.io.read_raw_edf(group['input']['path'],preload=True,verbose='ERROR')
            order=['EEG '+name+'-REF' for name in CHANNELS]
            assert all(name in raw.ch_names for name in order),('Missing channels',group['id'])
            raw.pick(order).reorder_channels(order)
            raw.filter(l_freq=0.1,h_freq=75.0,n_jobs=1,verbose='ERROR')
            raw.notch_filter(60.0,n_jobs=1,verbose='ERROR')
            raw.resample(200,n_jobs=1,verbose='ERROR')
            signal=raw.get_data(units='uV')
            times=raw.times
            if group['dataset']=='TUEV':
                events=np.atleast_2d(np.genfromtxt(group['annotation']['path'],delimiter=','))
                offset=signal.shape[1]
                tiled=np.concatenate([signal,signal,signal],axis=1)
            for name,index in group['windows']:
                if group['dataset']=='TUAB':
                    x=signal[:,index*2000:(index+1)*2000]
                    assert x.shape==(23,2000),(group['id'],index,x.shape)
                else:
                    start=np.where(times>=events[index,1])[0][0]
                    end=np.where(times>=events[index,2])[0][0]
                    x=tiled[:,offset+start-400:offset+end+400]
                    assert x.shape==(23,1000),(group['id'],index,x.shape)
                samples.append((name,np.asarray(x,dtype=np.float32),CHANNELS))
            raw.close()
    else:
        for info in group['inputs']:
            validate_input(info)
            with open(info['path'],'rb') as f: sample=pickle.load(f)
            samples.append((Path(info['path']).name,np.asarray(sample['X'],dtype=np.float32),sample['ch_names']))
    out=io.BytesIO(); min_v=float('inf'); max_v=-float('inf'); seconds=0; raw_bytes=0
    with gzip.GzipFile(fileobj=out,mode='wb',compresslevel=1,mtime=0) as gz:
        with tarfile.open(fileobj=gz,mode='w|') as tar:
            for name,x,channels in samples:
                assert x.ndim==2 and x.shape[0]==len(channels) and x.shape[1]%200==0
                assert x.shape[0]*(x.shape[1]//200)<=1024
                assert np.isfinite(x).all() and float(x.std())>1e-8,('Invalid signal',group['id'],name)
                min_v=min(min_v,float(x.min())); max_v=max(max_v,float(x.max()))
                seconds+=x.shape[1]/200
                data=pickle.dumps({'X':x,'ch_names':list(channels)},protocol=4)
                member=tarfile.TarInfo('train/'+group['dataset']+'/'+name)
                member.size=len(data); member.mode=0o600
                tar.addfile(member,io.BytesIO(data)); raw_bytes+=len(data)
    assert len(samples)==group['count']
    return group,out.getvalue(),{'samples':len(samples),'uncompressed_bytes':raw_bytes,
                                'window_seconds_sum':seconds,'min_uV':min_v,'max_uV':max_v}

def send_frame(header,body=b''):
    header=dict(header,payload_bytes=len(body),payload_sha256=hashlib.sha256(body).hexdigest())
    encoded=json.dumps(header,separators=(',',':')).encode()
    sys.stdout.buffer.write(struct.pack('!I',len(encoded))+encoded+body)
    sys.stdout.buffer.flush()

def stream(args):
    plan=json.loads(Path(args.plan).read_text())
    skip=json.load(sys.stdin)
    assert not skip.get('plan_sha256') or skip['plan_sha256']==plan['plan_sha256']
    done=set(skip.get('completed_groups',[]))
    send_frame({'kind':'plan'},zlib.compress(json.dumps(plan,separators=(',',':')).encode(),1))
    jobs=iter(g for index,g in enumerate(plan['groups'])
              if index % args.partitions == args.partition and g['id'] not in done)
    completed=0
    # Bound the number of in-flight records so network backpressure cannot fill RAM.
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[]
        for _ in range(args.workers):
            g=next(jobs,None)
            if g is not None: futures.append(pool.submit(make_group,g))
        while futures:
            future=futures.pop(0)
            group,body,stats=future.result()
            send_frame({'kind':'group','group':group,'stats':stats},body)
            completed+=1
            if completed%20==0: print(json.dumps({'sent_groups':completed,'last':group['id']}),file=sys.stderr,flush=True)
            g=next(jobs,None)
            if g is not None: futures.append(pool.submit(make_group,g))
    send_frame({'kind':'done','plan_sha256':plan['plan_sha256']})

def read_exact(n):
    out=bytearray()
    while len(out)<n:
        piece=sys.stdin.buffer.read(n-len(out))
        if not piece: raise EOFError('Incomplete transfer frame')
        out.extend(piece)
    return bytes(out)

def status(root):
    planpath=root/'source_plan.json'
    return {'plan_sha256':json.loads(planpath.read_text())['plan_sha256'] if planpath.exists() else None,
            'completed_groups':[p.stem for p in sorted((root/'transfer_receipts').glob('*.json'))]}

def receive(args):
    root=Path(args.output_root).resolve(); root.mkdir(parents=True,exist_ok=True)
    receipts=root/'transfer_receipts'; receipts.mkdir(exist_ok=True)
    while True:
        n=struct.unpack('!I',read_exact(4))[0]
        assert n<10_000_000
        header=json.loads(read_exact(n)); body=read_exact(header['payload_bytes'])
        assert hashlib.sha256(body).hexdigest()==header['payload_sha256']
        if header['kind']=='plan':
            plan=json.loads(zlib.decompress(body))
            old=status(root)
            assert not old['plan_sha256'] or old['plan_sha256']==plan['plan_sha256']
            dump_atomic(root/'source_plan.json',plan)
            continue
        if header['kind']=='done':
            state=status(root)
            expected={g['id'] for g in plan['groups']}
            actual=set(state['completed_groups'])
            assert actual.issubset(expected)
            summary={'ready':actual==expected,'plan_sha256':plan['plan_sha256'],'counts':plan['counts'],
                'total_samples':plan['total_samples'],'completed_groups':len(actual),
                'total_groups':len(expected),'time':time.time()}
            if summary['ready']: dump_atomic(root/'transfer_complete.json',summary)
            print(json.dumps(summary),flush=True)
            return
        group=header['group']; assert header['kind']=='group'
        if group['kind']=='text':
            data=zlib.decompress(body)
            assert len(data)==group['length'] and hashlib.sha256(data).hexdigest()==header['stats']['raw_sha256']
            p=root/group['target']; p.parent.mkdir(parents=True,exist_ok=True)
            with p.open('r+b' if p.exists() else 'w+b') as f:
                if p.stat().st_size!=group['input']['bytes']: f.truncate(group['input']['bytes'])
                f.seek(group['offset']); f.write(data); f.flush()
        else:
            count=0
            with tarfile.open(fileobj=io.BytesIO(body),mode='r:gz') as tar:
                for member in tar:
                    assert member.isfile() and member.name.startswith('train/'+group['dataset']+'/')
                    p=(root/member.name).resolve(); assert p.is_relative_to(root/'train')
                    p.parent.mkdir(parents=True,exist_ok=True)
                    data=tar.extractfile(member).read(); assert len(data)==member.size
                    tmp=p.with_suffix('.partial'); tmp.write_bytes(data); os.replace(tmp,p)
                    count+=1
            assert count==group['count']
        dump_atomic(receipts/(group['id']+'.json'),{'id':group['id'],
            'plan_sha256':plan['plan_sha256'],'compressed_sha256':header['payload_sha256'],
            'compressed_bytes':len(body),**header['stats']})
        if len(list(receipts.glob('*.json')))%20==0:
            report={'completed_groups':len(list(receipts.glob('*.json'))),'total_groups':len(plan['groups']),
                    'last':group['id'],'time':time.time()}
            dump_atomic(root/'transfer_progress.json',report); print(json.dumps(report),flush=True)

def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('mode',choices=['plan','stream','receive','status'])
    parser.add_argument('--source-root'); parser.add_argument('--output-root')
    parser.add_argument('--plan'); parser.add_argument('--workers',type=int,default=6)
    parser.add_argument('--partition',type=int,default=0)
    parser.add_argument('--partitions',type=int,default=1)
    args=parser.parse_args()
    assert 0<=args.partition<args.partitions
    if args.mode=='plan': make_plan(args)
    elif args.mode=='stream': stream(args)
    elif args.mode=='receive': receive(args)
    else: print(json.dumps(status(Path(args.output_root))))

if __name__=='__main__': main()
