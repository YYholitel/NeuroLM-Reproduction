"""Verify transfer, frozen split membership, text hashes and real VQ loading."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import sys
import time

COUNTS={'TUAB':298511,'TUEV':68270,'HMC':91248,'Workload':1508,'TUSL':119}
TEXT={'train.bin':(18071164978,'68a352b5dbd52e85b7a8a1556c1ccae65ce14b02e4913788b0883b89ceddc1d1'),
      'val.bin':(8869212,'009ecdd6da34f1ae915ff1f0f3aeb0416b02f2252356505b32ce44166dd5cfcc')}

def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(__doc__)
    ap.add_argument('--root',required=True)
    ap.add_argument('--assets',default='assets')
    ap.add_argument('--diagnostic',action='store_true')
    args=ap.parse_args(); root=Path(args.root).resolve(); assets=root/args.assets
    sys.path.insert(0,str(root/'code'))
    import numpy as np
    import torch
    from dataset import PickleLoader,read_pickle_sample
    torch.set_num_threads(2)
    runtime=json.loads((root/'code/bundle_manifest.json').read_text())
    for name,digest in runtime.items(): assert sha(root/'code'/name)==digest,('Runtime changed',name)
    transfer=json.loads((assets/'transfer_complete.json').read_text())
    plan=json.loads((assets/'source_plan.json').read_text())
    assert transfer['ready'] and transfer['plan_sha256']==plan['plan_sha256']
    if not args.diagnostic: assert plan['counts']==COUNTS and plan['total_samples']==459656
    entries=[]
    for p in sorted((assets/'indices').glob('*.json')): entries.extend(json.loads(p.read_text()))
    assert transfer['storage']=='indexed_tar'
    allfiles=[]; checks={}; rng=random.Random(20261107)
    for dataset,expected in sorted(plan['counts'].items()):
        files=sorted((e for e in entries if e['name'].split('/')[1]==dataset),key=lambda e:e['name'])
        assert len(files)==expected,(dataset,len(files),expected)
        names=[Path(e['name']).name for e in files]
        assert len(set(names))==len(names)
        membership=hashlib.sha256(('\n'.join(names)+'\n').encode()).hexdigest()
        if not args.diagnostic: assert membership==plan['split_membership_sha256'][dataset]
        allfiles.extend(files)
        chosen=sorted(set([0,len(files)-1]+rng.sample(range(len(files)),min(10,len(files)))))
        rows=[]
        for index in chosen:
            entry=files[index];location=dict(entry,shard=str(assets/entry['shard']))
            with open(location['shard'],'rb') as f:
                f.seek(entry['offset']);payload=f.read(entry['size'])
            assert hashlib.sha256(payload).hexdigest()==entry['sha256']
            sample=read_pickle_sample(location)
            assert set(sample)=={'X','ch_names'}
            x=sample['X']; assert x.dtype==np.float32 and np.isfinite(x).all()
            loaded=PickleLoader([location],GPT_training=False)[0]
            assert all(torch.isfinite(t).all() for t in loaded)
            if args.diagnostic:
                original=root/'preflight_input_v2'/entry['name']
                if original.exists():
                    unpacked=PickleLoader([str(original)],GPT_training=False)[0]
                    assert all(torch.equal(a,b) for a,b in zip(loaded,unpacked)),entry['name']
            n=x.shape[0]*(x.shape[1]//200)
            expected_x=torch.from_numpy((x/100).reshape(x.shape[0],-1,200).transpose(1,0,2).reshape(-1,200))
            assert torch.equal(loaded[0][:n],expected_x)
            assert int(loaded[-1].sum())==n
            rows.append({'file':Path(entry['name']).name,'shape':list(x.shape),'valid_tokens':n,
                         'min_uV':float(x.min()),'max_uV':float(x.max()),'mean_uV':float(x.mean()),
                         'loaded_finite':True,'scaling_exact':True,'storage':'indexed_tar'})
        checks[dataset]={'count':len(files),'membership_sha256':membership,'samples':rows}
        print(json.dumps({'verified_dataset':dataset,'count':len(files),'checked_pickles':len(rows)}),flush=True)
    text={}
    for name,(size,digest) in TEXT.items():
        p=assets/'text'/name
        if args.diagnostic and not any(g.get('target')=='text/'+name for g in plan['groups']): continue
        assert p.stat().st_size==size
        actual=sha(p); assert actual==digest,(name,actual)
        text[name]={'bytes':size,'sha256':actual}
        print(json.dumps({'verified_text':name,'bytes':size}),flush=True)
    if not args.diagnostic:
        embed=assets/'gpt2_wte.pt'
        assert sha(embed)=='a850ca8f8e1c4c8e6d6902686d5901740db9e82310bd45269cd84458d83d1bcf'
        assert (root/'code/train_vq.py').is_file()
        allfiles=sorted(allfiles,key=lambda e:e['name'])
        listing='\n'.join(json.dumps(e,separators=(',',':')) for e in allfiles)+'\n'
        (assets/'train_files.txt').write_text(listing)
        run=root/'run'; run.mkdir(exist_ok=True)
        link=run/'text'
        if not link.exists(): link.symlink_to(assets/'text',target_is_directory=True)
        assert link.resolve()==(assets/'text').resolve()
    result={'ready':True,'diagnostic':args.diagnostic,'plan_sha256':plan['plan_sha256'],
            'total_samples':sum(d['count'] for d in checks.values()),'counts':plan['counts'],
            'datasets':checks,'text':text,'runtime_sha256':runtime,'verified_at':time.time()}
    destination=assets/('diagnostic_preflight.json' if args.diagnostic else 'data_ready.json')
    tmp=destination.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2));os.replace(tmp,destination)
    print(json.dumps({'ready':True,'diagnostic':args.diagnostic,'samples':result['total_samples'],
                      'report':str(destination)}),flush=True)

if __name__=='__main__': main()
