"""Prepare a fixed 160-window held-out probe using the VQ training preprocessing.

Run on nccserv0 beside prepare_five_dataset_vq.py. TUAB/TUEV are rebuilt from
their source EDF with 60 Hz notch. Existing validation membership is preserved.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
from pathlib import Path
import random
import tarfile

from prepare_five_dataset_vq import fingerprint, make_group

SPLITS={'TUAB':'TUAB/processed/val','TUEV':'TUEV/processed_eval','HMC':'HMC/eval',
        'Workload':'EEGWorkload/eval','TUSL':'TUSL/eval'}


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--source-root',type=Path,required=True)
    parser.add_argument('--train-plan',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    args.output_dir.mkdir(parents=True,exist_ok=False)
    train=json.loads(args.train_plan.read_text())
    train_names=defaultdict(set)
    for group in train['groups']:
        if group['kind']=='edf':train_names[group['dataset']].update(name for name,index in group['windows'])
        elif group['kind']=='pickles':train_names[group['dataset']].update(Path(item['path']).name for item in group['inputs'])
    rng=random.Random(20261108);groups=[];selection={}
    for dataset,relative in SPLITS.items():
        candidates=sorted((args.source_root/relative).glob('*.pkl'))
        assert len(candidates)>=32,(dataset,len(candidates))
        selected=sorted(rng.sample(candidates,32));names=[path.name for path in selected]
        assert not set(names)&train_names[dataset],('validation/training overlap',dataset)
        selection[dataset]={'population':len(candidates),'selected_names':names,
                            'source_files':[fingerprint(path) for path in selected]}
        if dataset in ['TUAB','TUEV']:
            version='v3.0.1' if dataset=='TUAB' else 'v2.0.1'
            rawroot=args.source_root/'raw/tuh'/dataset.lower()/version/'edf/train'
            members=defaultdict(list)
            for path in selected:
                stem,index=path.stem.rsplit('_' if dataset=='TUAB' else '-',1)
                members[stem].append([path.name,int(index)])
            sources={}
            for path in rawroot.rglob('*.edf'):
                if path.stem in members:
                    assert path.stem not in sources
                    sources[path.stem]=path
            assert set(sources)==set(members)
            for stem,windows in members.items():
                group=dict(id=dataset+'-'+stem,kind='edf',dataset=dataset,input=fingerprint(sources[stem]),
                           windows=windows,count=len(windows))
                if dataset=='TUEV':group['annotation']=fingerprint(sources[stem].with_suffix('.rec'))
                groups.append(group)
        else:
            groups.append(dict(id=dataset+'-probe',kind='pickles',dataset=dataset,
                               inputs=[fingerprint(path) for path in selected],count=32))
    manifest=[]
    archive_path=args.output_dir/'validation_probe.tar.gz'
    with tarfile.open(archive_path,'w:gz') as output,ProcessPoolExecutor(max_workers=4) as pool:
        for group,body,stats in pool.map(make_group,groups):
            with tarfile.open(fileobj=io.BytesIO(body),mode='r:gz') as source:
                for member in source:
                    assert member.isfile() and member.name.startswith('train/'+group['dataset']+'/')
                    payload=source.extractfile(member).read()
                    name='val/'+member.name[len('train/'):]
                    target=tarfile.TarInfo(name);target.size=len(payload);target.mode=0o600
                    output.addfile(target,io.BytesIO(payload))
                    manifest.append(dict(name=name,sha256=hashlib.sha256(payload).hexdigest(),size=len(payload)))
            print(json.dumps({'prepared':len(manifest),'total':160}),flush=True)
        assert len(manifest)==160 and len({item['name'] for item in manifest})==160
        report=dict(seed=20261108,samples=160,selection=selection,files=manifest,
                    split='existing validation partitions; no training-window overlap',
                    source_train_plan_sha256=train['plan_sha256'],notch_hz={'TUAB':60,'TUEV':60},
                    preprocessing='same prepare_five_dataset_vq.make_group used for training inputs')
        data=json.dumps(report,indent=2).encode();entry=tarfile.TarInfo('validation_manifest.json');entry.size=len(data)
        output.addfile(entry,io.BytesIO(data))
    (args.output_dir/'validation_manifest.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'ready':True,'samples':160,'bytes':archive_path.stat().st_size,
                      'sha256':hashlib.sha256(archive_path.read_bytes()).hexdigest()}),flush=True)


if __name__=='__main__':main()
