import argparse
import collections
import concurrent.futures
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import pickle
import random
import time

import numpy as np

DATA=Path('/data/lihy/NeuroLM/datasets')
FOLDERS={
 'TUAB':{'train':'TUAB/processed/train','val':'TUAB/processed/val','test':'TUAB/processed/test'},
 'TUEV':{'train':'TUEV/processed_train','val':'TUEV/processed_eval','test':'TUEV/processed_test'},
 'HMC':{'train':'HMC/train','val':'HMC/eval','test':'HMC/test'},
 'Workload':{'train':'EEGWorkload/train','val':'EEGWorkload/eval','test':'EEGWorkload/test'},
 'TUSL':{'train':'TUSL-full-20261111-r1/train','val':'TUSL-full-20261111-r1/eval','test':'TUSL-full-20261111-r1/test'},
}
EXPECTED={'TUAB':[298511,73627,36945],'TUEV':[68270,15662,28305],
 'HMC':[91248,22124,23871],'Workload':[1508,290,290],'TUSL':[174,66,60]}
CATALOG={}
ANNOTATION_HASHES={}
OUT=None
LABELS_ONLY=False

def save(path,data):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(data,indent=2));os.replace(tmp,path)

def build_catalog():
    result={k:{} for k in FOLDERS}
    for p in (DATA/'raw/tuh/tuab/v3.0.1/edf').rglob('*.edf'):
        label=1 if 'abnormal' in p.parts else 0 if 'normal' in p.parts else None
        assert label is not None and p.stem not in result['TUAB'],p
        result['TUAB'][p.stem]=label
    for p in (DATA/'raw/tuh/tuev/v2.0.1/edf').rglob('*.rec'):
        rows=np.atleast_2d(np.genfromtxt(p,delimiter=','))
        labels=rows[:,3].tolist();assert all(x in (1,2,3,4,5,6) for x in labels)
        assert p.stem not in result['TUEV'];result['TUEV'][p.stem]=[int(x) for x in labels]
        ANNOTATION_HASHES[str(p.relative_to(DATA))]=hashlib.sha256(p.read_bytes()).hexdigest()
    mapping={'Sleep stage W':0,'Sleep stage N1':1,'Sleep stage N2':2,'Sleep stage N3':3,'Sleep stage R':4}
    for p in (DATA/'raw/hmc-sleep-staging-v1.1').rglob('*_sleepscoring.txt'):
        with p.open() as f:
            rows=[{k.strip():v.strip() for k,v in row.items()} for row in csv.DictReader(f)]
        selected=[r for r in rows if float(r['Duration'])==30]
        labels=[mapping[r['Annotation']] for r in selected]
        stem=p.name.removesuffix('_sleepscoring.txt')
        assert stem not in result['HMC'];result['HMC'][stem]=labels
        ANNOTATION_HASHES[str(p.relative_to(DATA))]=hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (DATA/'raw/eegmat-v1.0.0').glob('*.edf'):
        number=int(p.stem.rsplit('_',1)[1]);assert number in (1,2)
        result['Workload'][p.stem]=[number-1]*29
    labels={'bckg':0,'seiz':1,'slow':2}
    for edf in (DATA/'raw/tuh/tusl/v2.0.1').rglob('*.edf'):
        p=edf.with_suffix('.tse_agg')
        if not p.exists():p=edf.with_suffix('.tse_ag')
        rows=[line.split() for line in p.read_text().splitlines() if line.strip() and not line.strip().startswith(('#','version'))]
        result['TUSL'][edf.stem]=[labels[r[2]] for r in rows]
        ANNOTATION_HASHES[str(p.relative_to(DATA))]=hashlib.sha256(p.read_bytes()).hexdigest()
    return result

def scan(job):
    task,split,part,paths=job
    counts=collections.Counter();failures=collections.Counter()
    sample_groups=collections.defaultdict(list);seen=collections.Counter();examples=[]
    rng=random.Random(20261111+sum(map(ord,task+split))+part)
    digest=hashlib.sha256();start=time.monotonic();total_bytes=0;skipped_bytes=0
    progress=OUT/f'{task}-{split}-{part:04d}-progress.json'
    def report(checked):
        return dict(task=task,split=split,part=part,checked=checked,total=len(paths),classes=dict(counts),failures=dict(failures),
            seconds=time.monotonic()-start,logical_bytes_read=total_bytes,signal_bytes_not_read=skipped_bytes)
    for i,p in enumerate(paths,1):
        try:
            stem,index=p.stem.rsplit('_' if task=='TUAB' else '-',1);index=int(index)
            source=CATALOG[task][stem]
            expected=source if task=='TUAB' else source[index]
            if LABELS_ONLY:
                from label_only_pickle import read_label_envelope
                body,skipped=read_label_envelope(p)
            else:
                body=p.read_bytes();skipped=0
            total_bytes+=len(body)-skipped;skipped_bytes+=skipped;sample=pickle.loads(body)
            value=np.asarray(sample['label'] if task=='TUEV' else sample['y'])
            if value.size!=1 or not np.isfinite(value).all() or float(value.ravel()[0])!=int(value.ravel()[0]):
                raise ValueError('Non-scalar, nonfinite or noninteger label')
            observed=int(value.ravel()[0]);counts[observed]+=1
            if observed!=expected:
                failures['raw_annotation_mismatch']+=1
                if len(examples)<100:examples.append(dict(file=p.name,expected=expected,observed=observed))
            file_hash=hashlib.sha256(body).hexdigest()
            digest.update((p.name+'\t'+str(observed)+'\t'+str(expected)+'\t'+file_hash+'\n').encode())
            seen[observed]+=1;group=sample_groups[observed]
            row=dict(task=task,split=split,file=p.name,label=observed,expected=expected)
            row['label_envelope_sha256' if LABELS_ONLY else 'source_sha256']=file_hash
            if len(group)<20:group.append(row)
            else:
                position=rng.randrange(seen[observed])
                if position<20:group[position]=row
        except Exception as exc:
            failures['read_or_schema_error']+=1
            if len(examples)<100:examples.append(dict(file=p.name,error=repr(exc)))
        if i%1000==0:save(progress,report(i))
    result=dict(**report(len(paths)),
        row_digest_sha256=digest.hexdigest(),error_examples=examples,
        cloud_samples=[row for rows in sample_groups.values() for row in rows])
    save(OUT/f'{task}-{split}-{part:04d}.json',result);save(progress,report(len(paths)))
    print('SPLIT_DONE='+json.dumps({k:v for k,v in result.items() if k not in ('cloud_samples','error_examples')}),flush=True)
    return result

def main():
    global OUT,CATALOG,LABELS_ONLY,DATA
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--labels-only',action='store_true')
    parser.add_argument('--catalog',type=Path)
    parser.add_argument('--data-root',type=Path)
    parser.add_argument('--tusl-folder',default='TUSL-full-20261111-r1')
    parser.add_argument('--workers',type=int,default=16)
    args=parser.parse_args();LABELS_ONLY=args.labels_only
    if args.data_root:DATA=args.data_root
    FOLDERS['TUSL']={s:args.tusl_folder+'/'+suffix for s,suffix in [('train','train'),('val','eval'),('test','test')]}
    OUT=args.out;OUT.mkdir(parents=True,exist_ok=False)
    if args.catalog:
        exported=json.loads(args.catalog.read_text());CATALOG=exported['catalog']
        ANNOTATION_HASHES.update(exported['annotation_hashes'])
    else:CATALOG=build_catalog()
    save(OUT/'annotation_hashes.json',ANNOTATION_HASHES)
    print('CATALOG='+json.dumps({task:len(rows) for task,rows in CATALOG.items()}),flush=True)
    jobs=[];splits=[]
    for task in FOLDERS:
        for split,folder in FOLDERS[task].items():
            paths=sorted((DATA/folder).glob('*.pkl'))
            expected_count=EXPECTED[task][('train','val','test').index(split)]
            splits.append(dict(task=task,split=split,total=len(paths),expected_count=expected_count,count_matches=len(paths)==expected_count))
            jobs.extend((task,split,part,paths[start:start+5000]) for part,start in enumerate(range(0,len(paths),5000)))
    # Interleave splits so a large dataset cannot delay every smaller audit.
    jobs.sort(key=lambda job:job[2])
    assert 1<=args.workers<=128
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:results=list(pool.map(scan,jobs))
    for row in splits:
        parts=[r for r in results if r['task']==row['task'] and r['split']==row['split']]
        counts=collections.Counter();failures=collections.Counter()
        for part in parts:counts.update(part['classes']);failures.update(part['failures'])
        row.update(checked=sum(r['checked'] for r in parts),classes=dict(counts),failures=dict(failures),
            logical_bytes_read=sum(r['logical_bytes_read'] for r in parts),
            signal_bytes_not_read=sum(r['signal_bytes_not_read'] for r in parts),error_examples=[e for p in parts for e in p['error_examples']][:100],
            chunk_row_digests=[p['row_digest_sha256'] for p in parts])
    summary=dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_root=str(DATA),
        total_checked=sum(r['checked'] for r in results),errors=sum(sum(r['failures'].values()) for r in results),
        all_counts_match=all(r['count_matches'] for r in splits),splits=splits,
        scope='All source pickle labels vs raw annotations or source directory/task labels. No EEG clinical relabelling.',
        labels_only_reader=LABELS_ONLY,
        source_catalog_sha256=hashlib.sha256(args.catalog.read_bytes()).hexdigest() if args.catalog else None,
        integrity_scope='Label-envelope digests only; large EEG byte buffers replaced in memory, input files read-only. Full EEG content/hash is not audited.' if LABELS_ONLY else 'Complete pickle bytes read and hashed.')
    save(OUT/'summary.json',summary)
    candidates=collections.defaultdict(list)
    for r in results:
        for sample in r['cloud_samples']:candidates[(sample['task'],sample['split'],sample['label'])].append(sample)
    rng=random.Random(20261111)
    save(OUT/('label_examples.private.json' if LABELS_ONLY else 'cloud_samples.json'),[sample for group in candidates.values() for sample in rng.sample(group,min(20,len(group)))])
    print('AUDIT_COMPLETE='+json.dumps(summary),flush=True)
    assert summary['total_checked']==660951 and summary['errors']==0 and summary['all_counts_match']

if __name__=='__main__':main()
