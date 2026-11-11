"""Prepare a metadata-only patient split proposal; never move EEG or launch training."""
import argparse
import collections
import hashlib
import json
from pathlib import Path
import random

import numpy as np


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gold',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    rows=json.loads(args.gold.read_text())['records']
    groups=collections.defaultdict(list)
    for row in rows:groups[Path(row['source']).stem.split('_')[0]].append(row)
    patients=sorted(groups)
    assert len(patients)==38 and len(rows)==300
    # Columns: all windows, three labels, LE reference, 21-channel windows.
    features=np.array([[len(groups[g]),*[sum(r['label']==c for r in groups[g]) for c in range(3)],
                        sum(r['reference']=='LE' for r in groups[g]),sum(len(r['channels'])==21 for r in groups[g])]
                       for g in patients],dtype=float)
    fractions=np.array([.6,.2,.2])
    targets=fractions[:,None]*features.sum(axis=0)[None,:]
    weights=np.array([1,4,4,4,.5,.5])
    # Objective frozen before search; no model outputs or metrics are inputs.
    def score(order):
        members=[order[:23],order[23:30],order[30:]]
        sums=np.array([features[m].sum(axis=0) for m in members])
        value=float((((sums-targets)/np.maximum(targets,1))**2*weights).sum())
        for s in (1,2):
            concentration=features[members[s],0].max()/sums[s,0]
            value+=20*max(0.,concentration-.35)**2
        if (sums[:,1:4]<10).any() or (sums[:,4]<3).any():value+=1000
        return value
    rng=random.Random(20261111)
    base=list(range(38));best=None;best_score=float('inf')
    # Fixed search budget, retaining the first equally scored candidate.
    for _ in range(10000):
        order=base.copy();rng.shuffle(order);value=score(order)
        if value<best_score:best,best_score=order,value
    for _ in range(30000):
        candidate=best.copy();i,j=rng.sample(range(38),2);candidate[i],candidate[j]=candidate[j],candidate[i]
        value=score(candidate)
        if value<best_score:best,best_score=candidate,value
    sets=[set(patients[i] for i in best[:23]),set(patients[i] for i in best[23:30]),set(patients[i] for i in best[30:])]
    assert all(not sets[i]&sets[j] for i in range(3) for j in range(i+1,3))
    assignment={g:s for s,group in zip(('train','val','test'),sets) for g in group}
    manifest=[dict(source_file=r['file'],new_split=assignment[Path(r['source']).stem.split('_')[0]],
                   patient=Path(r['source']).stem.split('_')[0],label=r['label'],reference=r['reference'],
                   channels=len(r['channels'])) for r in rows]
    assert len(manifest)==len(set(r['source_file'] for r in manifest))==300
    result={'status':'PROPOSAL_ONLY_NOT_APPLIED','seed':20261111,'windows_retained':300,'patients':38,
            'patient_overlap':0,'gold_sha256':hashlib.sha256(args.gold.read_bytes()).hexdigest(),
            'method':'Fixed 23/7/8 patient groups; 10000 random candidates then 30000 pair swaps; metadata-only objective.',
            'objective_score':best_score,'splits':{},
            'weight_reuse_warning':'If old training patients enter new validation/test, restart the adapted stage 2 from the original official checkpoint. This only prevents leakage from our adaptation; official checkpoint provenance still applies.'}
    for split in ('train','val','test'):
        selected=[r for r in manifest if r['new_split']==split]
        counts=collections.Counter(r['patient'] for r in selected)
        result['splits'][split]={'windows':len(selected),'patients':len(counts),
            'classes':dict(collections.Counter(r['label'] for r in selected)),
            'references':dict(collections.Counter(r['reference'] for r in selected)),
            'channels':dict(collections.Counter(r['channels'] for r in selected)),
            'largest_patient_windows':max(counts.values()),'largest_patient_fraction':max(counts.values())/len(selected),
            'previous_train_windows':sum(r['source_file'].startswith('train/') for r in selected)}
    args.out.mkdir(parents=True,exist_ok=False)
    (args.out/'manifest.private.json').write_text(json.dumps(manifest,indent=2))
    result['private_manifest_sha256']=hashlib.sha256((args.out/'manifest.private.json').read_bytes()).hexdigest()
    (args.out/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
