"""Create full-file-hash evidence for up to 20 files per raw class and split."""
import argparse
import collections
import hashlib
import json
import pickle
import random

import numpy as np

import audit_all_labels as audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    args=parser.parse_args()
    catalog=audit.build_catalog();samples=[];rng=random.Random(20261111)
    for task in audit.FOLDERS:
        for split,folder in audit.FOLDERS[task].items():
            candidates=collections.defaultdict(list)
            for path in (audit.DATA/folder).glob('*.pkl'):
                stem,index=path.stem.rsplit('_' if task=='TUAB' else '-',1)
                expected=catalog[task][stem]
                if task!='TUAB':expected=expected[int(index)]
                candidates[expected].append(path)
            for label,paths in sorted(candidates.items()):
                for path in rng.sample(sorted(paths),min(20,len(paths))):
                    body=path.read_bytes();sample=pickle.loads(body)
                    observed=int(np.asarray(sample['label'] if task=='TUEV' else sample['y']).item())
                    assert observed==label
                    samples.append(dict(task=task,split=split,file=path.name,label=observed,
                        expected=label,source_sha256=hashlib.sha256(body).hexdigest()))
    audit.Path(args.out).write_text(json.dumps(samples,indent=2))
    print(json.dumps({'samples':len(samples),'tasks':dict(collections.Counter(r['task'] for r in samples))}))


if __name__=='__main__':main()
