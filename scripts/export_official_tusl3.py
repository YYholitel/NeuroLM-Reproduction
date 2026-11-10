"""Export compact, credential-free experiment evidence for review."""
import argparse
from collections import Counter
import csv
import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil


def read(path):return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();r=a.root;out=a.out;out.mkdir(parents=True,exist_ok=False)
    for name in ['preflight.json','build.json','indexed_tusl_audit.json','pipeline_status.json','assets/data_ready.json','code/bundle_manifest.json','author/bundle_manifest.json']:
        src=r/name;dst=out/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,dst)
    tusl=read(r/'data/TUSL/manifest.json');summary={k:v for k,v in tusl.items() if k!='records'}
    group=Counter()
    for row in tusl['records']:group[(row['split'],row['reference'],len(row['ch_names']))]+=1
    summary['record_groups']=[dict(split=k[0],reference=k[1],channels=k[2],records=v) for k,v in sorted(group.items())]
    (out/'tusl_summary.json').write_text(json.dumps(summary,indent=2))
    health=[];all_gpu_ids=set()
    for seed in [42,43,44]:
        job=r/f'seed-{seed}/stage2';dest=out/f'seed-{seed}/stage2';dest.mkdir(parents=True)
        for name in ['task.json','platform_status.json','platform_submit.json','run/run_config.json','run/initialization.json',
                     'run/initial_state.json','run/frozen_health.json','run/epochs.jsonl','run/completed.json','checkpoint_audit.json']:
            if (job/name).exists():
                dst=dest/name;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(job/name,dst)
        path=job/'run/metrics.jsonl'
        metrics=[json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []
        if metrics:(dest/'metrics.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in metrics))
        gpu=[];ids=set()
        for path in sorted(job.glob('node-*/gpu_metrics.csv')):
            with path.open() as f:rows=list(csv.DictReader(f))
            for row in rows:
                if row.get('gpu_uuid'):ids.add(row['gpu_uuid'].strip())
            if rows:gpu.extend(rows[-8*12:])
        all_gpu_ids.update(ids)
        health.append(dict(seed=seed,steps=len(metrics),all_logged_losses_finite=bool(metrics) and
            all(math.isfinite(row[k]) for row in metrics for k in ['eeg_loss','text_loss','grad_norm']),
            first=metrics[0] if metrics else None,last=metrics[-1] if metrics else None,
            distinct_gpus=len(ids),gpu_recent_mean_util_pct=sum(float(x['gpu_util_pct']) for x in gpu)/len(gpu) if gpu else None,
            max_recent_memory_MiB=max((float(x['memory_used_MiB']) for x in gpu),default=None),
            frozen_check=read(job/'run/frozen_health.json') if (job/'run/frozen_health.json').exists() else None))
    (out/'health.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        distinct_gpus=len(all_gpu_ids),seeds=health),indent=2))
    if (r/'monitor/training_curves.png').exists():shutil.copyfile(r/'monitor/training_curves.png',out/'training_curves.png')
    hashes={str(path.relative_to(out)):hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(out.rglob('*')) if path.is_file()}
    (out/'sha256.json').write_text(json.dumps(hashes,indent=2))
    print(json.dumps(dict(ok=True,files=len(hashes),seeds=[dict(seed=x['seed'],steps=x['steps'],gpus=x['distinct_gpus']) for x in health])),flush=True)


if __name__=='__main__':main()
