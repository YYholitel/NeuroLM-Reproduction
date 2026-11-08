"""Read-only progress, GPU and frozen-state checks; no downstream evaluation."""
import csv
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT=Path('/mnt/cfs-omni/lhy/neurolm-reproduction/pretrain-full20-20261109-r1')
run=ROOT/'run'
report_dir=ROOT/'monitor'
report_dir.mkdir(exist_ok=True)
def rows(path):
    result=[]
    if path.exists():
        for line in path.read_text().splitlines():
            try:result.append(json.loads(line))
            except json.JSONDecodeError:break  # trailing partial write
    return result
def write(path,value):
    temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,indent=2));temporary.replace(path)
def main():
    now=dt.datetime.now(dt.timezone.utc);metrics=rows(run/'metrics.jsonl')
    result=dict(observed_utc=now.isoformat(),steps=len(metrics),planned_steps=19140,epochs_planned=20)
    if not metrics:
        print(json.dumps(result));return
    elapsed=(now-dt.datetime.fromisoformat(metrics[0]['utc'])).total_seconds()
    stale=(now-dt.datetime.fromisoformat(metrics[-1]['utc'])).total_seconds()
    result.update(elapsed_seconds_since_first_update=elapsed,last_update_age_seconds=stale,latest=metrics[-1],
                  completed_epochs=len(rows(run/'epochs.jsonl')),
                  all_metrics_finite=all(math.isfinite(row[key]) for row in metrics for key in ('eeg_loss','text_loss','grad_norm','lr')),
                  continuous_steps=[r['step'] for r in metrics]==list(range(1,len(metrics)+1)))
    for key in ('eeg_loss','text_loss'):
        result[key]=dict(first_20_mean=statistics.mean(r[key] for r in metrics[:20]),
                         last_20_mean=statistics.mean(r[key] for r in metrics[-20:]))
    result['losses_decreased']=all(result[k]['last_20_mean']<result[k]['first_20_mean'] for k in ('eeg_loss','text_loss'))
    health=run/'frozen_health.json'
    result['frozen_check']=json.loads(health.read_text()) if health.exists() else None
    result['nodes']={}
    for rank in (0,1):
        folder=ROOT/f'node-{rank}';path=folder/'gpu_metrics.csv'
        latest={}
        if path.exists():
            with path.open() as f:
                for row in csv.DictReader(f):latest[row['gpu_uuid']]=row
        result['nodes'][str(rank)]=dict(gpus=list(latest.values()),
            exit_code=(folder/'exit_code.txt').read_text().strip() if (folder/'exit_code.txt').exists() else None)
    result['input_stream_steps']={str(rank):len(rows(run/f'input_stream_rank-{rank}.jsonl')) for rank in range(16)}
    result['healthy_15min']=bool(elapsed>=900 and stale<120 and result['all_metrics_finite'] and result['continuous_steps']
        and result['losses_decreased'] and result['frozen_check'] and result['frozen_check']['frozen_state_unchanged']
        and result['frozen_check']['gpt_weights_updated'] and result['frozen_check']['step']>=100
        and all(len(n['gpus'])==8 and n['exit_code'] is None for n in result['nodes'].values()))
    write(report_dir/'live_health.json',result)
    if result['healthy_15min'] and elapsed<=1020 and not (report_dir/'health_15min.json').exists():
        write(report_dir/'health_15min.json',result)
        (report_dir/'health_15min_log.txt').write_text('\n'.join((run/'pretrain.log').read_text().splitlines()[-180:])+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    for ax,key,title in zip(axes,['eeg_loss','text_loss','lr'],['EEG next-second CE','Text next-token CE','Learning rate']):
        ax.plot([r['step'] for r in metrics],[r[key] for r in metrics],linewidth=1,alpha=.7)
        ax.set(title=title,xlabel='Optimizer update');ax.grid(alpha=.25)
        if key=='text_loss':ax.set_yscale('log')
    fig.suptitle(f'NeuroLM-B stage 2 | {len(metrics):,} / 19,140 updates | 16 H20 | frozen 8192 x 128 VQ')
    fig.savefig(report_dir/'pretrain_curves.png',dpi=170);plt.close(fig)
    if (run/'completed.json').exists():
        done=json.loads((run/'completed.json').read_text())
        checkpoint=run/'checkpoints/NeuroLM-B/ckpt.pt'
        assert done['steps']==19140 and done['epochs_completed']==20
        if not (report_dir/'final_checkpoint.json').exists():
            with checkpoint.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            write(report_dir/'final_checkpoint.json',dict(path=str(checkpoint),sha256=digest,bytes=checkpoint.stat().st_size))
    print(json.dumps({k:v for k,v in result.items() if k not in ('nodes','frozen_check','input_stream_steps')}),flush=True)
if __name__=='__main__':main()
