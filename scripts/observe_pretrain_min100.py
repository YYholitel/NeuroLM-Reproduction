"""Monitor and export the bounded stage-two run; never submits another job."""
import argparse
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import tarfile
import time

ROOT=Path('/mnt/cfs-omni/lhy/neurolm-reproduction/pretrain-min100-20261108-r2')


def rows(path):
    if not path.exists():return []
    return [json.loads(line) for line in path.read_bytes().splitlines(keepends=True) if line.endswith(b'\n')]


def plot(metrics,validation,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    for ax,key,title in zip(axes.flat,['eeg_loss','text_loss','eeg_accuracy','text_accuracy'],
                            ['EEG next-second cross entropy','Text next-token cross entropy','EEG token accuracy','Text token accuracy']):
        if metrics:ax.plot([r['step'] for r in metrics],[r[key] for r in metrics],color='#337c9b',alpha=.8,label='Training batch')
        if validation:ax.plot([r['step'] for r in validation],[r[key] for r in validation],'o-',color='#d97727',label='Fixed held-out probe')
        ax.set(title=title,xlabel='Optimizer update');ax.grid(alpha=.18);ax.legend(fontsize=9)
    fig.suptitle('NeuroLM stage two: GPT-2 + repaired 5-epoch VQ',fontsize=17)
    fig.supxlabel('100-update warmup experiment; global EEG/text batch 480/32; VQ frozen.\nBalanced held-out probe: 32 windows per dataset (160 total). Not downstream task evaluation.',fontsize=10)
    out.mkdir(parents=True,exist_ok=True)
    for suffix in ['png','svg','pdf']:fig.savefig(out/f'pretrain_min100_curves.{suffix}',dpi=170)
    plt.close(fig)


def observe(root):
    metrics=rows(root/'run/metrics.jsonl');validation=rows(root/'run/validation.jsonl')
    exits={str(rank):(root/f'node-{rank}/exit_code.txt').read_text().strip()
           for rank in [0,1] if (root/f'node-{rank}/exit_code.txt').exists()}
    state=dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),updates=len(metrics),
               latest=metrics[-1] if metrics else None,validation=validation,exits=exits,
               completed=(root/'run/completed.json').exists())
    (root/'monitor').mkdir(exist_ok=True)
    temp=root/'monitor/latest.partial';temp.write_text(json.dumps(state,indent=2));os.replace(temp,root/'monitor/latest.json')
    if validation:plot(metrics,validation,root/'monitor/figures')
    print(json.dumps(state),flush=True)
    if any(value!='0' for value in exits.values()):raise RuntimeError('Stage-two worker failed')
    if not state['completed'] or len(exits)!=2:return False
    assert len(metrics)==100 and [r['step'] for r in metrics]==list(range(1,101))
    assert [r['step'] for r in validation]==[0,50,100]
    for row in metrics+validation:
        assert all(math.isfinite(row[key]) for key in ['eeg_loss','text_loss','eeg_accuracy','text_accuracy'])
    done=json.loads((root/'run/completed.json').read_text())
    assert done['steps']==100 and done['frozen_state_unchanged'] and done['gpt_weights_updated']
    for rank in range(16):assert len(rows(root/f'run/input_stream_rank-{rank}.jsonl'))==100
    ckpt=root/'run/checkpoints/NeuroLM-B/ckpt.pt'
    with ckpt.open('rb') as stream:checkpoint_sha=hashlib.file_digest(stream,'sha256').hexdigest()
    verified=dict(complete=True,steps=100,world_size=16,partial_epoch=True,checkpoint_sha256=checkpoint_sha,
                  frozen_state_unchanged=True,gpt_weights_updated=True,validation=validation,
                  training_first10={key:sum(r[key] for r in metrics[:10])/10 for key in ['eeg_loss','text_loss']},
                  training_last10={key:sum(r[key] for r in metrics[-10:])/10 for key in ['eeg_loss','text_loss']})
    (root/'verified.json').write_text(json.dumps(verified,indent=2))
    mapping={name:name for name in ['verified.json','cpu_test.json','preflight.json','preflight_node_0.json','preflight_node_1.json','submitted.json','task.json']}
    mapping['code/bundle_manifest.json']='bundle_manifest.json'
    mapping['probe/validation_manifest.json']='validation_manifest.json'
    mapping['gpt2/verified.json']='gpt2_verified.json'
    for name in ['metrics.jsonl','validation.jsonl','run_config.json','initial_state.json','completed.json']:mapping['run/'+name]=name
    for rank in [0,1]:
        for name in ['gpu_metrics.csv','exit_code.txt']:mapping[f'node-{rank}/{name}']=f'node_{rank}_{name}'
    for suffix in ['png','svg','pdf']:mapping[f'monitor/figures/pretrain_min100_curves.{suffix}']=f'figures/pretrain_min100_curves.{suffix}'
    hashes={target:hashlib.sha256((root/source).read_bytes()).hexdigest() for source,target in mapping.items()}
    (root/'artifact_sha256.json').write_text(json.dumps(hashes,indent=2))
    archive=root/'results_export.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for source,target in mapping.items():tar.add(root/source,arcname=target)
        tar.add(root/'artifact_sha256.json',arcname='artifact_sha256.json')
    print('PRETRAIN_RESULTS_READY='+json.dumps(dict(verified=verified,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())),flush=True)
    return True


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--once',action='store_true');parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args()
    while True:
        if observe(args.root) or args.once:break
        time.sleep(60)
