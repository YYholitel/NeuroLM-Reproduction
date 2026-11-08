"""Compare the compact VQ with its completed baseline; verify, audit, and export."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

ROOT=Path('/mnt/cfs-omni/lhy/neurolm-reproduction/vq-compact-e5-20261108-r1')
BASE=ROOT.parent/'vq-fixed-e5-20261108-r1'


def rows(path):
    if not path.exists():return []
    return [json.loads(line) for line in path.read_bytes().splitlines(keepends=True) if line.endswith(b'\n')]


def plot(root,base,metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(15,8),layout='constrained')
    baseline=rows(base/'run/metrics.jsonl')[:len(metrics)]
    for data,color,label,k in [(baseline,'#467b9c','8192 x 128, patience 100',8192),(metrics,'#db7d29','1024 x 64, patience 50',1024)]:
        x=(np.arange(len(data))+1)/897
        series=[[r[key] for r in data] for key in ['reconstruction','codebook','adversarial']]
        series += [[100*r['code_usage']['active_fraction'] for r in data],
                   [r['code_usage']['perplexity']/k for r in data],
                   [100*r['code_usage']['largest_code_fraction'] for r in data]]
        for ax,values in zip(axes.flat,series):
            values=np.array(values);assert np.isfinite(values).all()
            ax.plot(x,values,color=color,alpha=.12,lw=.5)
            smoothed=[values[max(0,i-99):i+1].mean() for i in range(len(values))]
            ax.plot(x,smoothed,color=color,lw=1.6,label=label)
    titles=['Valid-element reconstruction loss','Commitment loss','Adversarial loss','Per-batch active codes / capacity (%)','Per-batch perplexity / capacity','Largest code share per batch (%)']
    for ax,title in zip(axes.flat,titles):
        ax.set(title=title,xlabel='Epoch (897 updates)',xlim=(0,max(.5,len(metrics)/897)));ax.grid(alpha=.15)
    axes[1,0].set_ylim(0,100);axes[1,1].set_ylim(0,1);axes[0,0].legend(fontsize=8)
    fig.suptitle('NeuroLM VQ: capacity and dead-code policy comparison',fontsize=17)
    fig.supxlabel(f'{len(metrics)} / 4485 compact-run updates. Faint: individual batches; solid: trailing 100-update mean.\n'
                  'Same training data, order, batch and schedule. Three settings change together; not a one-factor ablation.',fontsize=10)
    output=root/'monitor/figures';output.mkdir(parents=True,exist_ok=True)
    for suffix in ['png','svg','pdf']:fig.savefig(output/f'vq_compact_comparison.{suffix}',dpi=170)
    plt.close(fig)


def finish(root,base):
    env=dict(os.environ,OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1')
    subprocess.run([sys.executable,str(root/'verify.py'),'--root',str(root)],check=True,env=env,stdout=(root/'verify.log').open('w'))
    subprocess.run([sys.executable,str(root/'audit.py'),'--root',str(root),'--checkpoint',str(root/'run/checkpoints/VQ/ckpt-4.pt'),
                    '--reference',str(root/'reference_audit.json'),'--output',str(root/'fixed_audit')],check=True,env=env,stdout=(root/'audit.log').open('w'))
    a=json.loads((base/'fixed_audit/audit.json').read_text());b=json.loads((root/'fixed_audit/audit.json').read_text())
    assert a['sample_names']==b['sample_names'] and a['valid']['tokens']==b['valid']['tokens']==4944
    comparison={label:dict(codebook_size=k,reconstruction=report['reconstruction'],commitment=report['commitment'],
                           **report['valid'],normalized_perplexity=report['valid']['perplexity']/k,
                           active_fraction=report['valid']['active_codes']/k)
                for label,report,k in [('baseline_8192x128',a,8192),('compact_1024x64',b,1024)]}
    (root/'fixed_comparison.json').write_text(json.dumps(comparison,indent=2))
    checkpoint=root/'run/checkpoints/VQ/ckpt-4.pt'
    with checkpoint.open('rb') as stream:checkpoint_sha=hashlib.file_digest(stream,'sha256').hexdigest()
    (root/'checkpoint_sha256.json').write_text(json.dumps(dict(final_checkpoint_sha256=checkpoint_sha)))
    mapping={name:name for name in ['verified.json','cpu_test.json','preflight.json','preflight_node_0.json','preflight_node_1.json','task.json','submitted.json','fixed_comparison.json','checkpoint_sha256.json']}
    mapping['code/bundle_manifest.json']='bundle_manifest.json';mapping['fixed_audit/audit.json']='fixed_audit.json'
    for name in ['metrics.jsonl','epoch_metrics.jsonl','completed.json','run_config.json','initial_state.jsonl','monitor_baseline.json']:mapping['run/'+name]=name
    for rank in [0,1]:
        for name in ['gpu_metrics.csv','exit_code.txt']:mapping[f'node-{rank}/{name}']=f'node_{rank}_{name}'
    for suffix in ['png','svg','pdf']:mapping[f'monitor/figures/vq_compact_comparison.{suffix}']=f'figures/vq_compact_comparison.{suffix}'
    hashes={target:hashlib.sha256((root/source).read_bytes()).hexdigest() for source,target in mapping.items()}
    (root/'artifact_sha256.json').write_text(json.dumps(hashes,indent=2))
    archive=root/'results_export.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for source,target in mapping.items():tar.add(root/source,arcname=target)
        tar.add(root/'artifact_sha256.json',arcname='artifact_sha256.json')
    report=dict(complete=True,archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),fixed_comparison=comparison)
    (root/'export_complete.json').write_text(json.dumps(report,indent=2));print('COMPACT_RESULTS_READY='+json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--once',action='store_true');a=p.parse_args()
    ROOT.joinpath('monitor').mkdir(exist_ok=True);last=-100
    while True:
        metrics=rows(ROOT/'run/metrics.jsonl');exits={str(i):(ROOT/f'node-{i}/exit_code.txt').read_text().strip() for i in [0,1] if (ROOT/f'node-{i}/exit_code.txt').exists()}
        complete=(ROOT/'run/completed.json').exists() and len(exits)==2
        latest=metrics[-1] if metrics else None
        report=dict(updates=len(metrics),complete=complete,exits=exits,latest=latest)
        (ROOT/'monitor/latest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
        if any(value!='0' for value in exits.values()):raise RuntimeError('Compact VQ worker failed')
        if metrics and (len(metrics)-last>=100 or complete):plot(ROOT,BASE,metrics);last=len(metrics)
        if complete:finish(ROOT,BASE);break
        if a.once:break
        time.sleep(60)
