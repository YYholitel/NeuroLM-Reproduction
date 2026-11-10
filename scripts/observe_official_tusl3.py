"""Print live progress and export per-seed curves without touching worker outputs."""
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]


def rows(path):
    if not path.exists():return []
    found=[]
    for line in path.read_text().splitlines():
        try:found.append(json.loads(line))
        except json.JSONDecodeError:pass  # Worker may be appending the final line.
    return found


def main():
    output=ROOT/'monitor';output.mkdir(exist_ok=True)
    figure,axes=plt.subplots(2,2,figsize=(12,8));summaries=[];evaluation=[]
    for seed in (42,43,44):
        base=ROOT/f'seed-{seed}';s2=rows(base/'stage2/run/metrics.jsonl');s3=rows(base/'stage3/train/train_steps.jsonl')
        summary=dict(seed=seed,stage2_steps=len(s2),stage2_epochs=len(rows(base/'stage2/run/epochs.jsonl')),
                     stage2_latest=s2[-1] if s2 else None,stage3_latest=s3[-1] if s3 else None)
        summaries.append(summary)
        for data,xkey,ykey,ax,title in [(s2,'step','eeg_loss',axes[0,0],'Stage 2 EEG loss'),
            (s2,'step','text_loss',axes[0,1],'Stage 2 text loss'),
            (s3,'iter_num','instruction_loss',axes[1,0],'Stage 3 instruction loss'),
            (s3,'iter_num','text_loss',axes[1,1],'Stage 3 text loss')]:
            ax.set_title(title);ax.set_xlabel('Optimizer update' if data is s2 else 'Microstep');ax.set_ylabel('Loss');ax.grid(alpha=.2)
            if data:
                width=100 if data is s2 else 20
                blocks=[data[i:i+width] for i in range(0,len(data),width)]
                ax.plot([b[-1][xkey] for b in blocks],[statistics.mean(p[ykey] for p in b) for b in blocks],label=f'Seed {seed}')
        path=base/'stage3/eval/summary.json';audit=base/'stage3/eval_audit.json'
        if path.exists() and audit.exists() and json.loads(audit.read_text()).get('status')=='passed':
            evaluation.append(dict(seed=seed,**json.loads(path.read_text())))
    for index,ax in enumerate(axes.flat):
        if ax.lines:ax.legend()
        else:ax.text(.5,.5,'Awaiting stage 2 completion' if index>=2 else 'Awaiting training metrics',
                     transform=ax.transAxes,ha='center',va='center',color='gray')
    figure.suptitle('Official NeuroLM initialization; full TUSL; 100/20-log-block means')
    figure.tight_layout();figure.savefig(output/'training_curves.png',dpi=170);plt.close(figure)
    report=dict(seeds=summaries,completed_evaluations=len(evaluation))
    if len(evaluation)==3:
        comparisons=[]
        for task in ('TUAB','TUEV','HMC','Workload','TUSL'):
            per_seed={e['seed']:next(x['metrics'] for x in e['results'] if x['task']==task and x['split']=='test') for e in evaluation}
            aggregate={}
            for metric in ['balanced_accuracy','accuracy','weighted_f1','cohen_kappa','roc_auc']:
                vals=[m[metric] for m in per_seed.values()]
                if all(v is not None for v in vals):aggregate[metric]=dict(mean=statistics.mean(vals),sample_sd=statistics.stdev(vals))
            comparisons.append(dict(task=task,per_seed=per_seed,aggregate=aggregate))
        report['test_results']=comparisons
    (output/'progress.json').write_text(json.dumps(report,indent=2))
    print('PROGRESS='+json.dumps(report),flush=True)


if __name__=='__main__':main()
