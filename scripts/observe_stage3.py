"""Read worker outputs; write plots only to the Notebook-owned monitor directory."""
import datetime
import json
import math
from pathlib import Path

ROOT=Path('/mnt/cfs-omni/lhy/neurolm-reproduction/stage3-new-pretrain-20261109-r1')
out=ROOT/'monitor';out.mkdir(exist_ok=True)
rows=[];path=ROOT/'train/train_steps.jsonl'
if path.exists():
    for line in path.read_text().splitlines():
        try:rows.append(json.loads(line))
        except json.JSONDecodeError:break
status=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),logged_training_steps=len(rows),
            last_logged=rows[-1] if rows else None,
            checkpoint_audited=(ROOT/'checkpoint_audit.json').exists(),
            completed_comparison=(ROOT/'comparison.json').exists(),evaluation={})
for variant in ('new','official'):
    results=[]
    for p in sorted((ROOT/f'eval-{variant}').glob('*-metrics.json')):
        d=json.loads(p.read_text());results.append(dict(task=d['task'],split=d['split'],metrics=d['metrics']))
    status['evaluation'][variant]=results
if rows:
    assert all(math.isfinite(r[k]) for r in rows for k in ('instruction_loss','text_loss','lr'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
    for ax,key,label in zip(axes,['instruction_loss','text_loss','lr'],['Instruction CE','Text CE','Learning rate']):
        ax.plot([r['step'] for r in rows],[r[key] for r in rows],linewidth=1)
        ax.set(title=label,xlabel='Microstep');ax.grid(alpha=.25)
    fig.suptitle('NeuroLM-B stage 3 | new VQ + 20-epoch pretraining | matched 1-epoch instruction tuning')
    fig.savefig(out/'training_curves.png',dpi=160);plt.close(fig)
temporary=out/'status.json.partial';temporary.write_text(json.dumps(status,indent=2));temporary.replace(out/'status.json')
print(json.dumps(status),flush=True)
