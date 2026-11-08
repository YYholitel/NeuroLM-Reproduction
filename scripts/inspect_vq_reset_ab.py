"""Read the A/B run's progress without modifying training."""
import datetime
import json
from pathlib import Path

root=Path('/mnt/cfs-omni/lhy/neurolm-reproduction/vq-reset-ab-500-20261108-r1')
report={'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'arms':{},'nodes':{}}
for arm in ['reset_off','reset_on']:
    path=root/'runs'/arm
    metrics=path/'metrics.jsonl'
    rows=[]
    if metrics.exists():
        for line in metrics.read_text().splitlines():
            try: rows.append(json.loads(line))
            except json.JSONDecodeError: pass
    data={'updates':len(rows),'complete':(path/'completed.json').exists()}
    if rows:
        data['last']=rows[-1]
        data['reset_total']=sum(r['dead_code_resets'] for r in rows)
    report['arms'][arm]=data
for rank in range(2):
    path=root/f'node-{rank}'
    data={}
    if (path/'exit_code.txt').exists(): data['exit_code']=(path/'exit_code.txt').read_text().strip()
    if (path/'worker.log').exists(): data['tail']=(path/'worker.log').read_text(errors='replace').splitlines()[-5:]
    if (path/'gpu_metrics.csv').exists(): data['gpu_last8']=(path/'gpu_metrics.csv').read_text().splitlines()[-8:]
    report['nodes'][rank]=data
report['comparison_verified']=(root/'comparison_verified.json').exists()
print(json.dumps(report,indent=2))
