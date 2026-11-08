"""Read progress of the five-epoch repaired VQ task."""
import datetime,json
from pathlib import Path
root=Path('/mnt/cfs-omni/lhy/neurolm-reproduction/vq-fixed-e5-20261108-r1')
def rows(path):
    if not path.exists():return []
    return [json.loads(line) for line in path.read_bytes().splitlines(keepends=True)
            if line.endswith(b'\n') and line.strip()]
metrics=rows(root/'run/metrics.jsonl');epochs=rows(root/'run/epoch_metrics.jsonl')
report=dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            updates=len(metrics),target_updates=4485,completed=(root/'run/completed.json').exists(),
            latest=metrics[-1] if metrics else None,
            epochs=[dict({k:r[k] for k in ['epoch','updates','mean','tail_code_usage','lr_last','review_flags']},
                         dead_code_resets=sum(m['dead_code_resets'] for m in metrics if m['epoch']==r['epoch']-1))
                    for r in epochs],nodes={})
for rank in range(2):
    node=root/f'node-{rank}';item={}
    if (node/'exit_code.txt').exists():item['exit_code']=(node/'exit_code.txt').read_text().strip()
    if not metrics or item.get('exit_code','0')!='0':
        if (node/'worker.log').exists():
            with (node/'worker.log').open('rb') as stream:
                stream.seek(max(0,(node/'worker.log').stat().st_size-10000))
                item['tail']=stream.read().decode(errors='replace').splitlines()[-8:]
    if (node/'gpu_metrics.csv').exists():
        with (node/'gpu_metrics.csv').open('rb') as stream:
            stream.seek(max(0,(node/'gpu_metrics.csv').stat().st_size-8000))
            item['gpu_last8']=stream.read().decode().splitlines()[-8:]
    report['nodes'][rank]=item
print(json.dumps(report))
