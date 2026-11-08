"""Retrospectively audit the first 15 minutes from timestamped training/GPU logs."""
import csv
import datetime as dt
import json
import math
from pathlib import Path
import statistics

ROOT=Path('/mnt/cfs-omni/lhy/neurolm-reproduction/pretrain-full20-20261109-r1')
out=ROOT/'monitor';out.mkdir(exist_ok=True)
all_rows=[json.loads(line) for line in (ROOT/'run/metrics.jsonl').read_text().splitlines()]
start=dt.datetime.fromisoformat(all_rows[0]['utc']);end=start+dt.timedelta(minutes=15)
rows=[row for row in all_rows if dt.datetime.fromisoformat(row['utc'])<=end]
assert dt.datetime.fromisoformat(all_rows[-1]['utc'])>=end
health=[]
log=(ROOT/'run/pretrain.log').read_text()
for line in log.splitlines():
    if line.startswith('PRETRAIN_FROZEN_HEALTH='):
        item=json.loads(line.split('=',1)[1])
        if dt.datetime.fromisoformat(item['utc'])<=end:health.append(item)
result=dict(audit_kind='retrospective first 15 minutes; live plotting failed due output directory permissions',
            audited_utc=dt.datetime.now(dt.timezone.utc).isoformat(),start_utc=start.isoformat(),end_utc=end.isoformat(),
            updates=len(rows),first=rows[0],last=rows[-1],
            continuous_steps=[r['step'] for r in rows]==list(range(1,len(rows)+1)),
            all_finite=all(math.isfinite(r[k]) for r in rows for k in ('eeg_loss','text_loss','lr','grad_norm')),
            frozen_checks=len(health),last_frozen_check=health[-1],
            all_frozen_checks_passed=all(h['frozen_state_unchanged'] and h['gpt_weights_updated'] for h in health))
for key in ('eeg_loss','text_loss'):
    result[key]=dict(first_20_mean=statistics.mean(r[key] for r in rows[:20]),
                     last_20_mean=statistics.mean(r[key] for r in rows[-20:]))
result['gpu']={}
for node in (0,1):
    devices={}
    with (ROOT/f'node-{node}/gpu_metrics.csv').open() as stream:
        for row in csv.DictReader(stream):
            timestamp=dt.datetime.strptime(row['timestamp'],'%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=dt.timezone(dt.timedelta(hours=8)))
            if start+dt.timedelta(minutes=10)<=timestamp<=end:
                devices.setdefault(row['gpu_uuid'].strip(),[]).append(row)
    assert len(devices)==8
    for uuid,observations in devices.items():
        memory=[float(r['memory_used_MiB']) for r in observations]
        utilization=[float(r['gpu_util_pct']) for r in observations]
        result['gpu'][uuid]=dict(node=node,samples=len(observations),memory_min_MiB=min(memory),memory_max_MiB=max(memory),
                                utilization_mean_pct=statistics.mean(utilization),utilization_max_pct=max(utilization))
result['passed']=bool(result['continuous_steps'] and result['all_finite'] and result['all_frozen_checks_passed']
    and len(health)>0 and len(result['gpu'])==16
    and all(result[k]['last_20_mean']<result[k]['first_20_mean'] for k in ('eeg_loss','text_loss')))
(out/'first_15min_audit.json').write_text(json.dumps(result,indent=2))
(out/'first_15min_metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
(out/'first_15min_frozen_checks.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in health))
print(json.dumps({k:v for k,v in result.items() if k not in ('gpu','last_frozen_check')}),flush=True)
assert result['passed']
