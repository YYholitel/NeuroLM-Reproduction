import json,time,csv,io
from pathlib import Path
root=Path('/mnt/cfs-omni/lhy/neurolm-reproduction/vq-five-full-20261107-r1')
paths=list((root/'assets/shard_receipts').glob('*.json'))
rows=[json.loads(p.read_text()) for p in paths]
counts={};compressed=0;textbytes=0
for r in rows:
    name=r['id'].split('-')[0]
    counts[name]=counts.get(name,0)+r.get('samples',0)
    compressed+=r['compressed_bytes']
    if name=='text': textbytes+=r['uncompressed_bytes']
print(json.dumps({'time':time.time(),'groups':len(rows),'groups_total':3361,'samples':counts,
                  'compressed_GiB':compressed/1024**3,'text_GiB':textbytes/1024**3,
                  'first_receipt_time':min((p.stat().st_mtime for p in paths),default=None),
                  'last_receipt_time':max((p.stat().st_mtime for p in paths),default=None)}))
for name in ['submitted.json','assets/data_ready.json','pipeline_state.json','run/completed.json']:
    p=root/name
    if p.exists():
        data=json.loads(p.read_text()); data.pop('datasets',None)
        print(json.dumps({'file':name,'data':data}))
metrics=root/'run/metrics.jsonl'
if metrics.exists():
    lines=metrics.read_text().splitlines()
    print(json.dumps({'metric_rows':len(lines),'first':json.loads(lines[0]),'last':json.loads(lines[-1])}))
epochs=root/'run/epoch_metrics.jsonl'
if epochs.exists():
    lines=epochs.read_text().splitlines()
    if lines: print(json.dumps({'completed_epoch_reports':len(lines),'last_epoch_report':json.loads(lines[-1])}))
for node in ['node-0','node-1']:
    path=root/'run'/node/'gpu_metrics.csv'
    if not path.exists(): continue
    with path.open('rb') as stream:
        header=stream.readline();start=max(len(header),path.stat().st_size-128*1024)
        stream.seek(start)
        if start>len(header): stream.readline()
        rows=list(csv.DictReader(io.StringIO((header+stream.read()).decode()),skipinitialspace=True))
    devices={}
    for row in rows:
        try: index=int(row['gpu_index'])
        except (KeyError,ValueError): continue
        devices.setdefault(index,[]).append(row)
    result=[]
    for index,history in sorted(devices.items()):
        recent=history[-12:];last=recent[-1]
        def number(field):
            try: return float(last[field])
            except (ValueError,TypeError,KeyError): return None
        values=[]
        for row in recent:
            try: values.append(float(row['gpu_util_pct']))
            except (ValueError,TypeError,KeyError): pass
        result.append({'index':index,'samples':len(recent),'node_timestamp':last.get('timestamp'),
                       'gpu_util_pct':number('gpu_util_pct'),'recent_mean_util_pct':sum(values)/len(values) if values else None,
                       'memory_used_MiB':number('memory_used_MiB'),'memory_total_MiB':number('memory_total_MiB'),
                       'power_W':number('power_W')})
    print(json.dumps({'gpu_monitor_node':node,'sample_interval_seconds':5,'gpus':result}))
