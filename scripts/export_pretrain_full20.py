"""Export non-secret evidence only; EEG samples, model weights and credentials stay remote."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

p=argparse.ArgumentParser(__doc__);p.add_argument('--phase',choices=['startup','final'],required=True);args=p.parse_args()
ROOT=Path('/mnt/cfs-omni/lhy/neurolm-reproduction/pretrain-full20-20261109-r1')
out=ROOT/('export_'+args.phase);out.mkdir(exist_ok=False)
files=['task.json','submitted.json','resource_capacity.json','preflight.json','preflight_node_0.json','preflight_node_1.json',
       'cpu_test.json','cpu_sample_source.json','source_manifest.json','code/bundle_manifest.json',
       'platform_status.json','run/run_config.json','run/initial_state.json','run/frozen_health.json','run/epochs.jsonl',
       'monitor/first_15min_audit.json','monitor/first_15min_metrics.jsonl','monitor/first_15min_frozen_checks.jsonl',
       'monitor/live_health.json','monitor/pretrain_curves.png']
if args.phase=='final':
    files+=['platform_final_status.json','run/completed.json','monitor/final_checkpoint.json',
            'node-0/exit_code.txt','node-1/exit_code.txt','run/metrics.jsonl',
            'monitor/final_audit_20261109.json','monitor/result_query_20261109.json']
for name in files:
    source=ROOT/name;target=out/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,target)
log=(ROOT/'run/pretrain.log').read_text().splitlines()
selected=[line for line in log if line.startswith(('PRETRAIN_WORKER','PRETRAIN_PREFLIGHT','PRETRAIN_STEP=','PRETRAIN_EPOCH_COMPLETE='))]
(out/'training_log_snapshot.txt').write_text('\n'.join(selected[:30]+['[middle omitted]']+selected[-100:])+'\n')
manifest={str(f.relative_to(out)):hashlib.sha256(f.read_bytes()).hexdigest() for f in out.rglob('*') if f.is_file()}
(out/'artifact_sha256.json').write_text(json.dumps(manifest,indent=2))
archive=ROOT/('evidence_'+args.phase+'.tar.gz')
with tarfile.open(archive,'w:gz') as tar:tar.add(out,arcname='.')
print(json.dumps(dict(phase=args.phase,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=len(manifest),
                     archive=str(archive),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),bytes=archive.stat().st_size)))
