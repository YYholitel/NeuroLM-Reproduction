"""Evaluate the completed baseline now and compact VQ after five complete epochs."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

base=Path('/mnt/cfs-omni/lhy/neurolm-reproduction');root=base/'vq-compact-e5-20261108-r1'
validation=base/'pretrain-min100-20261108-r2/probe/validation_manifest.json'
env=dict(os.environ,OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1')
for label,run in [('baseline',base/'vq-fixed-e5-20261108-r1'),('compact',root)]:
 while not (run/'run/completed.json').exists() or not all((run/f'node-{i}/exit_code.txt').exists() for i in [0,1]):
  if any((run/f'node-{i}/exit_code.txt').exists() and (run/f'node-{i}/exit_code.txt').read_text().strip()!='0' for i in [0,1]):raise RuntimeError('Worker failed')
  time.sleep(60)
 assert all((run/f'node-{i}/exit_code.txt').read_text().strip()=='0' for i in [0,1])
 subprocess.run([sys.executable,str(root/'audit_heldout.py'),'--runtime',str(run/'code'),'--checkpoint',str(run/'run/checkpoints/VQ/ckpt-4.pt'),
                 '--validation-manifest',str(validation),'--output',str(root/f'heldout_{label}.json')],env=env,check=True)
reports={label:json.loads((root/f'heldout_{label}.json').read_text()) for label in ['baseline','compact']}
assert reports['baseline']['validation_manifest_sha256']==reports['compact']['validation_manifest_sha256']
assert [r['name'] for r in reports['baseline']['sample_metrics']]==[r['name'] for r in reports['compact']['sample_metrics']]
summary={label:{key:report[key] for key in ['codebook_shape','overall','per_dataset']} for label,report in reports.items()}
(root/'heldout_comparison.json').write_text(json.dumps(summary,indent=2));print('HELDOUT_COMPARISON_READY='+json.dumps(summary),flush=True)
