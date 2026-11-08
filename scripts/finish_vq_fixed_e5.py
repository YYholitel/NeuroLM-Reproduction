"""Wait for this VQ job, then verify and audit it. Never starts another training task."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path('/mnt/cfs-omni/lhy/neurolm-reproduction/vq-fixed-e5-20261108-r1')
while True:
    exits = [root / f'node-{rank}/exit_code.txt' for rank in [0, 1]]
    if any(path.exists() and path.read_text().strip() != '0' for path in exits):
        print('VQ task failed; no next-stage training will be started.', flush=True)
        raise SystemExit(1)
    if (root / 'run/completed.json').exists() and all(path.exists() for path in exits):
        break
    time.sleep(60)
env = dict(os.environ, OMP_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1',
           MPLCONFIGDIR=str(root / 'monitor/.mplconfig'))
commands = [
    ['verify.py', '--root', str(root)],
    ['audit.py', '--root', str(root), '--checkpoint', str(root / 'run/checkpoints/VQ/ckpt-4.pt'),
     '--reference', str(root / 'reference_audit.json'), '--output', str(root / 'fixed_audit')],
    ['plot.py', '--input', str(root / 'run/metrics.jsonl'), '--output-dir', str(root / 'monitor/figures')],
    ['export.py'],
]
for command in commands:
    print(json.dumps({'executing': command}), flush=True)
    subprocess.run([sys.executable, str(root / command[0]), *command[1:]], check=True, env=env)
print('VQ_RESULTS_READY: inspect verified.json, fixed_audit/audit.json and curves before any next-stage submission.', flush=True)
