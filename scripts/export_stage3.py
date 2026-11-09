"""Export aggregate evidence only; never copy checkpoints, EEG, or prediction rows."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

ROOT = Path('/mnt/cfs-omni/lhy/neurolm-reproduction/stage3-new-pretrain-20261109-r1')
parser = argparse.ArgumentParser(__doc__)
parser.add_argument('phase', choices=('startup', 'trained', 'final'))
args = parser.parse_args()
out = ROOT / 'monitor' / 'exports' / args.phase
out.mkdir(parents=True, exist_ok=False)
if args.phase in ('trained', 'final'):
    assert json.loads((ROOT / 'checkpoint_audit.json').read_text())['ok']
if args.phase == 'final':
    assert json.loads((ROOT / 'comparison.json').read_text())['status'] == 'completed'
    assert (ROOT / 'exit_code.txt').read_text().strip() == '0'
    assert json.loads((ROOT / 'platform_final_status.json').read_text())['tasks'][0]['status'] == 'SUCCEED'
names = [
    'submitted.json', 'preflight.json', 'cpu_checks.log', 'source_manifest.json',
    'pre_submit_resources.json', 'platform_status.json', 'platform_final_status.json',
    'checkpoint_audit.json', 'comparison.json', 'audit-new.json', 'audit-official.json',
    'exit_code.txt', 'monitor/status.json', 'monitor/training_curves.png',
    'monitor/comparison.png', 'monitor/training_summary.json', 'monitor/platform_completion.json',
    'train/run_manifest.json', 'train/train_steps.jsonl',
    'author/bundle_manifest.json', 'gpu_metrics.csv',
]
for variant in ('new', 'official'):
    names.append(f'eval-{variant}/summary.json')
    names.extend(str(p.relative_to(ROOT)) for p in sorted((ROOT / f'eval-{variant}').glob('*-metrics.json')))
for name in names:
    source = ROOT / name
    if source.is_file():
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
for name in ('worker.log', 'tmux.log'):
    source = ROOT / name
    if source.exists():
        lines = source.read_text(errors='replace').splitlines()
        (out / f'{name}.snapshot.txt').write_text('\n'.join(lines[:30] + ['--- tail ---'] + lines[-30:]) + '\n')
metadata = dict(phase=args.phase, exported_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                source_root=str(ROOT), raw_predictions_exported=False, model_weights_exported=False,
                eeg_exported=False, live_snapshot=args.phase != 'final')
(out / 'export.json').write_text(json.dumps(metadata, indent=2))
hashes = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(out.rglob('*')) if p.is_file()}
(out / 'sha256.json').write_text(json.dumps(hashes, indent=2))
archive = out.with_suffix('.tar.gz')
with tarfile.open(archive, 'w:gz') as tar:
    tar.add(out, arcname=args.phase)
print(json.dumps(dict(archive=str(archive), sha256=hashlib.sha256(archive.read_bytes()).hexdigest(), files=len(hashes))))
