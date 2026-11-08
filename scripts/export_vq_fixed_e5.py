"""Export completed VQ diagnostics and provenance, excluding EEG and model weights."""
import hashlib
import json
from pathlib import Path
import tarfile

root = Path('/mnt/cfs-omni/lhy/neurolm-reproduction/vq-fixed-e5-20261108-r1')
assert json.loads((root / 'verified.json').read_text())['verified_complete']
mapping = {name:name for name in ['verified.json', 'preflight.json', 'preflight_node_0.json',
                                  'preflight_node_1.json', 'submitted.json', 'task.json']}
mapping['code/bundle_manifest.json'] = 'bundle_manifest.json'
for name in ['metrics.jsonl', 'run_config.json', 'completed.json', 'epoch_metrics.jsonl',
             'monitor_baseline.json', 'initial_state.jsonl']:
    mapping['run/' + name] = name
for rank in [0, 1]:
    for name in ['gpu_metrics.csv', 'exit_code.txt']:
        mapping[f'node-{rank}/{name}'] = f'node_{rank}_{name}'
mapping['fixed_audit/audit.json'] = 'fixed_audit.json'
for name in ['vq_fixed_e5_curves.png', 'vq_fixed_e5_curves.svg', 'vq_fixed_e5_curves.pdf', 'curve_metadata.json']:
    mapping['monitor/figures/' + name] = 'figures/' + name
hashes = {target:hashlib.sha256((root / source).read_bytes()).hexdigest() for source, target in mapping.items()}
(root / 'export_hashes.json').write_text(json.dumps(hashes, indent=2))
archive = root / 'results_export.tar.gz'
with tarfile.open(archive, 'w:gz') as tar:
    for source, target in mapping.items():
        tar.add(root / source, arcname=target)
    tar.add(root / 'export_hashes.json', arcname='artifact_sha256.json')
print(json.dumps(dict(ok=True, files=len(mapping), bytes=archive.stat().st_size,
                     archive=str(archive), sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
