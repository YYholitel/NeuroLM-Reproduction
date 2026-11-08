"""Export verified A/B metrics and provenance, excluding EEG and model weights."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path('/mnt/cfs-omni/lhy/neurolm-reproduction/vq-reset-ab-500-20261108-r1')
assert json.loads((root/'comparison_verified.json').read_text())['ok']
mapping={name:name for name in ['comparison_verified.json','early_match.json','preflight.json',
    'preflight_node_0.json','preflight_node_1.json','submitted.json','task.json','integration_test.json']}
mapping['code/bundle_manifest.json']='bundle_manifest.json'
for arm in ['reset_off','reset_on']:
    for name in ['metrics.jsonl','run_config.json','completed.json','epoch_metrics.jsonl',
                 'monitor_baseline.json','initial_state.jsonl']:
        mapping[f'runs/{arm}/{name}']=f'{arm}_{name}'
for rank in range(2):
    assert (root/f'node-{rank}/exit_code.txt').read_text().strip()=='0'
    for name in ['exit_code.txt','gpu_metrics.csv']:
        mapping[f'node-{rank}/{name}']=f'node_{rank}_{name}'
archive=root/'results_export.tar.gz'
hashes={}
with tarfile.open(archive,'w:gz') as tar:
    for source,target in mapping.items():
        hashes[target]=hashlib.sha256((root/source).read_bytes()).hexdigest()
        tar.add(root/source,arcname=target)
(root/'export_hashes.json').write_text(json.dumps(hashes,indent=2))
print(json.dumps(dict(ok=True,files=len(mapping),archive=str(archive),bytes=archive.stat().st_size,
                     sha256=hashlib.sha256(archive.read_bytes()).hexdigest())))
