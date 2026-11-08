"""Assemble full training from the exact successful 100-step runtime (no model edits)."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

p=argparse.ArgumentParser(__doc__)
p.add_argument('--source-runtime',type=Path,required=True)
p.add_argument('--output-runtime',type=Path,required=True)
a=p.parse_args()
repo=Path(__file__).resolve().parents[1]
manifest=json.loads((repo/'results/pretrain_full20_source_manifest.json').read_text())
for name,expected in manifest.items():
    assert hashlib.sha256((a.source_runtime/name).read_bytes()).hexdigest()==expected,name
assert not a.output_runtime.exists()
shutil.copytree(a.source_runtime,a.output_runtime,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
shutil.copyfile(repo/'scripts/pretrain_full20_runtime.py',a.output_runtime/'pretrain_minimal.py')
manifest['pretrain_minimal.py']=hashlib.sha256((a.output_runtime/'pretrain_minimal.py').read_bytes()).hexdigest()
(a.output_runtime/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(ok=True,runtime_files=len(manifest),changed_files=['pretrain_minimal.py'],
                     model_files_and_train_pretrain_unchanged=True)))
