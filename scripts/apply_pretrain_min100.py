"""Build a new stage-two runtime from the verified VQ runtime and author entrypoint."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

parser=argparse.ArgumentParser(__doc__)
parser.add_argument('--source-runtime',type=Path,required=True)
parser.add_argument('--upstream-root',type=Path,required=True)
parser.add_argument('--output-runtime',type=Path,required=True)
args=parser.parse_args();source=args.source_runtime.resolve();output=args.output_runtime.resolve()
assert not output.exists() and source not in output.parents
repo=Path(__file__).resolve().parents[1]
checks=json.loads((repo/'results/pretrain_min100_patch_checks_20261108.json').read_text())
patch=repo/'patches/0019-bounded-stage-two-with-frozen-repaired-vq.patch'
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
assert sha(patch)==checks['patch_sha256']
manifest=json.loads((source/'bundle_manifest.json').read_text())
for name,expected in manifest.items():assert sha(source/name)==expected,name
with tempfile.TemporaryDirectory(prefix='neurolm-stage2-') as directory:
 staged=Path(directory)
 for name,expected in checks['changed_files'].items():
  if expected['before'] is None:continue
  original=args.upstream_root/name if name=='train_pretrain.py' else source/name
  normalized=original.read_text().encode()
  assert hashlib.sha256(normalized).hexdigest()==expected['before'],name
  (staged/name).parent.mkdir(parents=True,exist_ok=True);(staged/name).write_bytes(normalized)
 env=dict(os.environ,GIT_CEILING_DIRECTORIES=str(staged.parent))
 subprocess.run(['git','-c','core.autocrlf=false','apply','--check',str(patch)],cwd=staged,env=env,check=True)
 subprocess.run(['git','-c','core.autocrlf=false','apply',str(patch)],cwd=staged,env=env,check=True)
 for name,expected in checks['changed_files'].items():assert sha(staged/name)==expected['after'],name
 shutil.copytree(source,output,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git'))
 for name in checks['changed_files']:shutil.copyfile(staged/name,output/name);manifest[name]=sha(output/name)
(output/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(ok=True,files=len(manifest),output=str(output))))
