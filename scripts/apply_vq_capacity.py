"""Apply the capacity CLI patch to a new copy of the verified repaired VQ runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

p=argparse.ArgumentParser(__doc__);p.add_argument('--source-runtime',type=Path,required=True);p.add_argument('--output-runtime',type=Path,required=True)
a=p.parse_args();source=a.source_runtime.resolve();output=a.output_runtime.resolve()
assert not output.exists() and source not in output.parents
repo=Path(__file__).resolve().parents[1];patch=repo/'patches/0020-configurable-vq-codebook-capacity.patch'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(patch)=='53bc021d4c9401b60b149edc0f95cf0ce82d8f498b3c4c1d83b317cb448244f7'
manifest=json.loads((source/'bundle_manifest.json').read_text())
for name,expected in manifest.items():assert sha(source/name)==expected,name
assert sha(source/'train_vq.py')=='3f79a6c97111d1d20ffc9733736a11aafe81dd8e959e8aa08f3e0417a17c48c1'
with tempfile.TemporaryDirectory(prefix='neurolm-capacity-') as temporary:
 staged=Path(temporary);(staged/'train_vq.py').write_text((source/'train_vq.py').read_text(),newline='\n')
 env=dict(os.environ,GIT_CEILING_DIRECTORIES=str(staged.parent))
 subprocess.run(['git','-c','core.autocrlf=false','apply','--check',str(patch)],cwd=staged,env=env,check=True)
 subprocess.run(['git','-c','core.autocrlf=false','apply',str(patch)],cwd=staged,env=env,check=True)
 shutil.copytree(source,output,ignore=shutil.ignore_patterns('__pycache__','*.pyc','.git'))
 shutil.copyfile(staged/'train_vq.py',output/'train_vq.py')
manifest['train_vq.py']=sha(output/'train_vq.py')
(output/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(dict(ok=True,files=len(manifest),train_vq_sha256=manifest['train_vq.py'],source_unchanged=True)))
