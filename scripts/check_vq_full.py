"""Read-only synchronization of the deployed five-dataset VQ run.

Run Test-TioneSts.ps1 first. Reuses the documented Notebook STS transport;
does not submit, stop or edit training jobs and never returns credentials.
"""
import argparse
from pathlib import Path
import shlex
import subprocess

ROOT='/mnt/cfs-omni/lhy/neurolm-reproduction/vq-five-full-20261107-r1'
parser=argparse.ArgumentParser(__doc__)
parser.add_argument('mode',choices=['status','files','logs','details','events'],default='status',nargs='?')
args=parser.parse_args()
ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','tencent']
if args.mode=='files':
    result=subprocess.run(ssh+[shlex.join(['python',ROOT+'/inspect.py'])],timeout=90)
else:
    repo=Path(__file__).resolve().parents[1]
    probe=(repo/'scripts/tione_notebook_sts_probe.py').read_text()
    start=probe.index("code = r'''");end=probe.index('\ntry:\n    specs =',start)
    code="import runpy,json\nns=runpy.run_path("+repr(ROOT+'/control.py')+")\nprint('TIONE_STS_PROBE_JSON='+json.dumps(ns['safe_execute']("+repr(args.mode)+")))"
    probe=probe[:start]+'code = '+repr(code)+'\n'+probe[end:]
    result=subprocess.run(ssh+['python -'],input=probe.encode(),timeout=115)
raise SystemExit(result.returncode)
