"""Run inside pretrain_neurolm tmux: submit once, monitor to terminal; never retry training."""
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/mnt/cfs-omni/lhy/neurolm-reproduction/pretrain-full20-20261109-r1')
def query(mode):
    response=subprocess.run([sys.executable,str(ROOT/'notebook.py'),mode],capture_output=True,text=True,timeout=115)
    result=json.loads(response.stdout)
    (ROOT/f'platform_{mode}.json').write_text(json.dumps(result,indent=2))
    assert response.returncode==0 and result['ok'],result
    return result
dry=query('dry-run');assert not dry['matching_tasks'],dry
submitted=query('submit');print('SUBMITTED='+json.dumps(submitted),flush=True)
assert submitted['submitted_id']
while True:
    try:
        status=query('status');tasks=status.get('tasks',[]);assert len(tasks)==1,tasks
        row=tasks[0];assert row['id']==submitted['submitted_id']
        print(datetime.datetime.now(datetime.timezone.utc).isoformat(),json.dumps(row),flush=True)
        subprocess.run([sys.executable,str(ROOT/'observe.py')],check=True)
        if row['status'] in ('SUCCEED','FAILED','STOPPED','DELETED','TERMINATED'):
            (ROOT/'platform_final_status.json').write_text(json.dumps(status,indent=2))
            if row['status']!='SUCCEED':raise SystemExit('Training stopped; no retry or code modification performed')
            done=json.loads((ROOT/'run/completed.json').read_text());assert done['epochs_completed']==20
            print('ALL_20_EPOCHS_COMPLETE',flush=True);break
    except (subprocess.SubprocessError,ValueError,AssertionError) as exc:
        print('MONITOR_QUERY_ERROR='+type(exc).__name__,flush=True)
    time.sleep(60)
