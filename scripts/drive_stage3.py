"""Submit the authorized matched stage-three run once and monitor it to termination."""
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/mnt/cfs-omni/lhy/neurolm-reproduction/stage3-new-pretrain-20261109-r1')
def query(mode):
    process=subprocess.run([sys.executable,str(ROOT/'notebook.py'),mode],capture_output=True,text=True,timeout=115)
    report=json.loads(process.stdout)
    (ROOT/f'platform_{mode}.json').write_text(json.dumps(report,indent=2))
    assert process.returncode==0 and report['ok'],report
    return report
dry=query('dry-run');assert not dry['matching_tasks'],dry
submitted=query('submit');assert submitted['submitted_id'];print('SUBMITTED='+json.dumps(submitted),flush=True)
while True:
    try:
        report=query('status');tasks=report['tasks'];assert len(tasks)==1 and tasks[0]['id']==submitted['submitted_id']
        row=tasks[0];print(datetime.datetime.now(datetime.timezone.utc).isoformat(),json.dumps(row),flush=True)
        terminal=row['status'] in ('SUCCEED','FAILED','STOPPED','DELETED','TERMINATED')
        if terminal:(ROOT/'platform_final_status.json').write_text(json.dumps(report,indent=2))
        process=subprocess.run([sys.executable,str(ROOT/'observe.py')])
        if terminal:
            if row['status']!='SUCCEED':raise SystemExit('Training/evaluation failed; no retry or model change performed')
            assert (ROOT/'exit_code.txt').read_text().strip()=='0'
            assert json.loads((ROOT/'comparison.json').read_text())['status']=='completed'
            print('STAGE3_COMPLETE',flush=True);break
        if process.returncode:print('MONITOR_RENDER_ERROR',process.returncode,flush=True)
    except (subprocess.SubprocessError,ValueError,AssertionError) as exc:
        print('MONITOR_QUERY_ERROR='+type(exc).__name__,flush=True)
    time.sleep(60)
