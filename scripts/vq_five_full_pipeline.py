"""Wait for the frozen transfer, verify it, then submit the authorized VQ run."""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path('/mnt/cfs-omni/lhy/neurolm-reproduction/vq-five-full-20261107-r1')

def state(stage,**extra):
    value={'stage':stage,'time':time.time(),**extra}
    temp=ROOT/'pipeline_state.tmp'
    temp.write_text(json.dumps(value,indent=2)); os.replace(temp,ROOT/'pipeline_state.json')
    print(json.dumps(value),flush=True)

def action(mode):
    result=subprocess.run([sys.executable,str(ROOT/'notebook_action.py'),mode],
                          capture_output=True,text=True,timeout=120)
    report=json.loads(result.stdout)
    assert result.returncode==0 and report['ok'] and report['temporary_probe_kernel_removed'],report
    return report

def main():
    lock=(ROOT/'pipeline.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state('WAITING_FOR_TRANSFER',expected_samples=459656)
    while not (ROOT/'assets/transfer_complete.json').exists():
        time.sleep(30)
    state('VERIFYING_DATA')
    check=subprocess.run([sys.executable,str(ROOT/'preflight.py'),'--root',str(ROOT)],timeout=3600)
    assert check.returncode==0,'Data verification failed; training was not submitted'
    state('DATA_VERIFIED')
    for attempt in range(6):
        try:
            dry=action('dry-run')
            (ROOT/'dry_run.json').write_text(json.dumps(dry,indent=2))
            report=action('submit')
            (ROOT/'submission_readback.json').write_text(json.dumps(report,indent=2))
            tasks=report.get('tasks',[])
            assert len(tasks)==1,report
            task_id=tasks[0]['id']
            state('SUBMITTED',task_id=task_id,status=tasks[0]['status'])
            break
        except Exception as exc:
            state('SUBMIT_RETRY',attempt=attempt+1,error_type=type(exc).__name__)
            if attempt==5: raise
            time.sleep(30)
    while True:
        time.sleep(30)
        try:
            report=action('status')
        except Exception as exc:
            state('STATUS_RETRY',task_id=task_id,error_type=type(exc).__name__)
            continue
        row=next(r for r in report['tasks'] if r['id']==task_id)
        status=row['status']
        metrics=ROOT/'run/metrics.jsonl'
        if metrics.exists():
            lines=metrics.read_text().splitlines()
            if len(lines)>=10:
                state('TRAINING_VERIFIED',task_id=task_id,status=status,
                      recorded_updates=len(lines),last_metrics=json.loads(lines[-1]))
                return
        if status in ['SUCCEED','FAILED','STOPPED','DELETED','TERMINATED']:
            state('TERMINAL',task_id=task_id,status=status)
            return
        prior=json.loads((ROOT/'pipeline_state.json').read_text())
        if prior.get('status')!=status:
            state('WAITING_FOR_INITIAL_UPDATES',task_id=task_id,status=status)

if __name__=='__main__':
    try: main()
    except Exception as exc:
        state('PIPELINE_FAILED',error_type=type(exc).__name__)
        raise
