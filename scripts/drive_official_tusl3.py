"""Run three independent stage2 -> stage3 pipelines; no automatic training retries."""
import concurrent.futures
import datetime
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
PY=str(ROOT.parent/'envs/neurolm-py311/bin/python')
TERMINAL={'SUCCEED','FAILED','STOPPED','DELETED','TERMINATED'}


def query(job,mode):
    p=subprocess.run([sys.executable,str(job/'notebook.py'),mode],capture_output=True,text=True,timeout=115)
    result=json.loads(p.stdout)
    (job/f'platform_{mode}.json').write_text(json.dumps(result,indent=2))
    if p.returncode or not result.get('ok'):raise RuntimeError(f'{job.name} {mode}: {result}')
    return result


def start(job):
    dry=query(job,'dry-run');assert not dry['matching_tasks'],'Duplicate task name; refusing automatic resubmission'
    result=query(job,'submit');assert result.get('submitted_id')
    print('SUBMITTED='+json.dumps(dict(job=str(job),**result)),flush=True)
    return result['submitted_id']


def main():
    assert json.loads((ROOT/'preflight.json').read_text())['ok']
    jobs={seed:ROOT/f'seed-{seed}/stage2' for seed in (42,43,44)}
    capacity=query(jobs[42],'resources')
    assert capacity['eligible_eight_gpu_nodes']>=6,'Three parallel 16-GPU runs need six available eight-GPU nodes'
    state={seed:dict(stage=2,status='SUBMITTING') for seed in jobs}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        futures={pool.submit(start,job):seed for seed,job in jobs.items()}
        for future in concurrent.futures.as_completed(futures):
            seed=futures[future]
            try:state[seed].update(id=future.result(),status='ACTIVE')
            except Exception as exc:state[seed].update(status='FAILED',error=str(exc));print('SUBMISSION_ERROR',seed,str(exc),flush=True)
    while any(s['status']=='ACTIVE' for s in state.values()):
        for seed,s in state.items():
            if s['status']!='ACTIVE':continue
            job=jobs[seed]
            try:
                result=query(job,'status');rows=result['tasks'];assert len(rows)==1 and rows[0]['id']==s['id']
            except Exception as exc:
                print('STATUS_QUERY_ERROR',seed,type(exc).__name__,flush=True);continue
            row=rows[0]
            if s.get('platform_status')!=row['status']:
                print('STATUS_CHANGE='+json.dumps(dict(seed=seed,stage=s['stage'],**row)),flush=True)
            s['platform_status']=row['status']
            if row['status'] not in TERMINAL:continue
            (job/'platform_final_status.json').write_text(json.dumps(result,indent=2))
            if row['status']!='SUCCEED':
                s.update(status='FAILED',error='TI-ONE task did not succeed');continue
            try:
                audit=json.loads((job/'checkpoint_audit.json').read_text());assert audit['ok']
                if s['stage']==2:
                    assert all((job/f'node-{i}/exit_code.txt').read_text().strip()=='0' for i in (0,1))
                    next_job=job.parent/'stage3'
                    p=subprocess.run([PY,str(ROOT/'scripts/preflight_official_stage3.py'),'--job',str(next_job)],timeout=300)
                    assert p.returncode==0,'Stage-three preflight failed'
                    ident=start(next_job);jobs[seed]=next_job;s.update(stage=3,id=ident,platform_status='SUBMITTED')
                else:
                    assert (job/'exit_code.txt').read_text().strip()=='0'
                    audit=json.loads((job/'eval_audit.json').read_text());assert audit['status']=='passed' and audit['total_samples']==201240
                    s.update(status='COMPLETE');print('SEED_COMPLETE',seed,flush=True)
            except Exception as exc:
                s.update(status='FAILED',error=str(exc));print('PIPELINE_GATE_ERROR',seed,str(exc),flush=True)
        report=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),seeds=state)
        (ROOT/'pipeline_status.json').write_text(json.dumps(report,indent=2))
        subprocess.run([PY,str(ROOT/'scripts/observe_official_tusl3.py')],timeout=90)
        if any(s['status']=='ACTIVE' for s in state.values()):time.sleep(45)
    print('PIPELINE_FINAL='+json.dumps(state),flush=True)
    if not all(s['status']=='COMPLETE' for s in state.values()):raise SystemExit(1)


if __name__=='__main__':main()
