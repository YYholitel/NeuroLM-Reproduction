"""Persistent file-based progress and plot monitoring; does not submit training."""
import datetime,json,os,subprocess,sys,time
from pathlib import Path
root=Path('/mnt/cfs-omni/lhy/neurolm-reproduction/vq-fixed-e5-20261108-r1')
monitor=root/'monitor';monitor.mkdir(exist_ok=True)
last_render=-1
while True:
    try:
        check=subprocess.run([sys.executable,str(root/'inspect.py')],capture_output=True,text=True,check=True,timeout=50)
        state=json.loads(check.stdout)
        temporary=monitor/'latest.tmp';temporary.write_text(json.dumps(state,indent=2));os.replace(temporary,monitor/'latest.json')
        if state['updates'] and (state['updates']-last_render>=100 or state['completed']):
            subprocess.run([sys.executable,str(root/'plot.py'),'--input',str(root/'run/metrics.jsonl'),
                            '--output-dir',str(monitor/'figures')],check=True,timeout=120,
                           env=dict(os.environ,MPLCONFIGDIR=str(monitor/'.mplconfig'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
            last_render=state['updates']
        latest=state.get('latest') or {}
        print(json.dumps(dict(checked_at=state['checked_at'],updates=state['updates'],completed=state['completed'],
                              epochs=len(state['epochs']),last_render=last_render,
                              losses={key:latest.get(key) for key in ['reconstruction','codebook','adversarial']},
                              per_batch=latest.get('code_usage'),rolling_50=latest.get('rolling_code_usage'))),flush=True)
        if state['completed'] or any(node.get('exit_code','0')!='0' for node in state['nodes'].values()):break
    except Exception as exc:
        print(json.dumps(dict(monitor_error=type(exc).__name__,time=datetime.datetime.now(datetime.timezone.utc).isoformat())),flush=True)
    time.sleep(60)
