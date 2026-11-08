#!/usr/bin/env python3
"""Read-only TI-ONE STS probe; stream this file to the authorized notebook via SSH.

Uses the existing root Jupyter launch token only on that server. Credentials are
never printed, copied out, or persisted. Creates and removes its own probe kernel.
"""
import os, pathlib, json, uuid, time, datetime
import requests, websocket
token = None
base = '/'
for p in pathlib.Path('/proc').iterdir():
    if not p.name.isdigit():
        continue
    try:
        if p.stat().st_uid != 0 or (p/'comm').read_text().strip() != 'jupyter-lab':
            continue
        args = [a.decode(errors='replace') for a in (p/'cmdline').read_bytes().split(b'\0') if a]
        for i,a in enumerate(args):
            key, sep, val = a.partition('=')
            if not sep and i+1 < len(args) and not args[i+1].startswith('--'):
                val = args[i+1]
            if key in ('--NotebookApp.token','--ServerApp.token','--IdentityProvider.token') and val:
                token = val
            elif key in ('--NotebookApp.base_url','--ServerApp.base_url'):
                base = val or '/'
    except OSError:
        pass
if not token:
    print(json.dumps({'ok':False,'stage':'notebook_auth','reason':'LAUNCH_TOKEN_NOT_FOUND'}))
    raise SystemExit(1)
for k in ('http_proxy','https_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','all_proxy'):
    os.environ.pop(k, None)
s = requests.Session()
s.trust_env = False
s.headers['Authorization'] = 'token ' + token
origin = 'http://127.0.0.1:8888'
root = origin + ('/' + base.strip('/') if base.strip('/') else '')
kernel_id = None
ws = None
report = {'ok':False, 'auth':'STS'}
import sys
assert sys.argv[1] in ('dry-run','submit','status','details','events','resources')
code = "import runpy,json\nns=runpy.run_path('/mnt/cfs-omni/lhy/neurolm-reproduction/pretrain-full20-20261109-r1/control.py')\nprint('TIONE_STS_PROBE_JSON='+json.dumps(ns['safe_execute'](MODE)))".replace('MODE',repr(sys.argv[1]))

try:
    specs = s.get(root+'/api/kernelspecs', timeout=8)
    specs.raise_for_status()
    all_specs = specs.json()['kernelspecs']
    name = 'python3' if 'python3' in all_specs else next((k for k,v in all_specs.items() if v.get('spec',{}).get('language')=='python'), None)
    if not name:
        raise RuntimeError('NoPythonKernel')
    created = s.post(root+'/api/kernels', json={'name':name}, timeout=15)
    created.raise_for_status()
    kernel_id = created.json()['id']
    session_id = uuid.uuid4().hex
    msg_id = uuid.uuid4().hex
    ws = websocket.create_connection(root.replace('http://','ws://',1)+'/api/kernels/'+kernel_id+'/channels?session_id='+session_id,
        header={'Authorization':'token '+token}, origin=origin, timeout=8)
    ws.send(json.dumps({'header':{'msg_id':msg_id,'username':'codex','session':session_id,
        'date':datetime.datetime.now(datetime.timezone.utc).isoformat(),'msg_type':'execute_request','version':'5.3'},
        'parent_header':{},'metadata':{},'channel':'shell',
        'content':{'code':code,'silent':False,'store_history':False,'user_expressions':{},'allow_stdin':False,'stop_on_error':True}}))
    deadline = time.monotonic()+75
    buffer = ''
    while time.monotonic()<deadline:
        try:
            msg = json.loads(ws.recv())
        except websocket.WebSocketTimeoutException:
            continue
        if msg.get('parent_header',{}).get('msg_id') != msg_id:
            continue
        typ = msg.get('msg_type') or msg.get('header',{}).get('msg_type')
        content = msg.get('content',{})
        if typ == 'stream':
            buffer += content.get('text','')
            for line in buffer.splitlines():
                if line.startswith('TIONE_STS_PROBE_JSON='):
                    report = json.loads(line.split('=',1)[1])
        elif typ == 'error':
            report = {'ok':False,'stage':'notebook_execution','error_type':content.get('ename','Unknown')}
        elif typ == 'status' and content.get('execution_state') == 'idle':
            break
    else:
        report = {'ok':False,'reason':'EXECUTION_TIMEOUT'}
except Exception as exc:
    report = {'ok':False,'stage':'notebook_interface','error_type':type(exc).__name__}
finally:
    if ws is not None:
        ws.close()
    if kernel_id is not None:
        try:
            deleted = s.delete(root+'/api/kernels/'+kernel_id, timeout=10)
            report['temporary_probe_kernel_removed'] = deleted.status_code in (200,204,404)
        except Exception:
            report['temporary_probe_kernel_removed'] = False
print(json.dumps(report,indent=2))

raise SystemExit(0 if report.get('ok') and report.get('temporary_probe_kernel_removed') else 1)
