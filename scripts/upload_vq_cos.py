"""Convert frozen groups and PUT with server-scoped URLs supplied through stdin."""
import concurrent.futures,hashlib,json,os,struct,sys,time
from pathlib import Path
import requests
from prepare_five_dataset_vq import make_group
SESSION=None

def upload(job):
    global SESSION
    group,url=job
    group,body,stats=make_group(group)
    header=json.dumps({'kind':'group','group':group,'stats':stats,'payload_bytes':len(body),
                       'payload_sha256':hashlib.sha256(body).hexdigest()},separators=(',',':')).encode()
    payload=struct.pack('!I',len(header))+header+body
    if SESSION is None:
        SESSION=requests.Session();SESSION.trust_env=False
    last_status=None;error_type=None
    for attempt in range(6):
        try:
            response=SESSION.put(url,data=payload,timeout=(10,90))
            last_status=response.status_code
            if response.status_code in (200,201):
                return {'ok':True,'id':group['id'],'bytes':len(payload),'samples':group['count']}
            if response.status_code==403: break
        except requests.RequestException as exc:
            error_type=type(exc).__name__
            if os.environ.get('VQ_UPLOAD_TRACE')=='1':
                chain=[];seen=set();pending_errors=[exc]
                while pending_errors:
                    error=pending_errors.pop()
                    if id(error) in seen: continue
                    seen.add(id(error));chain.append({'type':type(error).__name__,'errno':getattr(error,'errno',None)})
                    pending_errors.extend(item for item in [error.__cause__,error.__context__,getattr(error,'reason',None),*error.args] if isinstance(item,BaseException))
                print(json.dumps({'network_attempt':attempt+1,'group':group['id'],'error_chain':chain}),flush=True)
            SESSION.close();SESSION=requests.Session();SESSION.trust_env=False
        time.sleep(min(5,attempt+1))
    return {'ok':False,'id':group['id'],'status':last_status,'error_type':error_type}

if __name__=='__main__':
    manifest=json.load(sys.stdin)
    plan=json.loads(Path('/data/lihy/NeuroLM/outputs/vq-five-full-20261107-r1/source_plan.json').read_text())
    assert manifest['plan_sha256']==plan['plan_sha256']
    groups={g['id']:g for g in plan['groups']}
    started=time.time();done=0;sent=0
    from collections import deque
    jobs=deque((groups[k],url) for k,url in manifest['urls'].items());attempts={}
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as pool:
        pending={}
        for _ in range(8):
            job=jobs.popleft() if jobs else None
            if job: pending[pool.submit(upload,job)]=job[0]['id']
        while pending:
            ready,_=concurrent.futures.wait(pending,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in ready:
                pending.pop(future);row=future.result()
                if not row['ok']:
                    print(json.dumps({'retry':row}),flush=True)
                    attempts[row['id']]=attempts.get(row['id'],0)+1
                    if row['status']==403 or attempts[row['id']]>=4:
                        raise RuntimeError('COS_UPLOAD_FAILED group='+row['id']+' status='+str(row['status']))
                    jobs.append((groups[row['id']],manifest['urls'][row['id']]))
                else:
                    done+=1;sent+=row['bytes']
                if row['ok'] and (done<=5 or done%20==0):
                    print(json.dumps({'uploaded_groups':done,'total':len(manifest['urls']),'bytes':sent,'seconds':time.time()-started,'last':row['id']}),flush=True)
                job=jobs.popleft() if jobs else None
                if job: pending[pool.submit(upload,job)]=job[0]['id']
    print(json.dumps({'complete':True,'uploaded_groups':done,'bytes':sent,'seconds':time.time()-started}),flush=True)
