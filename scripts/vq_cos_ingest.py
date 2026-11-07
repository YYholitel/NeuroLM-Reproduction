"""Run inside the authorized Notebook identity; ingest only this run's COS prefix."""
import concurrent.futures,hashlib,json,struct,subprocess,sys,time,zlib
from pathlib import Path
from cos_control import ROOT,BUCKET,PREFIX,client

def frame(header,body=b''):
    header=dict(header,payload_bytes=len(body),payload_sha256=hashlib.sha256(body).hexdigest())
    data=json.dumps(header,separators=(',',':')).encode()
    return struct.pack('!I',len(data))+data+body

def main():
    assets=ROOT/'assets';plan=json.loads((assets/'source_plan.json').read_text())
    expected={g['id'] for g in plan['groups']}
    receiver=subprocess.Popen([sys.executable,'/mnt/cfs-omni/lhy/neurolm-reproduction/control/receive_vq_shards.py',
                               'receive','--output-root',str(assets)],stdin=subprocess.PIPE)
    receiver.stdin.write(frame({'kind':'plan'},zlib.compress(json.dumps(plan).encode(),1)));receiver.stdin.flush()
    cos,_=client();refreshed=time.time();ingested=0;totalbytes=0;started=time.time();sent=set()
    def fetch(group_id):
        body=cos.get_object(Bucket=BUCKET,Key=PREFIX+group_id+'.frame')['Body'].get_raw_stream().read()
        return group_id,body
    while True:
        have={p.stem for p in (assets/'shard_receipts').glob('*.json')}
        if have==expected: break
        if receiver.poll() is not None: raise RuntimeError('Shard receiver exited')
        if time.time()-refreshed>300:
            cos,_=client();refreshed=time.time()
        marker='';objects=set()
        while True:
            response=cos.list_objects(Bucket=BUCKET,Prefix=PREFIX,Marker=marker,MaxKeys=1000)
            for item in response.get('Contents',[]):
                key=item['Key'];name=key[len(PREFIX):]
                if name.endswith('.frame') and name[:-6] in expected: objects.add(name[:-6])
            if response.get('IsTruncated') not in ['true',True]: break
            marker=response['NextMarker']
        pending=sorted(objects-have-sent)
        if not pending:
            time.sleep(5);continue
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            jobs=iter(pending);futures={}
            for _ in range(4):
                key=next(jobs,None)
                if key: futures[pool.submit(fetch,key)]=key
            while futures:
                done,_=concurrent.futures.wait(futures,return_when=concurrent.futures.FIRST_COMPLETED)
                for future in done:
                    futures.pop(future);key,data=future.result()
                    receiver.stdin.write(data);receiver.stdin.flush()
                    sent.add(key)
                    ingested+=1;totalbytes+=len(data)
                    if ingested<=5 or ingested%20==0:
                        row={'ingested_groups':ingested,'bytes':totalbytes,'seconds':time.time()-started,'last':key}
                        (ROOT/'cos_ingest_state.json').write_text(json.dumps(row));print(json.dumps(row),flush=True)
                    key=next(jobs,None)
                    if key: futures[pool.submit(fetch,key)]=key
    receiver.stdin.write(frame({'kind':'done','plan_sha256':plan['plan_sha256']}));receiver.stdin.close()
    assert receiver.wait(timeout=60)==0
    print(json.dumps({'complete':True,'ingested_groups':ingested,'bytes':totalbytes,'seconds':time.time()-started}),flush=True)

if __name__=='__main__':
    try: main()
    except Exception as exc:
        row={'ok':False,'error_type':type(exc).__name__,'time':time.time()}
        if hasattr(exc,'get_error_code'): row['error_code']=exc.get_error_code()
        (ROOT/'cos_ingest_failed.json').write_text(json.dumps(row))
        print(json.dumps(row),flush=True);raise SystemExit(1)
