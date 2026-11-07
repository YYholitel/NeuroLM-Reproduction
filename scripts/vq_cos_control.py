import base64,contextlib,hashlib,io,json,os,struct,subprocess,sys,time
from pathlib import Path
ROOT=Path('/mnt/cfs-omni/lhy/neurolm-reproduction/vq-five-full-20261107-r1')
BUCKET='omni-eeg-1442740494'
PREFIX='lhy/neurolm-reproduction/transfer/vq-five-full-20261107-r1/'

def client():
    sys.path.insert(0,'/mnt/cfs-omni/chr/video_reasoning/cvpr2027/skills/tione-platform-ops')
    import tione_ops as ops
    from qcloud_cos import CosConfig,CosS3Client
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        ops._clear_proxy();creds=ops._fetch_sts_creds()
    assert creds
    return CosS3Client(CosConfig(Region='ap-beijing',SecretId=creds['TmpSecretId'],SecretKey=creds['TmpSecretKey'],Token=creds['Token'])),creds

def execute(mode):
    cos,creds=client()
    if mode=='expiry':
        return {'ok':True,'time':time.time(),'Expiration':creds.get('Expiration'),'ExpiredTime':creds.get('ExpiredTime')}
    plan=json.loads((ROOT/'assets/source_plan.json').read_text())
    if mode in ['sign','sign-test']:
        from cryptography.hazmat.primitives.asymmetric import padding
        from cryptography.hazmat.primitives import hashes,serialization
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        completed={p.stem for p in (ROOT/'assets/shard_receipts').glob('*.json')}
        selected=[g for g in plan['groups'] if g['id'] not in completed]
        if mode=='sign-test': selected=plan['groups'][:1]
        lifetime=max(300,min(18000,int(creds['ExpiredTime']-time.time()-300)))
        urls={g['id']:cos.get_presigned_url(Bucket=BUCKET,Key=PREFIX+g['id']+'.frame',Method='PUT',Expired=lifetime,
                         Params={'x-cos-security-token':creds['Token']}) for g in selected}
        plaintext=json.dumps({'plan_sha256':plan['plan_sha256'],'urls':urls,'signed_at':time.time()}).encode()
        key=os.urandom(32);nonce=os.urandom(12)
        public=serialization.load_pem_public_key((ROOT/'transport_public.pem').read_bytes())
        envelope={'key':base64.b64encode(public.encrypt(key,padding.OAEP(mgf=padding.MGF1(hashes.SHA256()),algorithm=hashes.SHA256(),label=None))).decode(),
                  'nonce':base64.b64encode(nonce).decode(),
                  'ciphertext':base64.b64encode(AESGCM(key).encrypt(nonce,plaintext,b'vq-five-full-20261107-r1')).decode()}
        p=ROOT/('upload_test.encrypted.json' if mode=='sign-test' else 'upload.encrypted.json')
        p.write_text(json.dumps(envelope))
        return {'ok':True,'mode':mode,'groups':len(selected),'encrypted_manifest':str(p),'credential_plaintext_returned':False}
    if mode=='test-object':
        g=plan['groups'][0]
        body=cos.get_object(Bucket=BUCKET,Key=PREFIX+g['id']+'.frame')['Body'].get_raw_stream().read()
        n=struct.unpack('!I',body[:4])[0];header=json.loads(body[4:4+n]);payload=body[4+n:]
        assert hashlib.sha256(payload).hexdigest()==header['payload_sha256']
        return {'ok':True,'mode':mode,'object':g['id'],'bytes':len(body),'payload_sha256':header['payload_sha256']}
    if mode=='start-ingest':
        assert not (ROOT/'cos_ingest.pid').exists(),'Ingest already launched'
        log=(ROOT/'cos_ingest.log').open('ab',buffering=0)
        process=subprocess.Popen([sys.executable,str(ROOT/'cos_ingest.py')],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
        (ROOT/'cos_ingest.pid').write_text(str(process.pid))
        return {'ok':True,'mode':mode,'pid':process.pid}
    raise ValueError('Unknown mode')

def safe_execute(mode):
    try: return execute(mode)
    except Exception as exc:
        result={'ok':False,'mode':mode,'error_type':type(exc).__name__}
        if hasattr(exc,'get_error_code'): result['error_code']=exc.get_error_code()
        return result
