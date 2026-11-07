"""Start a detached source uploader; decrypt capabilities only in server pipes."""
import json,os,subprocess,sys
from pathlib import Path
root=Path('/data/lihy/NeuroLM/outputs/vq-five-full-20261107-r1')
encrypted=sys.stdin.buffer.read()
decoder=subprocess.Popen(['python3','/data/lihy/NeuroLM/outputs/vq_transfer_crypto.py','decrypt'],
                         stdin=subprocess.PIPE,stdout=subprocess.PIPE)
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
log=(root/'cos_upload.log').open('ab',buffering=0)
uploader=subprocess.Popen(['/data/lihy/NeuroLM/envs/neurolm-py311/bin/python','/data/lihy/NeuroLM/outputs/upload_vq_cos.py'],
                          stdin=decoder.stdout,stdout=log,stderr=log,env=env,start_new_session=True)
decoder.stdout.close();decoder.stdin.write(encrypted);decoder.stdin.close()
assert decoder.wait(timeout=90)==0
(root/'cos_upload.pid').write_text(str(uploader.pid))
print(json.dumps({'started':True,'pid':uploader.pid,'log':str(root/'cos_upload.log')}))
