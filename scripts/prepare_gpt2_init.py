"""Download pinned public GPT-2 initialization; verify the published weights hash."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import requests

REVISION='607a30d783dfa663caf39e06633721c8d4cfcd7e'
WEIGHTS_SHA256='248dfc3911869ec493c76e65bf2fcf7f615828b0254c12b473182f0f81d3a707'


def main():
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args();args.output_dir.mkdir(parents=True,exist_ok=True)
    session=requests.Session();session.trust_env=False
    hashes={}
    for name in ['config.json','generation_config.json','model.safetensors']:
        path=args.output_dir/name
        if path.exists() and name=='model.safetensors':
            with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
            if digest==WEIGHTS_SHA256:hashes[name]=digest;continue
        for endpoint in ['https://huggingface.co','https://hf-mirror.com']:
            try:
                url=f'{endpoint}/openai-community/gpt2/resolve/{REVISION}/{name}'
                temporary=path.with_suffix(path.suffix+'.partial')
                digest=hashlib.sha256()
                with session.get(url,timeout=(15,60),stream=True) as response:
                    response.raise_for_status()
                    with temporary.open('wb') as stream:
                        for chunk in response.iter_content(8*1024*1024):
                            stream.write(chunk);digest.update(chunk)
                if name=='model.safetensors':assert digest.hexdigest()==WEIGHTS_SHA256
                os.replace(temporary,path);hashes[name]=digest.hexdigest()
                print(json.dumps({'downloaded':name,'bytes':path.stat().st_size,'sha256':hashes[name]}),flush=True)
                break
            except Exception as exc:
                print(json.dumps({'endpoint':endpoint,'file':name,'error':type(exc).__name__}),flush=True)
        else:raise RuntimeError('Unable to download verified GPT-2 file '+name)
    report=dict(repo='openai-community/gpt2',revision=REVISION,files=hashes,
                weights_hash_source='https://huggingface.co/openai-community/gpt2/blob/main/model.safetensors')
    (args.output_dir/'verified.json').write_text(json.dumps(report,indent=2))
    print('GPT2_INIT_READY='+json.dumps(report),flush=True)


if __name__=='__main__':main()
