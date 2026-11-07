"""Receive the frozen VQ stream as indexed tar shards to avoid tiny CFS writes."""
import argparse,gzip,hashlib,io,json,os,struct,tarfile,time,zlib
from pathlib import Path
from prepare_five_dataset_vq import read_exact,dump_atomic

def status(root):
    plan=root/'source_plan.json'
    return {'plan_sha256':json.loads(plan.read_text())['plan_sha256'] if plan.exists() else None,
            'completed_groups':[p.stem for p in sorted((root/'shard_receipts').glob('*.json'))]}

def receive(root):
    root=root.resolve();root.mkdir(parents=True,exist_ok=True)
    receipts=root/'shard_receipts';receipts.mkdir(exist_ok=True)
    while True:
        n=struct.unpack('!I',read_exact(4))[0];assert n<10_000_000
        header=json.loads(read_exact(n));body=read_exact(header['payload_bytes'])
        assert hashlib.sha256(body).hexdigest()==header['payload_sha256']
        if header['kind']=='plan':
            plan=json.loads(zlib.decompress(body));old=status(root)
            assert not old['plan_sha256'] or old['plan_sha256']==plan['plan_sha256']
            if not (root/'source_plan.json').exists(): dump_atomic(root/'source_plan.json',plan)
            continue
        if header['kind']=='done':
            expected={g['id'] for g in plan['groups']};actual=set(status(root)['completed_groups'])
            assert actual.issubset(expected)
            result={'ready':actual==expected,'storage':'indexed_tar','plan_sha256':plan['plan_sha256'],
                    'counts':plan['counts'],'total_samples':plan['total_samples'],
                    'completed_groups':len(actual),'total_groups':len(expected),'time':time.time()}
            if result['ready']: dump_atomic(root/'transfer_complete.json',result)
            print(json.dumps(result),flush=True);return
        assert header['kind']=='group';group=header['group'];stats=header['stats']
        if group['kind']=='text':
            data=zlib.decompress(body)
            assert len(data)==group['length'] and hashlib.sha256(data).hexdigest()==stats['raw_sha256']
            p=root/group['target'];assert p.is_file() and p.stat().st_size==group['input']['bytes']
            with p.open('r+b') as f:
                f.seek(group['offset']);f.write(data);f.flush()
            storage={}
        else:
            data=gzip.decompress(body);entries=[]
            shard='shards/'+group['id']+'.tar'
            with tarfile.open(fileobj=io.BytesIO(data),mode='r:') as tar:
                for member in tar:
                    assert member.isfile() and member.name.startswith('train/'+group['dataset']+'/')
                    assert '..' not in Path(member.name).parts
                    sample=data[member.offset_data:member.offset_data+member.size]
                    entries.append({'name':member.name,'shard':shard,'offset':member.offset_data,
                                    'size':member.size,'sha256':hashlib.sha256(sample).hexdigest()})
            assert len(entries)==group['count']
            p=root/shard;p.parent.mkdir(exist_ok=True)
            tmp=p.with_suffix('.partial');tmp.write_bytes(data);os.replace(tmp,p)
            index=root/'indices'/(group['id']+'.json');dump_atomic(index,entries)
            storage={'shard':shard,'archive_bytes':len(data),'archive_sha256':hashlib.sha256(data).hexdigest()}
        dump_atomic(receipts/(group['id']+'.json'),{'id':group['id'],'plan_sha256':plan['plan_sha256'],
                    'compressed_sha256':header['payload_sha256'],'compressed_bytes':len(body),**stats,**storage})
        done=len(list(receipts.glob('*.json')))
        if done%20==0:
            result={'completed_groups':done,'total_groups':len(plan['groups']),'last':group['id'],'time':time.time()}
            dump_atomic(root/'transfer_progress.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('mode',choices=['status','receive']);parser.add_argument('--output-root',required=True)
    args=parser.parse_args();root=Path(args.output_root)
    if args.mode=='status': print(json.dumps(status(root)))
    else: receive(root)
