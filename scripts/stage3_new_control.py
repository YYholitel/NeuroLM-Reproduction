"""Scoped TI-ONE VQ smoke submission/status actions; credentials stay in memory."""
import os,sys,json,io,contextlib
from pathlib import Path
ROOT=Path('/mnt/cfs-omni/lhy/neurolm-reproduction/stage3-new-pretrain-20261109-r1')
def execute(mode):
    assert mode in ('dry-run','submit','status','logs','details','events','resources')
    if not os.environ.get('QCLOUD_CONTAINER_INSTANCE_CREDENTIALS_URL'):
        return {'ok':False,'reason':'NOTEBOOK_STS_ENTRY_MISSING'}
    sys.path.insert(0,'/mnt/cfs-omni/chr/video_reasoning/cvpr2027/skills/tione-platform-ops')
    import tione_ops as ops
    from tikit.client import Client
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        ops._clear_proxy()
        creds=ops._fetch_sts_creds()
    if not creds: return {'ok':False,'reason':'STS_FETCH_FAILED'}
    client=Client(secret_id=creds['TmpSecretId'],secret_key=creds['TmpSecretKey'],token=creds['Token'],region='ap-beijing')
    if mode=='resources':
        groups=json.loads(client.describe_resource_groups(limit=20,search_word='rsg-b227f5pp').to_json_string())
        group=next(g for g in groups['ResourceGroupSet'] if g['ResourceGroupId']=='rsg-b227f5pp')
        eligible=[n for n in group['InstanceSet'] if n['InstanceStatus']=='RUNNING'
                  and n['AvailableResource']['Gpu']>=200 and n['AvailableResource']['Cpu']>=28000
                  and n['AvailableResource']['Memory']>=224*1024]
        return dict(ok=True,resource_group_id=group['ResourceGroupId'],eligible_two_gpu_nodes=len(eligible),
                    required_nodes=1,capacity_available=len(eligible)>=1,request_id=groups.get('RequestId'))
    spec=json.loads((ROOT/'task.json').read_text())
    kwargs=ops.build_task_kwargs(spec)
    def scoped_tasks():
        from tikit.tencentcloud.tione.v20211111.models import Filter
        f=Filter(); f.Name='Name'; f.Values=[spec['name']]
        response=client.describe_training_tasks(filters=[f],limit=50,offset=0)
        return [dict(name=t.Name,id=t.Id,status=t.Status) for t in (response.TrainingTaskSet or [])]
    tasks=scoped_tasks()
    matching=[r for r in tasks if r['name']==spec['name']]
    if mode=='dry-run':
        return {'ok':True,'mode':mode,'spec':spec,'matching_tasks':matching,
                'active_matching_tasks':sum(r['status'] not in ('SUCCEED','FAILED','STOPPED','DELETED','TERMINATED') for r in matching)}
    if mode=='submit' and not matching:
        assert json.loads((ROOT/'preflight.json').read_text())['ok']
        assert not (ROOT/'train').exists()
        from qcloud_cos import CosConfig,CosS3Client
        cos=CosS3Client(CosConfig(Region='ap-beijing',SecretId=creds['TmpSecretId'],SecretKey=creds['TmpSecretKey'],Token=creds['Token']))
        bucket,key=spec['cos_output'].split('/',1)
        cos.put_object(Bucket=bucket,Key=key,Body=b'')
        response=client.create_training_task(**kwargs)
        (ROOT/'submitted.json').write_text(json.dumps({'id':response.Id,'name':spec['name']}))
        matching=[r for r in scoped_tasks() if r['id']==response.Id]
        return {'ok':bool(matching),'mode':mode,'submitted_id':response.Id,'tasks':matching}
    if not matching: return {'ok':True,'mode':mode,'tasks':[]}
    row=matching[0]
    if mode=='events':
        from tikit.tencentcloud.tione.v20211111 import models as M
        detail=client.describe_training_task(row['id']).TrainingTaskDetail
        req=M.DescribeEventsRequest(); req.Service='TRAIN'; req.ServiceId=detail.LatestInstanceId; req.Limit=20
        events=json.loads(client._tione_client.DescribeEvents(req).to_json_string())
        import re
        blockers=[]
        ids=set()
        for event in events.get('Events',[]):
            if 'waiting in queue for' in event.get('Message',''):
                ids.update(re.findall(r'train-\d{16,}',event['Message'].split('waiting in queue for',1)[1]))
        for task_id in sorted(ids):
            blocked_by=client.describe_training_task(task_id).TrainingTaskDetail
            blockers.append({k:getattr(blocked_by,k,None) for k in ['Id','Status','Message','RuntimeInSeconds','CreateTime','StartTime','ResourceGroupId']})
        req=M.DescribeTrainingTaskSecondaryStatusRequest(); req.Id=row['id']
        secondary=json.loads(client._tione_client.DescribeTrainingTaskSecondaryStatus(req).to_json_string())
        return {'ok':True,'mode':mode,'task':row,'events':events,'secondary':secondary,'blocking_tasks':blockers}
    if mode=='details':
        detail=json.loads(client.describe_training_task(row['id']).to_json_string())
        groups=json.loads(client.describe_resource_groups(limit=20,search_word=spec['resource']['resource_group_id']).to_json_string())
        chosen={}
        for key,value in groups.items():
            if isinstance(value,list):
                chosen[key]=[{k:v for k,v in item.items() if k!='InstanceSet'} for item in value if spec['resource']['resource_group_id'] in json.dumps(item)]
        return {'ok':True,'mode':mode,'task':row,'detail':detail,'resource_group':chosen}
    if mode=='logs':
        pods=client.describe_training_task_pods(row['id'])
        return {'ok':True,'mode':mode,'task':row,'pods':json.loads(pods.to_json_string())}
    return {'ok':True,'mode':mode,'tasks':matching}

def safe_execute(mode):
    try: return execute(mode)
    except Exception as exc:
        result={'ok':False,'mode':mode,'error_type':type(exc).__name__}
        if hasattr(exc,'get_code'): result['error_code']=exc.get_code()
        return result
