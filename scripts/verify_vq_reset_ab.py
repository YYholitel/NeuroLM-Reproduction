"""Verify actual completion and matched initialization/data streams of both arms."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def verify(root):
    reports={}
    streams={}
    configs={}
    initial={}
    for arm in ['reset_off','reset_on']:
        out=root/'runs'/arm
        complete=json.loads((out/'completed.json').read_text())
        assert complete['steps']==500 and complete['world_size']==16,complete
        rows=[json.loads(line) for line in (out/'metrics.jsonl').read_text().splitlines()]
        assert len(rows)==500 and [r['iter'] for r in rows]==list(range(500))
        assert all(math.isfinite(r[k]) for r in rows for k in
                   ['total_loss','reconstruction','codebook','adversarial'])
        streams[arm]=[]
        for rank in range(16):
            data=(out/f'input_stream_rank-{rank}.jsonl').read_bytes()
            assert len(data.splitlines())==500
            streams[arm].append(hashlib.sha256(data).hexdigest())
        initial[arm]=json.loads((out/'initial_state.jsonl').read_text())['model_sha256']
        configs[arm]=json.loads((out/'run_config.json').read_text())
        keys=['total_loss','reconstruction','codebook','adversarial']
        average=lambda window:{k:sum(r[k] for r in window)/len(window) for k in keys}
        reports[arm]=dict(steps=500,first_50_mean=average(rows[:50]),last_50_mean=average(rows[-50:]),
                          last_batch_usage=rows[-1]['code_usage'],
                          last_50_usage=rows[-1]['rolling_code_usage'],
                          dead_code_resets=sum(r['dead_code_resets'] for r in rows),
                          lr_first=rows[0]['lr'],lr_last=rows[-1]['lr'],
                          alpha_last=rows[-1]['alpha'])
    assert streams['reset_off']==streams['reset_on'],'A/B training inputs differ'
    assert initial['reset_off']==initial['reset_on'],'A/B initialization differs'
    differences={k:[configs['reset_off'].get(k),configs['reset_on'].get(k)]
                 for k in set(configs['reset_off'])|set(configs['reset_on'])
                 if configs['reset_off'].get(k)!=configs['reset_on'].get(k)}
    assert set(differences)=={'out_dir','vq_dead_code_patience'},differences
    assert reports['reset_off']['dead_code_resets']==0
    result=dict(ok=True,arms=reports,initial_model_sha256=initial['reset_off'],
                matching_input_streams_all_16_ranks=True,input_stream_sha256=streams['reset_off'],
                configuration_differences=differences,schedule='first 500 updates of the original 50-epoch / 5-epoch-warmup schedule')
    (root/'comparison_verified.json').write_text(json.dumps(result,indent=2))
    print('AB_VERIFIED='+json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--root',type=Path,required=True)
    verify(parser.parse_args().root)
