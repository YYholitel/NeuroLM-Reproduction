"""Compare only audited, matched task/split predictions; no model selection on test results."""
import json
from pathlib import Path
ROOT=Path('/mnt/cfs-omni/lhy/neurolm-reproduction/stage3-new-pretrain-20261109-r1')
reports={name:json.loads((ROOT/f'eval-{name}/summary.json').read_text()) for name in ('new','official')}
for name in reports:
    audit=json.loads((ROOT/f'audit-{name}.json').read_text())
    assert audit['status']=='passed' and audit['total_samples']==201240 and len(audit['splits'])==10
new,old=reports['new'],reports['official']
for field in ('precision','world_size','batch_size_per_rank','mode','code_sha256','script_sha256'):
    assert new[field]==old[field],field
assert old['provenance']['neurolm_checkpoint_sha256']=='a10a23f6f4a69058c996c01096d47b0806dafd83a0d0ca6506e274a5c6372174'
assert new['provenance']['neurolm_checkpoint_sha256']==json.loads((ROOT/'checkpoint_audit.json').read_text())['checkpoint_sha256']
old_splits={(s['task'],s['split']):s for s in old['results']};rows=[]
for s in new['results']:
    o=old_splits[(s['task'],s['split'])]
    for field in ('selected_file_list_sha256','candidate_token_ids','answers','available_samples'):assert s[field]==o[field],field
    metrics={}
    for key in ('accuracy','balanced_accuracy','weighted_f1','cohen_kappa','roc_auc'):
        a,b=s['metrics'][key],o['metrics'][key]
        metrics[key]=dict(new=a,official=b,difference=None if a is None or b is None else a-b)
    rows.append(dict(task=s['task'],split=s['split'],samples=s['metrics']['samples'],metrics=metrics,
                     invalid_new=s['metrics']['parse_failures'],invalid_official=o['metrics']['parse_failures']))
result=dict(status='completed',training_protocol='matched five-task one-epoch seed-42 instruction tuning',
            initialization_comparison='our VQ plus full-20 stage-two checkpoint versus official NeuroLM-B initialization',
            paper_comparable=False,evaluation_samples_per_model=201240,results=rows,
            checkpoint_new=new['provenance'],checkpoint_official=old['provenance'])
(ROOT/'comparison.json').write_text(json.dumps(result,indent=2));print('COMPARISON='+json.dumps(result),flush=True)
