"""Frozen VQ audit on the same 160 held-out windows used by the stage-two probe."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch

p=argparse.ArgumentParser(__doc__)
p.add_argument('--runtime',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
p.add_argument('--validation-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();torch.set_num_threads(4);sys.path.insert(0,str(a.runtime))
from dataset import PickleLoader
from model.model_vq import VQ
from model.model_neural_transformer import NTConfig
from vq_monitor import usage_summary
manifest=json.loads(a.validation_manifest.read_text());assert manifest['samples']==160
for f in manifest['files']:assert hashlib.sha256((a.validation_manifest.parent/f['name']).read_bytes()).hexdigest()==f['sha256']
checkpoint=torch.load(a.checkpoint,map_location='cpu',weights_only=False,mmap=True)
assert checkpoint['epoch']==4 and checkpoint['iter_num']==4485 and checkpoint['loss_normalization']=='global_valid_elements'
model=VQ(NTConfig(**checkpoint['encoder_args']),NTConfig(**checkpoint['decoder_args']),**checkpoint['vq_options']).eval()
state={key.removeprefix('_orig_mod.'):value for key,value in checkpoint['model'].items()}
model.load_state_dict({key[3:]:value for key,value in state.items() if key.startswith('VQ.')},strict=True)
model.requires_grad_(False);del checkpoint,state
before={key:value.clone() for key,value in model.quantize.state_dict().items()}
loader=PickleLoader([a.validation_manifest.parent/f['name'] for f in manifest['files']])
capacity=model.quantize.num_tokens;counts=torch.zeros(capacity,dtype=torch.int64);groups={};samples=[]
with torch.inference_mode():
 for i,entry in enumerate(manifest['files']):
  batch=[value.unsqueeze(0) for value in loader[i]]
  _,_,log=model(*batch)
  hist=log['val/valid_code_counts'];assert hist.numel()==capacity and hist.sum()==batch[-1].sum()
  counts+=hist;dataset=entry['name'].split('/')[1];groups.setdefault(dataset,torch.zeros_like(counts)).add_(hist)
  samples.append(dict(name=entry['name'],dataset=dataset,tokens=int(hist.sum()),
                      reconstruction=float(log['val/rec_freq_loss']+log['val/rec_raw_loss']),
                      frequency=float(log['val/rec_freq_loss']),raw=float(log['val/rec_raw_loss']),commitment=float(log['val/quant_loss'])))
  if (i+1)%32==0:print(json.dumps(dict(audited=i+1,samples=160)),flush=True)
assert all(torch.equal(value,model.quantize.state_dict()[key]) for key,value in before.items())
def summarize(selected,histogram):
 tokens=sum(row['tokens'] for row in selected)
 result={key:sum(row['tokens']*row[key] for row in selected)/tokens for key in ['reconstruction','frequency','raw','commitment']}
 return dict(samples=len(selected),**result,**usage_summary(histogram),normalized_perplexity=usage_summary(histogram)['perplexity']/capacity)
report=dict(checkpoint=str(a.checkpoint),codebook_shape=list(model.quantize.embedding.weight.shape),
            validation_manifest_sha256=hashlib.sha256(a.validation_manifest.read_bytes()).hexdigest(),
            frozen_quantizer_unchanged=True,split='balanced held-out probe; not full population validation',
            overall=summarize(samples,counts),per_dataset={name:summarize([r for r in samples if r['dataset']==name],hist) for name,hist in groups.items()},
            sample_metrics=samples,counts=counts.tolist())
a.output.write_text(json.dumps(report,indent=2));print('HELDOUT_AUDIT='+json.dumps({k:v for k,v in report.items() if k not in ['sample_metrics','counts']}),flush=True)
