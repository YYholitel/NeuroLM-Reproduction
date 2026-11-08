"""Measure text behavior before/after the author's GPT-2 vocabulary expansion."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
import torch.nn.functional as F

p=argparse.ArgumentParser(__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
sys.path.insert(0,str(a.root/'code'));torch.set_num_threads(4);torch.manual_seed(42)
from model.model import GPT
model=GPT.from_pretrained('gpt2',dict(dropout=0.),local_path=str(a.root/'gpt2')).eval()
data=np.memmap(a.root/'assets/text/val.bin',dtype=np.uint16,mode='r')
x=torch.from_numpy(np.array(data[:1024],dtype=np.int64)).unsqueeze(0)
y=torch.from_numpy(np.array(data[1:1025],dtype=np.int64)).unsqueeze(0)
with torch.no_grad():
 original,base_loss,base_acc=model(x_text=x,y_text=y)
 old_weights=model.lm_head.weight.detach().clone()
 model.enlarge_wte(50304);model.enlarge_lm_head(58496)
 expanded,expanded_loss,expanded_acc=model(x_text=x,y_text=y)
 assert torch.equal(old_weights,model.lm_head.weight[:50257])
 torch.testing.assert_close(original,expanded[:,:,:50257],rtol=0,atol=0)
 old_logmass=original.logsumexp(-1);new_logmass=expanded[:,:,50257:].logsumexp(-1)
 report=dict(original_text_loss=float(base_loss),expanded_text_loss=float(expanded_loss),
             original_accuracy=float(base_acc),expanded_accuracy=float(expanded_acc),
             pretrained_logits_exactly_preserved=True,old_logsumexp_mean=float(old_logmass.mean()),
             new_logsumexp_mean=float(new_logmass.mean()),new_classes_win_fraction=float((new_logmass>old_logmass).float().mean()),
             old_logits_mean=float(original.mean()),new_logits_mean=float(expanded[:,:,50257:].mean()),
             text_source='fixed first 1024 validation-bin tokens; CPU fp32')
(a.root/'gpt2_expansion_diagnostic.json').write_text(json.dumps(report,indent=2));print('GPT2_EXPANSION='+json.dumps(report),flush=True)
