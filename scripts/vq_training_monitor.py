"""VQ epoch summaries and repeatable probes; no changes to the training loss."""
import json
import math
from pathlib import Path
import torch
import torch.distributed as dist


def usage_summary(counts):
    counts=counts.detach().double().cpu()
    total=int(counts.sum().item())
    active=int((counts>0).sum().item())
    probabilities=counts[counts>0]/max(total,1)
    perplexity=float(torch.exp(-(probabilities*probabilities.log()).sum())) if total else 0.0
    return dict(active_codes=active,codebook_size=counts.numel(),active_fraction=active/counts.numel(),
                valid_eeg_tokens=total,perplexity=perplexity,
                largest_code_fraction=float(probabilities.max()) if total else 0.0)


def fixed_probe(vq, batch, device, autocast_context):
    """All DDP ranks must call this together: the upstream quantizer reduces stats."""
    assert bool(vq.quantize.embedding.initted.item()), 'Probe must not initialize the training codebook'
    previous_training=vq.training
    saved_buffers={name:value.clone() for name,value in vq.quantize.named_buffers()}
    target=torch.device(device)
    devices=[target.index if target.index is not None else torch.cuda.current_device()] if target.type=='cuda' else []
    try:
        vq.eval()
        with torch.random.fork_rng(devices=devices),torch.no_grad(),autocast_context:
            values=[item.to(device) for item in batch]
            _,_,log=vq(*values)
            metrics=torch.stack([log['val/rec_freq_loss'],log['val/rec_raw_loss'],log['val/quant_loss']]).double()
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(metrics);metrics/=dist.get_world_size()
        frequency,raw,codebook=metrics.tolist()
        assert all(math.isfinite(value) for value in [frequency,raw,codebook])
        return dict(reconstruction=frequency+raw,frequency=frequency,raw=raw,codebook=codebook,
                    unique_samples=len(batch[0]),evaluation=True)
    finally:
        for name,value in vq.quantize.named_buffers(): value.copy_(saved_buffers[name])
        vq.train(previous_training)


def epoch_summary(epoch, rows, counts, tail_counts, probe, baseline, history):
    window=min(100,len(rows))
    keys=['reconstruction','frequency','raw','codebook','adversarial','adversarial_eeg','adversarial_text']
    def average(selected): return {key:sum(row[key] for row in selected)/len(selected) for key in keys}
    report=dict(epoch=epoch,updates=len(rows),mean=average(rows),first_window=average(rows[:window]),
                last_window=average(rows[-window:]),window_updates=window,
                codebook_min=min(r['codebook'] for r in rows),codebook_max=max(r['codebook'] for r in rows),
                lr_first=rows[0]['lr'],lr_last=rows[-1]['lr'],alpha_first=rows[0]['alpha'],alpha_last=rows[-1]['alpha'],
                code_usage=usage_summary(counts),tail_code_usage=usage_summary(tail_counts),
                fixed_diagnostic=probe,fixed_diagnostic_baseline=baseline,
                milestones=[x for x in [1,5,20,25,30,50] if x==epoch])
    reference=baseline['reconstruction']
    report['fixed_diagnostic_relative_improvement']=1-probe['reconstruction']/max(reference,1e-12)
    flags=[]
    if epoch==1 and report['fixed_diagnostic_relative_improvement']<=0:
        flags.append('review_epoch_1_no_fixed_probe_improvement_during_warmup')
    if epoch==5 and not math.isclose(report['lr_last'],5e-5,rel_tol=1e-6):
        flags.append('warmup_peak_lr_mismatch')
    if epoch>=5 and report['tail_code_usage']['active_codes']<512:
        flags.append('review_low_recent_code_usage_fewer_than_512')
    if epoch>=5 and report['tail_code_usage']['perplexity']<128:
        flags.append('review_concentrated_code_assignments')
    if report['last_window']['codebook']>max(0.1,100*baseline['codebook']):
        flags.append('review_large_codebook_loss')
    if probe['reconstruction']>10*max(reference,1e-12):
        flags.append('review_large_fixed_probe_reconstruction')
    if 20<=epoch<=30 and len(history)>=4:
        recent=history[-4:]+[report]
        probes=[r['fixed_diagnostic']['reconstruction'] for r in recent]
        losses=[r['last_window']['reconstruction'] for r in recent]
        probe_gain=1-min(probes)/max(probes[0],1e-12)
        train_gain=1-min(losses)/max(losses[0],1e-12)
        report['five_epoch_best_improvement']=dict(fixed_probe=probe_gain,training_tail=train_gain)
        if probe_gain<0.005 and train_gain<0.005: flags.append('review_five_epoch_plateau')
    report['review_flags']=flags
    report['automatic_stop']=False
    report['interpretation']='Review flags are diagnostic heuristics, not paper thresholds or automatic stop rules.'
    return report


def append_json(path,row):
    with Path(path).open('a') as stream: stream.write(json.dumps(row)+'\n')
