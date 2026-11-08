"""Full 20-epoch stage-two execution using the author's NeuroLM/GPT objective and models."""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from dataset import PickleLoader
from model.model import GPTConfig
from model.model_neurolm import NeuroLM
from model.model_neural_transformer import NTConfig
from model.model_vq import VQ
from vq_scheduler import cosine_scheduler


def write_json(path, value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.partial')
    temporary.write_text(json.dumps(value,indent=2));os.replace(temporary,path)


def append_json(path, value):
    with Path(path).open('a') as stream:stream.write(json.dumps(value)+'\n')


def state_hash(module):
    digest=hashlib.sha256()
    for name,value in sorted(module.state_dict().items()):
        array=value.detach().cpu().contiguous().numpy()
        digest.update(name.encode());digest.update(str(array.shape).encode());digest.update(array.tobytes())
    return digest.hexdigest()


@torch.no_grad()
def build_targets(tokenizer, x, chans, times, mask, num_chans, num_tokens, vocab_size):
    indices=tokenizer.get_codebook_indices(x,chans,times,mask)
    assert torch.equal(mask.sum(1).cpu(),num_tokens.cpu())
    assert ((indices[mask]>=0)&(indices[mask]<tokenizer.quantize.num_tokens)).all()
    targets=torch.full(x.shape[:2],-1-vocab_size,dtype=torch.long,device=x.device)
    for i,(channels,tokens) in enumerate(zip(num_chans.tolist(),num_tokens.tolist())):
        assert 0<channels<tokens<=x.shape[1] and tokens%channels==0
        targets[i,:tokens-channels]=indices[i,channels:tokens]
    assert (targets>=0).sum()>0
    return targets


class JointStep(torch.nn.Module):
    """One DDP forward contains both original EEG and text forward graphs."""
    def __init__(self,network):
        super().__init__();self.network=network

    def forward(self,x,y,chans,times,mask,gpt_mask,text,labels):
        eeg_loss,eeg_log,_=self.network(x,y,None,None,chans,times,mask,eeg_mask=gpt_mask)
        text_loss,text_log,_=self.network(None,None,text,labels)
        diagnostics=torch.tensor([eeg_log['train/loss'],text_log['train/loss'],
                                  eeg_log['train/accuracy'],text_log['train/accuracy']],device=x.device)
        return eeg_loss+text_loss,diagnostics


def run(args):
    assert args.max_updates==19140 and args.gradient_accumulation_steps==1
    assert not args.wandb_log and not args.compile
    assert args.epochs==20 and args.warmup_epochs==2 and args.block_size==1024
    rank=int(os.environ['RANK']);local_rank=int(os.environ['LOCAL_RANK']);world=int(os.environ['WORLD_SIZE'])
    assert world==16 and args.eeg_batch_size*world==480 and args.text_batch_size*world==32
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=10))
    device=torch.device('cuda',local_rank);torch.manual_seed(args.seed+rank)
    torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    assert torch.cuda.is_bf16_supported()
    ctx=lambda:torch.autocast('cuda',dtype=torch.bfloat16)
    out=Path(args.out_dir);out.mkdir(parents=True,exist_ok=True)
    checkpoint_dir=out/'checkpoints/NeuroLM-B'
    if rank==0:
        assert not checkpoint_dir.exists(),'Full pretraining requires a fresh output directory'
        checkpoint_dir.mkdir(parents=True)
    dist.barrier()

    ready=json.loads(Path(args.data_ready_manifest).read_text())
    assert ready['ready'] and ready['total_samples']==459656 and not ready['diagnostic']
    train_files=[dict(entry,shard=str(Path(args.dataset_dir)/entry['shard']))
                 for entry in map(json.loads,Path(args.train_manifest).read_text().splitlines())]
    assert len(train_files)==459656
    train=PickleLoader(train_files,GPT_training=True)
    sampler=DistributedSampler(train,num_replicas=world,rank=rank,seed=args.seed,shuffle=True)
    loader=DataLoader(train,sampler=sampler,batch_size=args.eeg_batch_size,num_workers=args.num_workers,
                      pin_memory=True,drop_last=True,persistent_workers=args.num_workers>0)
    assert len(loader)==957

    checkpoint=Path(args.tokenizer_path)
    saved=torch.load(checkpoint,map_location='cpu',weights_only=False,mmap=True)
    assert saved['epoch']==4 and saved['iter_num']==4485 and saved['loss_normalization']=='global_valid_elements'
    state={key.removeprefix('_orig_mod.'):value for key,value in saved['model'].items()}
    codebook_shape=list(state['VQ.quantize.embedding.weight'].shape)
    assert codebook_shape==[8192,128]
    options=dict(saved['vq_options'],n_embed=codebook_shape[0],embed_dim=codebook_shape[1])
    tokenizer=VQ(NTConfig(**saved['encoder_args']),NTConfig(**saved['decoder_args']),**options).eval()
    tokenizer.load_state_dict({key[3:]:value for key,value in state.items() if key.startswith('VQ.')},strict=True)
    tokenizer.requires_grad_(False);tokenizer.to(device)
    assert bool(tokenizer.quantize.embedding.initted)
    del saved,state
    network=NeuroLM(GPTConfig(),str(checkpoint),init_from='gpt2',eeg_vocab_size=codebook_shape[0],
                    gpt2_init_dir=args.gpt2_init_dir).to(device)
    assert network.GPT2.lm_head.out_features==network.GPT2.config.vocab_size+codebook_shape[0]
    assert all(not parameter.requires_grad for parameter in network.tokenizer.parameters())
    optimizer=network.configure_optimizers(args.weight_decay,args.learning_rate,(args.beta1,args.beta2),'cuda')
    frozen_ids={id(parameter) for parameter in network.tokenizer.parameters()}
    assert not any(id(parameter) in frozen_ids for group in optimizer.param_groups for parameter in group['params'])
    joint=DDP(JointStep(network),device_ids=[local_rank],broadcast_buffers=False)
    network=joint.module.network
    lr_values=cosine_scheduler(args.learning_rate,args.min_lr,args.epochs,len(loader),warmup_epochs=args.warmup_epochs)
    text_generator=torch.Generator().manual_seed(args.seed+910000+rank)

    def text_batch(split,generator):
        data=np.memmap(Path(args.text_dir)/(split+'.bin'),dtype=np.uint16,mode='r')
        offsets=torch.randint(len(data)-args.block_size,(args.text_batch_size,),generator=generator)
        x=torch.stack([torch.from_numpy(np.array(data[i:i+args.block_size],dtype=np.int64)) for i in offsets])
        y=torch.stack([torch.from_numpy(np.array(data[i+1:i+1+args.block_size],dtype=np.int64)) for i in offsets])
        assert x.max()<50257 and y.max()<50257
        return x.pin_memory().to(device,non_blocking=True),y.pin_memory().to(device,non_blocking=True),offsets

    def move(batch):
        x,chans,times,mask,gpt_mask,num_chans,num_tokens=batch
        return (x.to(device,non_blocking=True),chans.to(device,non_blocking=True),times.to(device,non_blocking=True),
                mask.to(device,non_blocking=True),gpt_mask.to(device,non_blocking=True),num_chans,num_tokens)

    initial={}
    if rank==0:
        initial=dict(frozen_encoder_sha256=state_hash(network.tokenizer),target_encoder_sha256=state_hash(tokenizer.encoder),
                     quantizer_sha256=state_hash(tokenizer.quantize),gpt_block_sha256=state_hash(network.GPT2.transformer.h[0]))
        assert initial['frozen_encoder_sha256']==initial['target_encoder_sha256']
        write_json(out/'initial_state.json',initial)
        write_json(out/'run_config.json',dict(vars(args),world_size=world,train_samples=len(train),validation_samples=0,
                                             steps_per_epoch=len(loader),codebook_shape=codebook_shape,
                                             model_args=vars(network.GPT2.config),global_eeg_batch=480,global_text_batch=32,
                                             training_loss_normalization='author DDP mean of per-rank EEG CE plus text CE',
                                             validation_loss_normalization=None, total_updates=args.max_updates,
                                             frozen_encoder_requires_grad=False, frozen_encoder_in_optimizer=False,
                                             start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
    network.train();network.tokenizer.eval()
    optimizer.zero_grad(set_to_none=True);step_number=0

    def verify_frozen(step):
        if rank!=0:return None
        after=dict(frozen_encoder_sha256=state_hash(network.tokenizer),target_encoder_sha256=state_hash(tokenizer.encoder),
                   quantizer_sha256=state_hash(tokenizer.quantize),gpt_block_sha256=state_hash(network.GPT2.transformer.h[0]))
        assert all(initial[key]==after[key] for key in ['frozen_encoder_sha256','target_encoder_sha256','quantizer_sha256'])
        assert initial['gpt_block_sha256']!=after['gpt_block_sha256'],'GPT weights did not train'
        report=dict(step=step,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    frozen_state_unchanged=True,gpt_weights_updated=True,initial=initial,final=after)
        write_json(out/'frozen_health.json',report)
        print('PRETRAIN_FROZEN_HEALTH='+json.dumps(report),flush=True)
        return after

    for epoch in range(args.epochs):
        sampler.set_epoch(epoch);ordered_indices=list(iter(sampler))
        epoch_sums=np.zeros(2);epoch_started=time.monotonic()
        for step,batch in enumerate(loader):
            started=time.monotonic();lr=float(lr_values[step_number])
            for group in optimizer.param_groups:group['lr']=lr
            x,chans,times,mask,gpt_mask,num_chans,num_tokens=move(batch)
            with torch.no_grad(),ctx():
                targets=build_targets(tokenizer,x,chans,times,mask,num_chans,num_tokens,network.GPT2.config.vocab_size)
            text,labels,offsets=text_batch('train',text_generator)
            with ctx():loss,diagnostics=joint(x,targets,chans,times,mask,gpt_mask,text,labels)
            finite=torch.isfinite(loss).to(torch.int32);dist.all_reduce(finite,op=dist.ReduceOp.MIN)
            assert int(finite),'Nonfinite stage-two loss'
            loss.backward()
            grad_norm=torch.nn.utils.clip_grad_norm_(network.parameters(),args.grad_clip)
            finite=torch.isfinite(grad_norm).to(torch.int32);dist.all_reduce(finite,op=dist.ReduceOp.MIN)
            assert int(finite),'Nonfinite stage-two gradient'
            assert all(parameter.grad is None for parameter in network.tokenizer.parameters())
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            dist.all_reduce(diagnostics);diagnostics/=world
            counts=torch.tensor([int((targets>=0).sum()),labels.numel()],device=device,dtype=torch.int64);dist.all_reduce(counts)
            step_number+=1
            if rank==0:epoch_sums+=diagnostics[:2].cpu().numpy()
            if step_number%100==0:
                verify_frozen(step_number)
                dist.barrier()
            if rank==0:
                row=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),step=step_number,epoch=epoch,lr=lr,eeg_loss=float(diagnostics[0]),text_loss=float(diagnostics[1]),
                         eeg_accuracy=float(diagnostics[2]),text_accuracy=float(diagnostics[3]),
                         eeg_targets=int(counts[0]),text_targets=int(counts[1]),grad_norm=float(grad_norm),
                         step_seconds=time.monotonic()-started)
                append_json(out/'metrics.jsonl',row)
                if step_number%args.log_interval==0:print('PRETRAIN_STEP='+json.dumps(row),flush=True)
            append_json(out/f'input_stream_rank-{rank}.jsonl',dict(step=step_number,
                eeg_indices=ordered_indices[step*args.eeg_batch_size:(step+1)*args.eeg_batch_size],text_offsets=offsets.tolist()))
        dist.barrier()
        after=verify_frozen(step_number)
        if rank==0:
            saved=dict(model=network.state_dict(),optimizer=optimizer.state_dict(),model_args=vars(network.GPT2.config),
                       iter_num=step_number,epoch=epoch,epoch_complete=True,tokenizer_checkpoint=str(checkpoint),
                       tokenizer_shape=codebook_shape,protocol='full 20-epoch multi-channel autoregressive pretraining',
                       run_args=vars(args))
            temporary=checkpoint_dir/'ckpt.pt.partial'
            torch.save(saved,temporary);os.replace(temporary,checkpoint_dir/'ckpt.pt')
            report=dict(epoch_completed=epoch+1,step=step_number,eeg_loss=float(epoch_sums[0]/len(loader)),
                        text_loss=float(epoch_sums[1]/len(loader)),seconds=time.monotonic()-epoch_started,
                        frozen_state_unchanged=True,gpt_weights_updated=True,
                        utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            append_json(out/'epochs.jsonl',report)
            write_json(out/'progress.json',report)
            print('PRETRAIN_EPOCH_COMPLETE='+json.dumps(report),flush=True)
        dist.barrier()
    assert step_number==args.max_updates and epoch+1==20
    if rank==0:
        write_json(out/'completed.json',dict(steps=step_number,world_size=world,epochs_completed=epoch+1,
                                             partial_epoch=False,frozen_state_unchanged=True,gpt_weights_updated=True,
                                             initial=initial,final=after,utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
        print('PRETRAIN_FULL20_COMPLETE',flush=True)
    dist.barrier();dist.destroy_process_group()
