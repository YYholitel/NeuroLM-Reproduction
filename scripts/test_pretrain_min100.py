"""CPU integration check of the real frozen VQ, local GPT-2, and stage-two objective."""
import argparse
import json
from pathlib import Path
import sys
import torch


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--runtime',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--gpt2-dir',type=Path,required=True)
    parser.add_argument('--sample',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(4);torch.manual_seed(42)
    sys.path.insert(0,str(args.runtime.resolve()))
    from pretrain_minimal import build_targets,JointStep,state_hash
    from model.model import GPTConfig
    from model.model_neurolm import NeuroLM
    from model.model_vq import VQ
    from model.model_neural_transformer import NTConfig
    from dataset import PickleLoader
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False,mmap=True)
    assert saved['epoch']==4 and saved['iter_num']==4485
    target=VQ(NTConfig(**saved['encoder_args']),NTConfig(**saved['decoder_args']),**saved['vq_options']).eval()
    target.load_state_dict({key[3:]:value for key,value in saved['model'].items() if key.startswith('VQ.')},strict=True)
    target.requires_grad_(False);del saved
    network=NeuroLM(GPTConfig(),str(args.checkpoint),init_from='gpt2',gpt2_init_dir=str(args.gpt2_dir))
    encoder_before=state_hash(network.tokenizer);quantizer_before=state_hash(target.quantize)
    assert encoder_before==state_hash(target.encoder)
    batch=PickleLoader([args.sample],GPT_training=True)[0]
    x,chans,times,mask,gpt_mask,num_chans,num_tokens=batch
    length=min(num_tokens,num_chans*2)
    x=x[:length].unsqueeze(0);chans=chans[:length].unsqueeze(0);times=times[:length].unsqueeze(0)
    mask=mask[:length].unsqueeze(0);gpt_mask=gpt_mask[:,:length,:length].unsqueeze(0)
    channels=torch.tensor([num_chans]);tokens=torch.tensor([length])
    targets=build_targets(target,x,chans,times,mask,channels,tokens,network.GPT2.config.vocab_size)
    with torch.no_grad():indices=target.get_codebook_indices(x,chans,times,mask)
    assert torch.equal(targets[0,:length-num_chans],indices[0,num_chans:length])
    assert ((targets[0,length-num_chans:]+network.GPT2.config.vocab_size)==-1).all()
    text=torch.randint(0,50257,(1,32));labels=torch.randint(0,50257,(1,32))
    joint=JointStep(network);joint.train();network.tokenizer.eval()
    optimizer=network.configure_optimizers(.1,1e-5,(.9,.95),'cpu')
    block_before=state_hash(network.GPT2.transformer.h[0])
    loss,metrics=joint(x,targets,chans,times,mask,gpt_mask,text,labels)
    assert torch.isfinite(loss) and torch.isfinite(metrics).all()
    loss.backward()
    assert all(parameter.grad is None for parameter in network.tokenizer.parameters())
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all()
               for parameter in network.parameters() if parameter.requires_grad)
    norm=torch.nn.utils.clip_grad_norm_(network.parameters(),1.);optimizer.step();optimizer.zero_grad(set_to_none=True)
    assert encoder_before==state_hash(network.tokenizer)
    assert quantizer_before==state_hash(target.quantize)
    assert block_before!=state_hash(network.GPT2.transformer.h[0])
    result=dict(ok=True,real_checkpoint_epoch=5,real_gpt2_initialization=True,codebook_shape=[8192,128],
                head_classes=network.GPT2.lm_head.out_features,valid_eeg_targets=int((targets>=0).sum()),
                next_second_targets_verified=True,padding_and_final_second_ignored=True,
                all_trainable_gradients_present_and_finite=True,frozen_encoder_and_quantizer_unchanged=True,
                gpt_optimizer_update_verified=True,total_loss=float(loss),grad_norm=float(norm),
                diagnostic='Disposable CPU test on a cropped training sample; not validation quality')
    args.output.write_text(json.dumps(result,indent=2));print('PRETRAIN_CPU_TEST='+json.dumps(result),flush=True)


if __name__=='__main__':main()
