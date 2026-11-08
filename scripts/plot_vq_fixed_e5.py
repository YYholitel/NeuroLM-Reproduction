"""Plot only recorded repaired-VQ updates, including padding-excluded code usage."""
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    rows=[json.loads(line) for line in args.input.read_bytes().splitlines(keepends=True) if line.endswith(b'\n') and line.strip()]
    if not rows:return
    steps=np.array([r['iter']+1 for r in rows]);assert np.array_equal(steps,np.arange(1,len(rows)+1))
    x=steps/897
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.hashsalt':'vq-fixed-e5-20261108'})
    fig,axes=plt.subplots(2,3,figsize=(15,8.3),constrained_layout=True)
    for ax,key,title in zip(axes[0],['reconstruction','codebook','adversarial'],
                            ['Reconstruction loss','Commitment loss','Adversarial loss (EEG + text)']):
        values=np.array([r[key] for r in rows]);assert np.isfinite(values).all()
        smooth=np.array([values[max(0,i-99):i+1].mean() for i in range(len(values))])
        ax.plot(x,values,color='#2678b2',alpha=.18,lw=.6)
        ax.plot(x,smooth,color='#2678b2',lw=1.5,label='Trailing 100-update mean')
        ax.set_title(title)
    for ax,key,scale,title in [(axes[1,0],'active_fraction',100,'Active codes / 8192 (%)'),
                               (axes[1,1],'perplexity',1,'Codebook perplexity')]:
        ax.plot(x,[scale*r['code_usage'][key] for r in rows],color='#d77722',alpha=.2,lw=.6,label='Per batch')
        ax.plot(x,[scale*r['rolling_code_usage'][key] for r in rows],color='#d77722',lw=1.2,label='Rolling 50 updates')
        ax.set(title=title,ylim=(0,100 if scale==100 else 8192))
        ax.legend(frameon=False,fontsize=8)
    axes[1,2].plot(x,[r['lr'] for r in rows],color='#458a70',lw=1.5)
    axes[1,2].set(title='Learning rate',ylim=(0,5.25e-5));axes[1,2].ticklabel_format(axis='y',style='sci',scilimits=(0,0))
    for ax in axes.flat:
        for epoch in range(1,6):ax.axvline(epoch,color='#777777',ls=':',lw=.7,alpha=.4)
        ax.set(xlabel='Epoch (897 updates per epoch)',xlim=(0,5));ax.grid(alpha=.16)
    state='5 epochs recorded' if len(rows)==4485 else f'In progress: {len(rows)} / 4485 updates'
    fig.suptitle('NeuroLM VQ | valid-token EMA + dead-code reset | 2 x 8 H20',fontsize=16,weight='bold')
    fig.supxlabel(f'{state}. All usage statistics exclude padding. Loss uses valid-element normalization.\n'
                  'Reset: 100 unused updates, checked every 50. Original 50-epoch schedule; this run stops at the end of warmup.',fontsize=10)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for suffix in ['png','svg','pdf']:
        out=args.output_dir/f'vq_fixed_e5_curves.{suffix}';fig.savefig(out,dpi=180)
        if suffix=='svg':out.write_bytes(('\n'.join(line.rstrip() for line in out.read_text().splitlines())+'\n').encode())
    meta=dict(observed_updates=len(rows),target_updates=4485,trace_complete=len(rows)==4485,
              last=rows[-1],total_dead_code_resets=sum(r['dead_code_resets'] for r in rows))
    (args.output_dir/'curve_metadata.json').write_text(json.dumps(meta,indent=2))
    print(json.dumps(dict(plotted=True,updates=len(rows))))


if __name__=='__main__':main()
