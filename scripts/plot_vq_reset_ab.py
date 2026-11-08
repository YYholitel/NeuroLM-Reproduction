"""Plot measured matched 500-step VQ reset-off/reset-on experiments."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--input-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    styles={'reset_off':('#2678b2','Reset OFF'),'reset_on':('#d77722','Reset ON')}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'svg.hashsalt':'vq-reset-ab-500-20261108'})
    fig,axes=plt.subplots(2,3,figsize=(15,8.3),constrained_layout=True)
    rows_by_arm={}
    for arm,(color,label) in styles.items():
        rows=[json.loads(line) for line in (args.input_dir/f'{arm}_metrics.jsonl').read_text().splitlines()]
        assert len(rows)==500 and [r['iter'] for r in rows]==list(range(500))
        rows_by_arm[arm]=rows
        x=np.arange(1,501)
        for ax,key,title in zip(axes[0],['reconstruction','codebook','adversarial'],
                                ['Reconstruction loss','Commitment loss','Adversarial loss (EEG + text)']):
            y=np.array([r[key] for r in rows])
            smooth=np.array([y[max(0,i-19):i+1].mean() for i in range(len(y))])
            ax.plot(x,y,color=color,alpha=.2,lw=.7)
            ax.plot(x,smooth,color=color,label=label,lw=1.7)
            ax.set_title(title)
        usage=[100*r['rolling_code_usage']['active_fraction'] for r in rows]
        perp=[r['rolling_code_usage']['perplexity'] for r in rows]
        axes[1,0].plot(x,usage,color=color,label=label,lw=1.8)
        axes[1,1].plot(x,perp,color=color,label=label,lw=1.8)
        axes[1,2].step(x,np.cumsum([r['dead_code_resets'] for r in rows]),where='post',color=color,label=label,lw=1.8)
    axes[1,0].set(title='Active codes / 8192 (%)',ylim=(0,100),ylabel='Rolling 50 updates; valid EEG only')
    axes[1,1].set(title='Codebook perplexity',ylim=(0,8192),ylabel='Rolling 50 updates; valid EEG only')
    axes[1,2].set(title='Cumulative dead-code replacements',ylabel='Replacement events, not unique codes')
    for ax in axes.flat:
        ax.axvline(100,color='#777777',ls=':',lw=1,alpha=.6)
        ax.set(xlabel='Optimizer update',xlim=(1,500))
        ax.grid(alpha=.18)
    axes[0,0].legend(frameon=False)
    fig.suptitle('NeuroLM VQ: dead-code reset OFF vs ON | 500 updates each | 2 x 8 H20',fontsize=16,weight='bold')
    fig.supxlabel('Same initialization, EEG order and text stream. Loss: valid-element mean; bold = trailing 20-step mean.\n'
                  'Original 50-epoch schedule retained: these 500 updates are within warmup. Dotted line: first reset eligibility.',fontsize=10)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for suffix in ['png','svg','pdf']:
        target=args.output_dir/f'vq_reset_ab_comparison.{suffix}'
        fig.savefig(target,dpi=180)
        if suffix=='svg':
            target.write_bytes(('\n'.join(line.rstrip() for line in target.read_text().splitlines())+'\n').encode())
    early={}
    for key in ['reconstruction','codebook','adversarial']:
        a=np.array([r[key] for r in rows_by_arm['reset_off'][:99]])
        b=np.array([r[key] for r in rows_by_arm['reset_on'][:99]])
        early[key]={'max_abs_difference_before_first_reset':float(abs(a-b).max()),
                    'mean_abs_difference_before_first_reset':float(abs(a-b).mean())}
    (args.output_dir/'early_curve_agreement.json').write_text(json.dumps(early,indent=2))
    print(json.dumps(early,indent=2))
    print(args.output_dir/'vq_reset_ab_comparison.png')


if __name__=='__main__':main()
