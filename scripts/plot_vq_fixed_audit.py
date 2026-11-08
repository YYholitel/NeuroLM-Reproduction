"""Compare frozen VQ checkpoints on the same 32 diagnostic training samples."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--old',type=Path,required=True)
    parser.add_argument('--epoch1',type=Path,required=True)
    parser.add_argument('--epoch5',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args=parser.parse_args()
    reports=[json.loads(path.read_text()) for path in [args.old,args.epoch1,args.epoch5]]
    assert all(r['sample_names']==reports[0]['sample_names'] for r in reports)
    assert all(r['valid']['tokens']==4944 for r in reports)
    labels=['Previous VQ\n50 epochs','Repaired VQ\n1 epoch','Repaired VQ\n5 epochs']
    colors=['#697581','#3595b4','#dc7626']
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.hashsalt':'vq-fixed-audit-20261108'})
    fig,axes=plt.subplots(2,2,figsize=(12,8.2),constrained_layout=True)
    for ax,key,title,scale in [(axes[0,0],'active_codes','Distinct codes used',1),
                               (axes[0,1],'perplexity','Assignment perplexity',1),
                               (axes[1,0],'top5_fraction','Token share assigned to the top 5 codes (%)',100)]:
        values=[r['valid'][key]*scale for r in reports]
        bars=ax.bar(labels,values,color=colors,width=.58)
        ax.bar_label(bars,labels=[f'{v:,.0f}' if key=='active_codes' else f'{v:,.1f}' for v in values],padding=4)
        ax.set(title=title,ylim=(0,max(values)*1.22));ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    ax=axes[1,1]
    for report,label,color in zip(reports,labels,colors):
        counts=np.array(report.get('counts',report['valid'].get('counts')))
        assert counts.shape==(8192,) and counts.sum()==4944
        frequencies=np.sort(counts[counts>0])[::-1]/counts.sum()
        ax.plot(np.arange(1,len(frequencies)+1),frequencies,color=color,label=label.replace('\n',' '),lw=1.8)
    ax.set(xscale='log',yscale='log',xlabel='Code rank (most frequent first)',ylabel='Fraction of valid tokens',
           title='Frozen-checkpoint assignment distribution')
    ax.legend(frameon=False,fontsize=9);ax.grid(alpha=.18)
    fig.suptitle('NeuroLM VQ: fixed-sample collapse check',fontsize=17,weight='bold')
    fig.supxlabel('Same 32 training samples, 4,944 valid EEG tokens; padding excluded. Not held-out validation.\n'
                  'Training durations differ. These are frozen-checkpoint diagnostics, not rolling training-window unions.',fontsize=10)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    for suffix in ['png','svg','pdf']:
        path=args.output_dir/f'vq_fixed_sample_comparison.{suffix}';fig.savefig(path,dpi=180)
        if suffix=='svg':path.write_bytes(('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n').encode())
    print(json.dumps({'plotted':True,'samples':32,'valid_tokens':4944}))


if __name__=='__main__':main()
