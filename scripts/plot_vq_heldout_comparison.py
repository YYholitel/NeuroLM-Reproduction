"""Render only measured results from the paired frozen held-out VQ audits."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p=argparse.ArgumentParser(__doc__);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--compact',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
reports=[json.loads(path.read_text()) for path in [a.baseline,a.compact]]
assert reports[0]['validation_manifest_sha256']==reports[1]['validation_manifest_sha256']
assert [r['name'] for r in reports[0]['sample_metrics']]==[r['name'] for r in reports[1]['sample_metrics']]
assert all(r['frozen_quantizer_unchanged'] and r['overall']['samples']==160 for r in reports)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(13,9),layout='constrained')
colors=['#467b9c','#db7d29'];labels=['8192 x 128, patience 100','1024 x 64, patience 50']
datasets=['TUAB','TUEV','HMC','Workload','TUSL'];width=.36
for index,(report,color,label) in enumerate(zip(reports,colors,labels)):
 values=[report['overall']['reconstruction']]+[report['per_dataset'][d]['reconstruction'] for d in datasets]
 x=np.arange(6)+(index-.5)*width
 axes[0,0].bar(x,values,width,color=color,label=label)
 for xx,value in zip(x,values):axes[0,0].annotate(f'{value:.3f}',(xx,value),xytext=(0,4),textcoords='offset points',ha='center',fontsize=8)
 metrics=[report['overall']['active_fraction'],report['overall']['normalized_perplexity']]
 axes[0,1].bar(np.arange(2)+(index-.5)*width,100*np.array(metrics),width,color=color)
 for xx,value in zip(np.arange(2)+(index-.5)*width,metrics):axes[0,1].annotate(f'{100*value:.1f}%',(xx,100*value),xytext=(0,4),textcoords='offset points',ha='center',fontsize=10)
 metrics=[report['overall']['active_codes'],report['overall']['perplexity']]
 axes[1,0].bar(np.arange(2)+(index-.5)*width,metrics,width,color=color)
 for xx,value in zip(np.arange(2)+(index-.5)*width,metrics):axes[1,0].annotate(f'{value:,.1f}',(xx,value),xytext=(0,4),textcoords='offset points',ha='center',fontsize=10)
 counts=np.array(report['counts']);probabilities=np.sort(counts[counts>0])[::-1]/counts.sum()
 axes[1,1].plot(np.arange(1,len(probabilities)+1),probabilities,color=color,label=label)
axes[0,0].set(title='Held-out reconstruction loss (valid-token weighted)',xticks=np.arange(6),xticklabels=['All']+datasets,ylabel='Frequency + waveform MSE');axes[0,0].legend(fontsize=8,loc='upper left')
axes[0,1].set(title='Capacity-normalized assignment metrics',xticks=[0,1],xticklabels=['Active codes / K','Perplexity / K'],ylabel='Percent',ylim=(0,110))
axes[1,0].set(title='Absolute assignment metrics',xticks=[0,1],xticklabels=['Distinct codes','Perplexity'])
axes[1,1].set(title='Frozen code assignment distribution',xlabel='Code rank (most frequent first)',ylabel='Fraction of valid tokens',xscale='log',yscale='log')
for ax in axes.flat:ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True);ax.margins(y=.15)
fig.suptitle('NeuroLM VQ: five-epoch held-out comparison',fontsize=18)
fig.supxlabel('Same 160 held-out windows (32 per dataset), 24,672 valid tokens; both VQ checkpoints frozen.\n'
              'One seed; capacity, embedding dimension and reset patience change together. Not downstream task accuracy.',fontsize=10)
a.output_dir.mkdir(parents=True,exist_ok=True)
for suffix in ['png','svg','pdf']:fig.savefig(a.output_dir/f'vq_heldout_comparison.{suffix}',dpi=170)
plt.close(fig)
print(json.dumps(dict(plotted=True,samples=160)))
