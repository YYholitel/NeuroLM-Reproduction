"""Plot measured epoch-50 token assignments; no simulated or post-fix results."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.input.read_text(encoding='utf-8'))
    valid = np.array(data['valid']['counts'])
    padding = np.array(data['padding']['counts'])
    active = np.flatnonzero(valid)
    order = np.argsort(-valid, kind='stable')[:20]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.hashsalt': 'vq-audit-20261108'})
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6), constrained_layout=True)
    fig.suptitle('NeuroLM VQ: measured codebook underuse at epoch 50', fontsize=17, weight='bold')
    ax = axes[0, 0]
    ax.scatter(active, valid[active], s=14, color='#2176ae', alpha=.85)
    ax.set(xlim=(-100, 8292), ylim=(0, max(valid)*1.2), xlabel='Code index (8192 available)',
           ylabel='Assignments to valid EEG tokens', title='Only 179 codes used in 4944 valid tokens')
    ax.grid(axis='y', alpha=.2)
    ax = axes[0, 1]
    ax.bar(np.arange(20), valid[order], color='#2176ae')
    ax.set_xticks(np.arange(20), order, rotation=65, fontsize=8)
    ax.set(xlabel='Code index, ordered by valid-token count', ylabel='Valid-token count',
           title='Top 5 codes = 6.23%; perplexity = 166.93')
    ax.grid(axis='y', alpha=.2)
    ax = axes[1, 0]
    top_pad = int(np.argmax(padding))
    categories = [f'Code {top_pad}', 'All other codes']
    eeg = [valid[top_pad], valid.sum()-valid[top_pad]]
    pad = [padding[top_pad], padding.sum()-padding[top_pad]]
    ax.barh(categories, eeg, label='Valid EEG', color='#2176ae')
    ax.barh(categories, pad, left=eeg, label='Padding', color='#e69436')
    ax.set(xlabel='Token assignments in the sampled 32 windows',
           title='All 27,824 padding tokens selected code 0')
    ax.legend(loc='lower right', frameon=False)
    ax = axes[1, 1]
    names = list(data['per_dataset'])
    counts = [data['per_dataset'][name]['active_codes'] for name in names]
    ax.bar(names, counts, color='#458a70')
    ax.set(ylim=(0, 205), ylabel='Active codes (out of 8192)',
           title='Underuse appears in all five datasets')
    for i, value in enumerate(counts):
        ax.text(i, value+3, str(value), ha='center')
    ax.grid(axis='y', alpha=.2)
    fig.supxlabel('Audit: TUAB 8, TUEV 8, HMC 8, Workload 4, TUSL 4 windows. '
                  'Existing checkpoint; these are not post-fix training results.', fontsize=9)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'svg', 'pdf'):
        path = args.output.with_suffix('.'+extension)
        fig.savefig(path, dpi=180)
        if extension == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')
    print(args.output.with_suffix('.png'))


if __name__ == '__main__':
    main()
