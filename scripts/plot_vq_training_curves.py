"""Plot the full VQ run's observed losses; never synthesize missing updates."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, ScalarFormatter
import numpy as np

REMOTE_ROOT = '/mnt/cfs-omni/lhy/neurolm-reproduction/vq-five-full-20261107-r1'
REPO = Path(__file__).resolve().parents[1]
METRICS = ('reconstruction', 'codebook', 'adversarial')


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--fetch', action='store_true', help='Read the current metrics over authorized SSH')
    parser.add_argument('--input', type=Path, default=REPO / 'results/vq_five_full_metrics_20261107.jsonl')
    parser.add_argument('--output-dir', type=Path, default=REPO / 'results/figures/vq-five-full-20261107-r1')
    args = parser.parse_args()
    if args.fetch:
        remote = REMOTE_ROOT + '/run/metrics.jsonl'
        command = 'if test -s ' + shlex.quote(remote) + '; then cat ' + shlex.quote(remote) + '; fi'
        result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
                                 'tencent', command], capture_output=True, timeout=90)
        if result.returncode:
            raise RuntimeError('Could not read the full VQ metrics; SSH exit=' + str(result.returncode))
        if not result.stdout.strip():
            print(json.dumps({'plotted': False, 'reason': 'No full-training metrics yet'}))
            return
        args.input.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.input.with_suffix('.tmp')
        temporary.write_bytes(result.stdout)
        temporary.replace(args.input)
    if not args.input.exists():
        print(json.dumps({'plotted': False, 'reason': 'Metrics file does not exist'}))
        return

    content = args.input.read_bytes()
    # The producer appends one newline-terminated JSON row per optimizer update.
    # Ignore an in-flight final row, retaining every completed row unchanged.
    lines = content.splitlines(keepends=True)
    rows = [json.loads(line) for line in lines if line.endswith(b'\n') and line.strip()]
    if not rows:
        print(json.dumps({'plotted': False, 'reason': 'No complete optimizer updates'}))
        return
    steps = np.array([r['iter'] + 1 for r in rows], dtype=int)
    assert (np.diff(steps) > 0).all(), 'Duplicate or out-of-order updates'
    assert all(np.isfinite([r[k] for k in METRICS]).all() for r in rows), 'Non-finite loss in input'
    window = min(100, len(rows) // 10) if len(rows) >= 30 else 1
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(3, 1, figsize=(9, 7.4), sharex=True)
    titles = ('Reconstruction loss', 'Codebook loss', 'Adversarial loss')
    colors = ('#0072B2', '#7B3294', '#D55E00')
    for ax, key, title, color in zip(axes, METRICS, titles, colors):
        values = np.array([r[key] for r in rows])
        if window > 1:
            ax.plot(steps, values, color=color, alpha=0.28, lw=0.8, label='Observed steps')
            average = np.convolve(values, np.ones(window) / window, mode='valid')
            ax.plot(steps[window - 1:], average, color=color, lw=1.6,
                    label=f'Trailing {window}-step mean')
            ax.legend(loc='best', frameon=False, fontsize=8)
        else:
            ax.plot(steps, values, color=color, lw=1.5, marker='o', ms=3)
        ax.set_title(title, loc='left', fontsize=11, fontweight='bold')
        ax.set_ylabel('Loss')
        ax.grid(axis='y', color='#D9DEE3', lw=0.65, alpha=0.7)
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_powerlimits((-2, 4))
        ax.yaxis.set_major_formatter(formatter)
        ax.margins(x=0.02, y=0.15)
    axes[-1].set_xlabel('Optimizer update (1-based)')
    axes[-1].xaxis.set_major_locator(MaxNLocator(integer=True, nbins=9))
    last = rows[-1]
    fig.suptitle('NeuroLM | Five-dataset VQ pretraining', x=0.11, ha='left',
                 y=0.985, fontsize=15, fontweight='bold')
    fig.text(0.11, 0.944,
             f'{len(rows):,} observed updates | epoch {last["epoch"] + 1}, '
             f'update {last["step"] + 1}/897 | 459,656 EEG windows | 16 H20', fontsize=9.5)
    fig.text(0.11, 0.022,
             'Each observed point is a mean over 16 DDP ranks. Only recorded updates are shown.\n'
             'This is an in-progress training trace; early updates alone do not establish convergence.',
             fontsize=8, color='#46515C')
    fig.tight_layout(rect=(0, 0.065, 1, 0.925), h_pad=1.5)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for extension in ('png', 'svg', 'pdf'):
        path = args.output_dir / ('vq_training_curves.' + extension)
        fig.savefig(path, dpi=180, facecolor='white')
        if extension == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines()) + '\n', encoding='utf-8')
        outputs.append(str(path.resolve()))
    plt.close(fig)
    metadata = dict(plotted=True, source=str(args.input.resolve()),
                    source_sha256=hashlib.sha256(content).hexdigest(),
                    observed_updates=len(rows), first_update=int(steps[0]), last_update=int(steps[-1]),
                    epoch=last['epoch'] + 1, update_in_epoch=last['step'] + 1,
                    rolling_window=window, first={k: rows[0][k] for k in METRICS},
                    last={k: last[k] for k in METRICS},
                    generated_at=datetime.now(timezone.utc).isoformat(), outputs=outputs)
    (args.output_dir / 'curve_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(json.dumps(metadata))


if __name__ == '__main__':
    main()
