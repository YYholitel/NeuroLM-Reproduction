"""Plot audited five-task test results in percent and percentage-point deltas."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path('/mnt/cfs-omni/lhy/neurolm-reproduction/stage3-new-pretrain-20261109-r1')
data = json.loads((ROOT / 'comparison.json').read_text())
assert data['status'] == 'completed'
by_task = {row['task']: row for row in data['results'] if row['split'] == 'test'}
tasks = ['TUAB', 'TUEV', 'HMC', 'Workload', 'TUSL']
fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout='constrained')
x = np.arange(len(tasks))
for ax, key, title in zip(axes, ['balanced_accuracy', 'accuracy'], ['Balanced accuracy', 'Accuracy']):
    old = [100 * by_task[t]['metrics'][key]['official'] for t in tasks]
    new = [100 * by_task[t]['metrics'][key]['new'] for t in tasks]
    a = ax.bar(x - .19, old, .36, label='Official initialization', color='#75879a')
    b = ax.bar(x + .19, new, .36, label='Our VQ + stage 2', color='#067e85')
    ax.bar_label(a, fmt='%.1f', fontsize=8, padding=2)
    ax.bar_label(b, fmt='%.1f', fontsize=8, padding=2)
    for i, (before, after) in enumerate(zip(old, new)):
        ax.text(i, max(before, after) + 7, f'{after - before:+.1f} pp', ha='center', fontsize=9)
    ax.set(title=title, ylabel='Test score (%)', ylim=(0, 110), xticks=x,
           xticklabels=[f'{t}\nn={by_task[t]["samples"]:,}' for t in tasks])
    ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
axes[0].legend(loc='upper left', fontsize=8)
fig.suptitle('NeuroLM stage 3: matched 1-epoch instruction tuning, seed 42\nSame constrained decoding and complete held-out splits', fontsize=13)
fig.savefig(ROOT / 'monitor/comparison.png', dpi=180)
plt.close(fig)

rows = [json.loads(line) for line in (ROOT / 'train/train_steps.jsonl').read_text().splitlines()]
steps = np.array([row['step'] for row in rows])
fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout='constrained')
for ax, key, label, late in zip(axes.flat,
        ['instruction_loss', 'instruction_loss', 'text_loss', 'lr'],
        ['Instruction CE: full run', 'Instruction CE: after warmup', 'Text CE', 'Learning rate'],
        [False, True, False, False]):
    values = np.array([row[key] for row in rows])
    keep = steps >= 718 if late else np.ones(len(steps), dtype=bool)
    ax.plot(steps[keep], values[keep], color='#75a9ac', alpha=.65, linewidth=.8, label='Logged rank-0 microbatch')
    if key != 'lr':
        smooth = np.convolve(values, np.ones(20) / 20, mode='valid')
        valid = steps[19:] >= 718 if late else np.ones(len(smooth), dtype=bool)
        ax.plot(steps[19:][valid], smooth[valid], color='#005a60', linewidth=1.7, label='Trailing 20-log-point mean')
    if label == 'Instruction CE: full run':
        ax.set_yscale('log'); ax.legend(fontsize=8)
    ax.set(title=label, xlabel='Microstep')
    ax.grid(alpha=.2); ax.spines[['top', 'right']].set_visible(False)
fig.suptitle('NeuroLM stage 3 | 1 epoch, 2 H20, seed 42\nLosses are logged rank-0 microbatches, not full-dataset evaluation means', fontsize=13)
fig.savefig(ROOT / 'monitor/training_curves.png', dpi=180)
plt.close(fig)
stats = dict(logged_points=len(rows), last_logged_microstep=int(steps[-1]),
             loss_scope='rank-0 microbatch, every 10 microsteps; not an epoch-wide metric',
             first_20_log_mean={k:float(np.mean([r[k] for r in rows[:20]])) for k in ('instruction_loss','text_loss')},
             last_20_log_mean={k:float(np.mean([r[k] for r in rows[-20:]])) for k in ('instruction_loss','text_loss')})
(ROOT / 'monitor/training_summary.json').write_text(json.dumps(stats, indent=2))
