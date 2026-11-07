"""Read-only TI-ONE monitoring and plot refresh for a detached tmux session."""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def complete_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_bytes().splitlines(keepends=True)
            if line.endswith(b'\n') and line.strip()]


def gpu_summary(path):
    if not path.exists():
        return {'available': False}
    with path.open('rb') as stream:
        header = stream.readline()
        start = max(len(header), path.stat().st_size - 128 * 1024)
        stream.seek(start)
        if start > len(header):
            stream.readline()
        rows = list(csv.DictReader(io.StringIO((header + stream.read()).decode()), skipinitialspace=True))
    devices = {}
    for row in rows:
        devices.setdefault(int(row['gpu_index']), []).append(row)
    result = []
    for index, history in sorted(devices.items()):
        recent = history[-12:]
        result.append({'index': index, 'timestamp': recent[-1]['timestamp'],
                       'mean_util_pct': sum(float(r['gpu_util_pct']) for r in recent) / len(recent),
                       'memory_MiB': float(recent[-1]['memory_used_MiB'])})
    return {'available': True, 'devices': result}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--interval', type=int, default=60)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    monitor = root / 'monitor'
    monitor.mkdir(exist_ok=True)
    lock = (monitor / 'watch.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    render = monitor / '.render'
    render.mkdir(exist_ok=True)
    figures = monitor / 'figures'
    figures.mkdir(exist_ok=True)
    metrics = root / 'run/metrics.jsonl'
    last_plot_update = None
    api = None
    api_checked = 0
    while True:
        started = time.monotonic()
        try:
            rows = complete_rows(metrics)
            if not rows:
                raise RuntimeError('No completed optimizer updates')
            last = rows[-1]
            if time.monotonic() - api_checked > 300:
                check = subprocess.run([sys.executable, str(root / 'notebook_action.py'), 'status'],
                                       capture_output=True, text=True, timeout=115)
                if check.returncode == 0:
                    api = json.loads(check.stdout)
                else:
                    api = {'ok': False, 'exit_code': check.returncode}
                api_checked = time.monotonic()
            plot_error = None
            if last['iter'] != last_plot_update:
                snapshot = render / 'metrics.jsonl'
                snapshot.write_text(''.join(json.dumps(r) + '\n' for r in rows))
                env = dict(os.environ, MPLCONFIGDIR=str(monitor / '.mplconfig'),
                           OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
                plot = subprocess.run([sys.executable, str(monitor / 'plot_vq_training_curves.py'),
                                       '--input', str(snapshot), '--output-dir', str(render)],
                                      capture_output=True, text=True, env=env, timeout=120)
                if plot.returncode:
                    plot_error = {'exit_code': plot.returncode}
                    (monitor / 'plot_error.log').write_text(plot.stderr)
                else:
                    meta = json.loads((render / 'curve_metadata.json').read_text())
                    names = ['vq_training_curves.png', 'vq_training_curves.svg', 'vq_training_curves.pdf']
                    for name in names:
                        os.replace(render / name, figures / name)
                    os.replace(snapshot, monitor / 'metrics_snapshot.jsonl')
                    meta['source'] = str(monitor / 'metrics_snapshot.jsonl')
                    meta['outputs'] = [str(figures / name) for name in names]
                    (render / 'curve_metadata.json').write_text(json.dumps(meta, indent=2))
                    os.replace(render / 'curve_metadata.json', figures / 'curve_metadata.json')
                    last_plot_update = last['iter']
            epochs = complete_rows(root / 'run/epoch_metrics.jsonl')
            summary = None
            if epochs:
                latest = epochs[-1]
                summary = {k: latest[k] for k in ['epoch', 'mean', 'lr_last', 'tail_code_usage',
                                                 'fixed_diagnostic', 'review_flags']}
            report = dict(observed_at=datetime.now(timezone.utc).isoformat(),
                          metrics_age_seconds=time.time() - metrics.stat().st_mtime,
                          updates=len(rows), epoch=last['epoch'] + 1, update_in_epoch=last['step'] + 1,
                          losses={k: last[k] for k in ['reconstruction', 'codebook', 'adversarial']},
                          last_completed_epoch=summary, api=api, plot_error=plot_error,
                          last_plotted_update=None if last_plot_update is None else last_plot_update + 1,
                          gpu={node: gpu_summary(root / 'run' / node / 'gpu_metrics.csv')
                               for node in ['node-0', 'node-1']})
            report['stale_metrics'] = report['metrics_age_seconds'] > 300
            temporary = monitor / 'latest.tmp'
            temporary.write_text(json.dumps(report, indent=2))
            os.replace(temporary, monitor / 'latest.json')
            with (monitor / 'watch.log').open('a') as stream:
                stream.write(json.dumps(report) + '\n')
            print(f'\n{report["observed_at"]} | epoch {report["epoch"]}, '
                  f'update {report["update_in_epoch"]}/897 | total {len(rows)}', flush=True)
            print('Loss: ' + ' | '.join(f'{key}={value:.6g}' for key, value in report['losses'].items()), flush=True)
            for node, gpu in report['gpu'].items():
                print(node + ' recent GPU utilization: ' + ', '.join(
                    f'{g["index"]}:{g["mean_util_pct"]:.0f}%' for g in gpu.get('devices', [])), flush=True)
            if summary:
                usage = summary['tail_code_usage']
                print(f'Completed epoch {summary["epoch"]}: recent active codes '
                      f'{usage["active_codes"]}/{usage["codebook_size"]}; '
                      f'perplexity={usage["perplexity"]:.2f}; flags={summary["review_flags"]}', flush=True)
            print(f'Plot update={report["last_plotted_update"]}; stale={report["stale_metrics"]}; '
                  f'plot_error={plot_error}; API tasks={(api or {}).get("tasks", [])}', flush=True)
            print('PNG: ' + str(figures / 'vq_training_curves.png'), flush=True)
            task_id = json.loads((root / 'submitted.json').read_text())['id']
            terminal = next((t['status'] for t in (api or {}).get('tasks', [])
                             if t['id'] == task_id and t['status'] in
                             ['SUCCEED', 'FAILED', 'STOPPED', 'TERMINATED', 'DELETED']), None)
            if terminal or (root / 'run/completed.json').exists() or args.once:
                return
        except Exception as exc:
            print(json.dumps({'monitor_error': type(exc).__name__,
                              'observed_at': datetime.now(timezone.utc).isoformat()}), flush=True)
            if args.once:
                raise
        time.sleep(max(1, args.interval - (time.monotonic() - started)))


if __name__ == '__main__':
    main()
