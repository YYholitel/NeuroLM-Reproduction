"""Recompute completed instruction metrics from every saved prediction.

The author metric behavior is preserved for comparison. Separate strict
accuracy/recall treats unparseable answers as errors, never as a valid class.
No EEG, generated text, filenames containing subject IDs, or private paths
are included in the aggregate report.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

TASKS = {
    'TUAB': (2, 73627, 36945), 'TUEV': (6, 15662, 28305),
    'HMC': (5, 22124, 23871), 'Workload': (2, 290, 290),
    'TUSL': (3, 66, 60),
}
TRAIN_COUNTS = {'TUAB': 298511, 'TUEV': 68270, 'HMC': 91248, 'Workload': 1508, 'TUSL': 119}


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_predictions(files, count, classes, world_size):
    assert len(files) == world_size, 'Missing rank prediction file'
    rows = {}
    hashes = {}
    for file in files:
        hashes[file.name] = sha256(file)
        with file.open(encoding='utf-8') as stream:
            for line in stream:
                item = json.loads(line)
                index, label, pred = (item[key] for key in ('index', 'label', 'prediction'))
                assert all(type(value) is int for value in (index, label, pred)), 'Non-integer prediction field'
                assert index not in rows, 'Duplicate prediction index'
                assert 0 <= index < count and 0 <= label < classes, 'Invalid sample index or label'
                assert -1 <= pred < classes, 'Prediction outside supported class range'
                rows[index] = (label, pred)
    assert set(rows) == set(range(count)), 'Missing prediction index'
    labels, preds = zip(*(rows[index] for index in range(count)))
    return list(labels), list(preds), hashes


def strict_statistics(labels, preds, classes):
    assert len(labels) == len(preds) and labels
    counts = Counter(labels)
    correct = Counter(label for label, pred in zip(labels, preds) if label == pred)
    confusion = [[0] * (classes + 1) for _ in range(classes)]
    for label, pred in zip(labels, preds):
        confusion[label][classes if pred == -1 else pred] += 1
    return {
        'accuracy': sum(correct.values()) / len(labels),
        'balanced_accuracy': sum(correct[label] / counts[label] for label in counts) / len(counts),
        'class_support': [counts[label] for label in range(classes)],
        'parse_failures': preds.count(-1),
        'confusion_matrix': confusion,
        'confusion_columns': list(range(classes)) + ['unparseable'],
    }


def audit(run, author_dir):
    assert (run / 'exit_code.txt').read_text().strip() == '0', 'Worker did not exit successfully'
    summary = json.loads((run / 'summary.json').read_text())
    manifest = json.loads((run / 'run_manifest.json').read_text())
    completed = json.loads((run / 'completed.json').read_text())
    preflight = json.loads((run / 'input_preflight.json').read_text())
    assert preflight['passed'] and not preflight['errors'], 'Inputs failed preflight'
    assert summary['status'] == 'completed' and not summary['paper_comparable']
    assert completed['seed'] == 42 and completed['epochs'] == 1
    assert completed['iter_num'] == manifest['microsteps_per_epoch'] == 7182
    assert manifest['world_size'] == 2 and set(completed['datasets']) == set(TASKS)
    for name, (_, val_count, test_count) in TASKS.items():
        assert manifest['dataset_counts'][name] == {'train': TRAIN_COUNTS[name], 'val': val_count, 'test': test_count}
    checkpoint_hash = sha256(run / 'checkpoints/instruction-B/ckpt.pt')
    assert checkpoint_hash == summary['checkpoint_sha256'], 'Final checkpoint changed'
    import torch
    checkpoint = torch.load(run / 'checkpoints/instruction-B/ckpt.pt', map_location='cpu', weights_only=False)
    optimizer_steps = sorted(set(int(value['step']) for value in checkpoint['optimizer']['state'].values()))
    assert checkpoint['epoch'] == 0 and checkpoint['iter_num'] == 7182 and optimizer_steps == [898]
    training = {'checkpoint_epoch': checkpoint['epoch'], 'completed_microsteps': checkpoint['iter_num'],
                'optimizer_steps': optimizer_steps, 'optimizer_states': len(checkpoint['optimizer']['state'])}
    del checkpoint
    if manifest['args']['eval_only']:
        provenance = json.loads((run / 'training_provenance.json').read_text())
        assert provenance == summary['training_provenance']
        assert provenance['checkpoint_sha256'] == checkpoint_hash
        assert provenance['iter_num'] == 7182 and provenance['optimizer_steps'] == [898]
        training['separate_evaluation_recovery'] = True
        training['source_training_run'] = provenance['source_training_run']
        training['source_worker_exit_code'] = provenance['training_worker_exit_code']
        if provenance.get('inference_lm_head_min_batch'):
            padding = json.loads((run / 'lm_head_padding_probe.json').read_text())
            assert padding['status'] == 'passed' and padding['padding_batch'] == 48
            assert len(padding['cases']) == 16
            training['inference_lm_head_min_batch'] = 48
            training['padding_probe_sha256'] = sha256(run / 'lm_head_padding_probe.json')
    rows = summary['metrics']
    assert len(rows) == len(TASKS) and {row['dataset'] for row in rows} == set(TASKS)
    sys.path.insert(0, str(author_dir.resolve()))
    import numpy as np
    from utils import get_metrics
    reports = []
    total_samples = 0
    for row in rows:
        name = row['dataset']
        classes, val_count, test_count = TASKS[name]
        binary = name in ('TUAB', 'Workload')
        for split, count in (('val', val_count), ('test', test_count)):
            expected = row[split]
            files = sorted((run / 'predictions').glob(f'{name}-{split}-epoch0-rank*.jsonl'))
            labels, preds, hashes = load_predictions(files, count, classes, manifest['world_size'])
            strict = strict_statistics(labels, preds, classes)
            assert expected['samples'] == count
            assert expected['parse_failures'] == strict['parse_failures']
            metric_names = [key for key in expected if key not in ('samples', 'parse_failures')]
            author_predictions = np.asarray(preds) if binary else np.eye(classes)[preds]
            actual = get_metrics(author_predictions, np.asarray(labels), metric_names, binary)
            for key in metric_names:
                assert math.isfinite(actual[key]) and math.isclose(actual[key], expected[key], rel_tol=1e-10, abs_tol=1e-10), (name, split, key)
            reports.append({'dataset': name, 'split': split, 'samples': count,
                            'author_metrics_recomputed': actual,
                            'strict_parse_failure_as_error': strict,
                            'prediction_file_sha256': hashes})
            total_samples += count
    assert total_samples == 201240
    return {'status': 'passed', 'paper_comparable': False, 'epochs': 1, 'seed': 42,
            'total_validation_test_samples': total_samples,
            'checkpoint_sha256_independently_recomputed': checkpoint_hash,
            'training_checkpoint_verified': training,
            'splits': reports,
            'metric_caveat': 'Author metrics retained: binary AUC uses generated labels as scores; binary classification maps parse failures to 0, multiclass to the last class. Strict accuracy and balanced accuracy always count parse failures as errors.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--author-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run_dir, args.author_dir)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'total_validation_test_samples': report['total_validation_test_samples']}))
