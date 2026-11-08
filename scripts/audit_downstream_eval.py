"""Independently verify full corrected evaluation outputs and sklearn metrics."""
import argparse
import json
import math
from pathlib import Path

from checkpoint_eval_io import sha256, write_json


def audit(run):
    from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, recall_score, roc_auc_score
    summary = json.loads((run / 'summary.json').read_text(encoding='utf-8'))
    assert summary['status'] == 'completed' and not summary['training_performed']
    reports = []
    seen_splits = set()
    expected_counts = {'TUSL': {'val': 66, 'test': 60}, 'TUAB': {'val': 73627, 'test': 36945}}
    for split in summary['results']:
        task, name = split['task'], split['split']
        assert (task, name) not in seen_splits, 'Duplicate split report'
        seen_splits.add((task, name))
        assert split['available_samples'] == expected_counts[task][name]
        rows, failure_rows, hashes = {}, {}, {}
        answers = split['answers']
        classes = len(answers)
        assert split['full_split'], 'Refusing to audit a smoke subset as a full split'
        for rank in range(summary['world_size']):
            path = run / f'{task}-{name}-rank{rank}.jsonl'
            hashes[path.name] = sha256(path)
            assert hashes[path.name] == split['prediction_sha256'][path.name]
            with path.open(encoding='utf-8') as stream:
                for line in stream:
                    row = json.loads(line)
                    index, label, prediction = row['index'], row['label'], row['prediction']
                    assert all(type(v) is int for v in [index, label, prediction])
                    assert index not in rows and 0 <= label < classes and -1 <= prediction < classes
                    # Independent strict parser, not the production parse function.
                    before_eos = row['generated_text'].split('<|endoftext|>', 1)[0]
                    parts = before_eos.split('Answer:')
                    expected = answers.get(parts[1].strip(), -1) if len(parts) == 2 else -1
                    assert prediction == expected, 'Stored prediction disagrees with raw output'
                    if summary['mode'] == 'constrained':
                        assert prediction != -1 and '<|endoftext|>' in row['generated_text']
                    rows[index] = row
            path = run / f'{task}-{name}-rank{rank}-invalid.jsonl'
            hashes[path.name] = sha256(path)
            with path.open(encoding='utf-8') as stream:
                for line in stream:
                    row = json.loads(line)
                    assert row['index'] not in failure_rows and row['prediction'] == -1
                    failure_rows[row['index']] = row
        assert set(rows) == set(range(split['available_samples']))
        assert failure_rows == {i: row for i, row in rows.items() if row['prediction'] == -1}
        rows = [rows[i] for i in range(len(rows))]
        labels = [r['label'] for r in rows]
        predictions = [r['prediction'] for r in rows]
        observed = sorted(set(labels))
        actual = {
            'accuracy': accuracy_score(labels, predictions),
            'balanced_accuracy': recall_score(labels, predictions, labels=observed, average='macro', zero_division=0),
            'weighted_f1': f1_score(labels, predictions, labels=list(range(classes)), average='weighted', zero_division=0),
            'cohen_kappa': cohen_kappa_score(labels, predictions),
        }
        for key, value in actual.items():
            expected = split['metrics'][key]
            assert math.isclose(value, expected, abs_tol=1e-12, rel_tol=1e-12), (task, name, key, value, expected)
        assert split['metrics']['parse_failures'] == predictions.count(-1)
        assert split['metrics']['samples'] == len(rows)
        if split['metrics']['roc_auc'] is not None:
            for row in rows:
                scores = row['conditional_first_token_probabilities']
                assert len(scores) == classes and all(math.isfinite(s) and 0 <= s <= 1 for s in scores)
                assert math.isclose(sum(scores), 1, abs_tol=1e-6)
            score = roc_auc_score(labels, [r['conditional_first_token_probabilities'][1] for r in rows])
            assert math.isclose(score, split['metrics']['roc_auc'], abs_tol=1e-12)
            actual['roc_auc'] = score
        reports.append({'task': task, 'split': name, 'samples': len(rows),
                        'parse_failures': len(failure_rows), 'independent_metrics': actual,
                        'artifact_sha256': hashes})
    return {'status': 'passed', 'summary_sha256': sha256(run / 'summary.json'),
            'total_samples': sum(item['samples'] for item in reports), 'splits': reports,
            'independent_strict_parser': True, 'invalid_log_matches_all_failures': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.run_dir)
    write_json(args.output, report)
    print(json.dumps({'status': report['status'], 'total_samples': report['total_samples']}))
