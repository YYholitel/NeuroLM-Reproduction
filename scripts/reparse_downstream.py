"""Reparse saved predictions; log every invalid original string, preserve coverage."""
import argparse
import json
from pathlib import Path

from checkpoint_eval_io import sha256, write_json
from downstream_eval_core import classification_metrics, record_prediction

ANSWERS = {'TUSL': {'(A)': 0, '(B)': 1, '(C)': 2}, 'TUAB': {'No': 0, 'Yes': 1}}


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('--input-files', type=Path, nargs='+', required=True)
    p.add_argument('--task', choices=list(ANSWERS), required=True)
    p.add_argument('--expected-count', type=int, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--allow-missing-close', action='store_true', help='Explicit tolerant-parser ablation, not strict parsing')
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=False)
    rows, hashes, indices = [], {}, set()
    with (args.out_dir / 'predictions.jsonl').open('x', encoding='utf-8') as out, (args.out_dir / 'invalid.jsonl').open('x', encoding='utf-8') as failures:
        for path in args.input_files:
            hashes[path.name] = sha256(path)
            with path.open(encoding='utf-8') as source:
                for line in source:
                    item = json.loads(line)
                    index = item['index']
                    if index in indices:
                        raise ValueError('Duplicate sample index')
                    indices.add(index)
                    rows.append(record_prediction(out, failures, index=index, label=item['label'],
                                                  text=item['generated_text'], answers=ANSWERS[args.task],
                                                  allow_missing_close=args.allow_missing_close))
    if indices != set(range(args.expected_count)):
        raise ValueError('Missing or out-of-range sample index')
    report = {'task': args.task, 'protocol': 'tolerant_missing_close_ablation' if args.allow_missing_close else 'strict',
              'metrics': classification_metrics([r['label'] for r in rows], [r['prediction'] for r in rows], len(ANSWERS[args.task])),
              'source_sha256': hashes, 'prediction_sha256': sha256(args.out_dir / 'predictions.jsonl'),
              'invalid_sha256': sha256(args.out_dir / 'invalid.jsonl')}
    write_json(args.out_dir / 'summary.json', report)
    print(json.dumps(report, allow_nan=False))


if __name__ == '__main__':
    main()
