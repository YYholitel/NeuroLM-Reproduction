"""Synthetic integrity checks; no model training or measured results are created."""
import json
from pathlib import Path
import tempfile
import unittest

from audit_instruction_results import load_predictions, strict_statistics


class AuditChecks(unittest.TestCase):
    def test_unparseable_answers_are_always_errors(self):
        result = strict_statistics([0, 1, 2, 2], [-1, 1, -1, 2], 3)
        self.assertEqual(result['accuracy'], .5)
        self.assertEqual(result['balanced_accuracy'], .5)
        self.assertEqual(result['parse_failures'], 2)
        self.assertEqual(result['confusion_matrix'], [[0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 1]])

    def test_binary_failure_is_not_a_correct_negative(self):
        result = strict_statistics([0, 1], [-1, 1], 2)
        self.assertEqual(result['accuracy'], .5)
        self.assertEqual(result['balanced_accuracy'], .5)

    def test_full_coverage_and_corruption_rejection(self):
        with tempfile.TemporaryDirectory(prefix='instruction-audit-test-') as folder:
            files = [Path(folder) / f'rank{rank}.jsonl' for rank in range(2)]
            original = [[{'index': 0, 'label': 0, 'prediction': -1}, {'index': 2, 'label': 1, 'prediction': 1}],
                        [{'index': 1, 'label': 0, 'prediction': 0}, {'index': 3, 'label': 1, 'prediction': 0}]]
            def write(rows):
                for file, entries in zip(files, rows):
                    file.write_text(''.join(json.dumps(row) + '\n' for row in entries))
            write(original)
            labels, preds, hashes = load_predictions(files, 4, 2, 2)
            self.assertEqual(labels, [0, 0, 1, 1])
            self.assertEqual(preds, [-1, 0, 1, 0])
            self.assertEqual(len(hashes), 2)
            cases = [
                ([original[0], original[1][:-1]], 'Missing prediction index'),
                ([original[0], [original[0][0], original[1][1]]], 'Duplicate prediction index'),
                ([original[0], [dict(original[1][0], label=2), original[1][1]]], 'Invalid sample index or label'),
                ([original[0], [dict(original[1][0], prediction=-2), original[1][1]]], 'Prediction outside supported class range'),
            ]
            for rows, message in cases:
                with self.subTest(message=message):
                    write(rows)
                    with self.assertRaisesRegex(AssertionError, message):
                        load_predictions(files, 4, 2, 2)


if __name__ == '__main__':
    unittest.main()
