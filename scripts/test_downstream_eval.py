"""CPU tests: invalid-answer denominators, raw logging, trie and padding semantics."""
import io
import math
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

import torch
from downstream_eval_core import (TokenTrie, build_candidates, classification_metrics,
                                  entropy_statistics, generate_constrained, grow_attention_mask,
                                  parse_answer, record_prediction)


class EvaluationTests(unittest.TestCase):
    def test_parser_does_not_read_prompt_options(self):
        answers = {'(A)': 0, '(B)': 1, '(C)': 2}
        self.assertEqual(parse_answer('Options: (A) yes (B) no Answer: (B) <|endoftext|>(C)', answers)[0], 1)
        for text in ['Options: (A) (B) (C)', 'Answer: (E)', 'Answer: (B) (C)', 'Answer: (Bfoo)', 'Answer: (B<|endoftext|>A))']:
            self.assertEqual(parse_answer(text, answers)[0], -1)

    def test_tolerance_is_explicit_eos_bounded_and_task_bounded(self):
        answers = {'(A)': 0, '(B)': 1, '(C)': 2}
        self.assertEqual(parse_answer('Answer: (B<|endoftext|>A))', answers, True)[0], 1)
        for text in ['Answer: (B', 'Answer: (E<|endoftext|>', 'Answer: (BA<|endoftext|>']:
            self.assertEqual(parse_answer(text, answers, True)[0], -1)

    def test_tuab_exact_words(self):
        answers = {'Yes': 1, 'No': 0}
        self.assertEqual(parse_answer('Answer:\tYes  <|endoftext|>No', answers)[0], 1)
        for text in ['Answer: Yesterday', 'Answer: No Yes', 'Answer: (A)', 'Answer: yes']:
            self.assertEqual(parse_answer(text, answers)[0], -1)

    def test_invalid_printed_and_logged_verbatim(self):
        output, failures, stdout = io.StringIO(), io.StringIO(), io.StringIO()
        original = 'Question: x\nAnswer: (E) <|endoftext|>garbage'
        with redirect_stdout(stdout):
            record_prediction(output, failures, index=0, label=0, text=original, answers={'(A)': 0})
        import json
        self.assertEqual(json.loads(failures.getvalue())['generated_text'], original)
        self.assertIn('UNPARSEABLE_OUTPUT ', stdout.getvalue())
        self.assertEqual(json.loads(output.getvalue())['prediction'], -1)

    def test_invalid_not_negative_index_or_removed(self):
        m = classification_metrics([0, 1, 2, 2], [-1, 1, -1, 0], 3)
        self.assertEqual(m['samples'], 4)
        self.assertEqual(m['parse_failures'], 2)
        self.assertEqual(m['accuracy'], 0.25)
        self.assertAlmostEqual(m['balanced_accuracy'], 1 / 3)
        self.assertEqual(m['confusion_matrix'][2], [1, 0, 0, 1])
        self.assertEqual(classification_metrics([0, 1], [-1, -1], 2)['balanced_accuracy'], 0)

    def test_metrics_against_independent_sklearn(self):
        from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, recall_score
        import random
        rng = random.Random(3)
        for classes in [2, 3, 6]:
            labels = list(range(classes)) * 15
            preds = [rng.randrange(-1, classes) for _ in labels]
            m = classification_metrics(labels, preds, classes)
            self.assertAlmostEqual(m['accuracy'], accuracy_score(labels, preds))
            self.assertAlmostEqual(m['weighted_f1'], f1_score(labels, preds, labels=list(range(classes)), average='weighted', zero_division=0))
            self.assertAlmostEqual(m['balanced_accuracy'], recall_score(labels, preds, labels=list(range(classes)), average='macro', zero_division=0))
            self.assertAlmostEqual(m['cohen_kappa'], cohen_kappa_score(labels, preds))

    def test_mask_preserves_padding_and_causality(self):
        mask = torch.ones(1, 1, 4, 4, dtype=torch.bool).tril()
        mask[..., :, 1] = False
        result = grow_attention_mask(mask)
        self.assertFalse(result[..., :, 1].any())
        self.assertFalse(result[..., :-1, -1].any())
        self.assertTrue(result[0, 0, -1, -1])
        self.assertTrue(result[0, 0, -1, 2])

    def test_trie_multitoken_shared_prefix(self):
        trie = TokenTrie({0: (4, 5, 99), 1: (4, 6, 7, 99)})
        self.assertEqual(trie.allowed(()), [4])
        self.assertEqual(trie.allowed((4,)), [5, 6])
        with self.assertRaises(ValueError):
            TokenTrie({0: (4,), 1: (4, 5)})

    def test_constrained_rejects_highest_illegal_token_and_early_eos(self):
        class GPT:
            config = SimpleNamespace(block_size=40)
            calls = 0
            def __call__(self, **kwargs):
                self.calls += 1
                logits = torch.zeros(2, 1, 120)
                logits[:, :, 98] = 1000  # illegal option
                logits[:, :, 99] = 900   # EOS must wait for complete answer
                logits[0, :, 5] = 20
                logits[1, :, 6] = 20
                return logits, None, None
        model = SimpleNamespace(GPT2=GPT(), tokenizer=lambda x, *a, **k: x,
                                encode_transform_layer=lambda x: x,
                                pos_embed=lambda chans: torch.zeros(2, 1, 2))
        x = torch.ones(2, 1, 2)
        text = torch.tensor([[10], [10]])
        chans = torch.zeros(2, 1, dtype=torch.long)
        mask = torch.ones(2, 1, 2, 2, dtype=torch.bool).tril()
        output, probs = generate_constrained(model, x, text, chans, chans,
                                             torch.ones(2, 1, dtype=torch.bool), mask,
                                             {0: (4, 5, 99), 1: (4, 6, 7, 99)}, eos_id=99)
        self.assertEqual(output[0].tolist(), [10, 4, 5, 99, 99])
        self.assertEqual(output[1].tolist(), [10, 4, 6, 7, 99])
        self.assertEqual(probs, [None, None])
        self.assertEqual(model.GPT2.calls, 1)  # shared prefix and singleton suffixes are forced

    def test_candidates_preserve_prompt_token_boundary(self):
        loader = SimpleNamespace(prompt=torch.tensor([50257, 10, 11]),
                                 text={0: torch.tensor([50257, 10, 11, 12, 50256])})
        candidates, answers = build_candidates(loader, lambda ids: 'Question: x Answer: (A) <|endoftext|>')
        self.assertEqual(candidates[0], (12, 50256))
        self.assertEqual(answers, {'(A)': 0})
        loader.prompt[-1] = 20
        with self.assertRaises(ValueError):
            build_candidates(loader, str)

    def test_perplexity_excludes_padding(self):
        ids = torch.tensor([[0, 1, 7, 7]])
        mask = torch.tensor([[True, True, False, False]])
        stats = entropy_statistics(ids, mask, 8)
        self.assertAlmostEqual(stats['perplexity'], 2)
        self.assertEqual(stats['active_codes'], 2)
        self.assertEqual(stats['histogram'][7], 0)
        self.assertAlmostEqual(entropy_statistics(torch.zeros_like(ids), mask, 8)['perplexity'], 1)
        with self.assertRaises(ValueError):
            entropy_statistics(ids, torch.zeros_like(mask), 8)


if __name__ == '__main__':
    unittest.main()
