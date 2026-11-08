"""Strict parsers, all-sample metrics and exact-token constrained decoding.

No author training code is modified. Invalid answers remain -1 and are always
errors; they are never used as a NumPy index or removed from denominators.
"""
import json
import math
import re

EOS_TEXT = '<|endoftext|>'


def parse_answer(text, answer_to_label, allow_missing_close=False):
    before_eos, eos, _ = text.partition(EOS_TEXT)
    if before_eos.count('Answer:') != 1:
        return -1, 'missing_or_ambiguous_answer_marker'
    answer = before_eos.split('Answer:', 1)[1].strip()
    if answer in answer_to_label:
        return answer_to_label[answer], 'exact'
    if allow_missing_close and eos and re.fullmatch(r'\([A-Z]', answer):
        if answer + ')' in answer_to_label:
            return answer_to_label[answer + ')'], 'tolerant_missing_close'
    return -1, 'invalid_answer'


def record_prediction(stream, failure_stream, *, index, label, text, answers,
                      context=None, allow_missing_close=False, probabilities=None):
    prediction, reason = parse_answer(text, answers, allow_missing_close)
    row = dict(context or {}, index=int(index), label=int(label),
               prediction=prediction, generated_text=text, parse_reason=reason)
    if probabilities is not None:
        row['conditional_first_token_probabilities'] = probabilities
    encoded = json.dumps(row, ensure_ascii=False, allow_nan=False)
    stream.write(encoded + '\n')
    if prediction == -1:
        # Full text is mandatory, both on stdout and in an explicit failure file.
        failure_stream.write(encoded + '\n')
        failure_stream.flush()
        print('UNPARSEABLE_OUTPUT ' + encoded, flush=True)
    return row


def classification_metrics(labels, predictions, classes):
    if len(labels) != len(predictions) or not labels or classes < 2:
        raise ValueError('Metrics need equal, nonempty sequences and >=2 classes')
    matrix = [[0] * (classes + 1) for _ in range(classes)]
    for label, pred in zip(labels, predictions):
        if not 0 <= label < classes or not -1 <= pred < classes:
            raise ValueError('Out-of-range label or prediction')
        matrix[label][classes if pred == -1 else pred] += 1
    support = [sum(row) for row in matrix]
    predicted = [sum(row[col] for row in matrix) for col in range(classes)]
    count = len(labels)
    correct = sum(matrix[i][i] for i in range(classes))
    recalls = [matrix[i][i] / support[i] if support[i] else None for i in range(classes)]
    f1 = [2 * matrix[i][i] / (support[i] + predicted[i]) if support[i] + predicted[i] else 0.0
          for i in range(classes)]
    chance = sum(a * b for a, b in zip(support, predicted)) / (count * count)
    return {
        'samples': count, 'parse_failures': sum(row[-1] for row in matrix),
        'accuracy': correct / count,
        'balanced_accuracy': sum(r for r in recalls if r is not None) / sum(n > 0 for n in support),
        'weighted_f1': sum(n * f for n, f in zip(support, f1)) / count,
        'cohen_kappa': (correct / count - chance) / (1 - chance) if chance < 1 else None,
        'class_support': support, 'class_recall': recalls,
        'missing_true_classes': [i for i, n in enumerate(support) if not n],
        'confusion_matrix': matrix,
        'confusion_columns': list(range(classes)) + ['invalid'],
        'invalid_policy': 'retain_in_denominator_and_count_wrong',
    }


def build_candidates(loader, decode, eos_id=50256):
    """Use complete training templates; never re-tokenize an answer in isolation."""
    prompt = [int(t) for t in loader.prompt]
    candidates, answers = {}, {}
    for label, tensor in loader.text.items():
        full = [int(t) for t in tensor]
        if full[:len(prompt)] != prompt:
            raise ValueError('Training template is not an exact token-prefix extension of prompt')
        suffix = tuple(full[len(prompt):])
        if not suffix or suffix[-1] != eos_id or eos_id in suffix[:-1]:
            raise ValueError('Candidate must end at its first EOS')
        if suffix in candidates.values():
            raise ValueError('Duplicate token continuation')
        candidates[int(label)] = suffix
        full_text = decode(full[1:])  # [SEP] is outside the GPT-2 text vocabulary.
        if full_text.count('Answer:') != 1:
            raise ValueError('Ambiguous template')
        answer = full_text.partition('Answer:')[2].partition(EOS_TEXT)[0].strip()
        if answer in answers:
            raise ValueError('Duplicate decoded answer')
        answers[answer] = int(label)
    if sorted(candidates) != list(range(len(candidates))):
        raise ValueError('Labels must be contiguous from zero')
    return candidates, answers


class TokenTrie:
    def __init__(self, candidates):
        self.candidates = dict(candidates)
        self.children, self.terminals = {}, {}
        for label, tokens in candidates.items():
            tokens = tuple(tokens)
            if not tokens or tokens in self.terminals:
                raise ValueError('Empty or duplicate candidate')
            self.terminals[tokens] = label
            for length, token in enumerate(tokens):
                self.children.setdefault(tokens[:length], set()).add(token)
        if set(self.children).intersection(self.terminals):
            raise ValueError('One complete candidate is a prefix of another')

    def allowed(self, prefix):
        return sorted(self.children.get(tuple(prefix), ()))


def grow_attention_mask(mask):
    import torch
    if mask is None:
        return None
    # Existing queries cannot see the future. Inherit the previous final query's
    # visibility so a new query does NOT accidentally unmask padded EEG keys.
    last = mask[..., -1:, :]
    mask = torch.cat((mask, torch.zeros((*mask.shape[:-1], 1), dtype=mask.dtype, device=mask.device)), -1)
    new_row = torch.cat((last, torch.ones((*last.shape[:-1], 1), dtype=mask.dtype, device=mask.device)), -1)
    return torch.cat((mask, new_row), -2)


def generate_constrained(model, x_eeg, x_text, chans, times, valid_mask, attention_mask,
                         candidates, eos_id=50256):
    """Greedy token-trie decoding, exactly matching legal template continuations.

    Returns text tensors and conditional probabilities at the first branch only
    when every class has a distinct first token. These are explicitly *not*
    unconstrained full-sequence likelihoods.
    """
    import torch
    trie = TokenTrie(candidates)
    if any(tokens[-1] != eos_id or eos_id in tokens[:-1] for tokens in candidates.values()):
        raise ValueError('Every candidate must end at its first EOS')
    max_new = max(map(len, candidates.values()))
    if x_eeg.size(1) + x_text.size(1) + max_new > model.GPT2.config.block_size:
        raise ValueError('Prompt plus complete answer exceeds model block size')
    batch = x_text.size(0)
    prefixes = [() for _ in range(batch)]
    done = [False] * batch
    first_probs = [None] * batch
    labels = sorted(candidates)
    first = [candidates[label][0] for label in labels]
    distinct_first = len(set(first)) == len(first)
    with torch.inference_mode():
        encoder_mask = valid_mask[:, None, None, :].expand(-1, 1, x_eeg.size(1), -1)
        features = model.tokenizer(x_eeg, chans, times, encoder_mask, return_all_tokens=True)
        features = model.encode_transform_layer(features) + model.pos_embed(chans)
        for step in range(max_new):
            allowed_rows = [trie.allowed(prefixes[i]) if not done[i] else [eos_id] for i in range(batch)]
            if any(not allowed for allowed in allowed_rows):
                raise RuntimeError('No legal continuation before terminal state')
            # Singleton branches are forced regardless of logits. Skipping these
            # forwards is exactly equivalent to applying a singleton token mask.
            needs_logits = any(len(allowed) > 1 for allowed in allowed_rows)
            if needs_logits:
                logits, _, _ = model.GPT2(x_eeg=features, x_text=x_text, eeg_time_idx=times,
                                         eeg_mask=None, eeg_text_mask=attention_mask)
                logits = logits[:, -1, :50257].float()
            next_tokens = []
            for i in range(batch):
                if done[i]:
                    next_tokens.append(eos_id)  # inert padding; decoding truncates at first EOS
                    continue
                allowed = allowed_rows[i]
                token = allowed[0]
                if len(allowed) > 1:
                    scores = logits[i, allowed]
                    if not torch.isfinite(scores).all():
                        raise RuntimeError('Nonfinite allowed logits')
                    token = allowed[int(scores.argmax())]  # deterministic ties: smallest token ID
                if step == 0 and distinct_first:
                    first_probs[i] = torch.softmax(logits[i, first], -1).cpu().tolist()
                prefixes[i] += (token,)
                done[i] = prefixes[i] in trie.terminals
                next_tokens.append(token)
            x_text = torch.cat((x_text, torch.tensor(next_tokens, device=x_text.device)[:, None]), 1)
            if all(done):
                break
            attention_mask = grow_attention_mask(attention_mask)
    if not all(done):
        raise RuntimeError('Constrained decoding did not reach a legal EOS')
    return x_text, first_probs


def entropy_statistics(indices, valid_mask, size):
    import torch
    if indices.shape != valid_mask.shape or valid_mask.dtype != torch.bool:
        raise ValueError('Indices and boolean validity mask must have matching shapes')
    selected = indices[valid_mask].long()
    if not selected.numel() or selected.min() < 0 or selected.max() >= size:
        raise ValueError('No valid tokens, or codebook index out of range')
    counts = torch.bincount(selected.cpu(), minlength=size).double()
    probabilities = counts[counts > 0] / counts.sum()
    entropy = -(probabilities * probabilities.log()).sum().item()
    return dict(valid_tokens=selected.numel(), padding_tokens=(~valid_mask).sum().item(),
                active_codes=int((counts > 0).sum()), codebook_size=size,
                theoretical_perplexity_upper_bound=min(size, selected.numel()),
                usage_fraction=(counts > 0).sum().item() / size,
                entropy_nats=entropy, perplexity=math.exp(entropy),
                top1_fraction=counts.max().item() / counts.sum().item(),
                histogram=counts.long().tolist())
