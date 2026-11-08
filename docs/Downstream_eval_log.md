# Downstream evaluation log

## Scope and provenance — 2026-11-08

Evaluation only. No new training and no changes to `train_vq.py`. TUAB and TUSL
use the original Appendix B prompts and answer token boundaries.

The completed five-task instruction checkpoint has SHA-256
`a10a23f6f4a69058c996c01096d47b0806dafd83a0d0ca6506e274a5c6372174`.
It was initialized from the official NeuroLM-B checkpoint, not the independently
trained VQ that later exhibited collapse. Its actual downstream encoder is the
`tokenizer.*` state embedded in the NeuroLM checkpoint. Downstream inference does
not invoke VQ quantization. The claim that the previous TUSL errors were caused
by the newly trained collapsed VQ is therefore not established.

An exact tensor comparison subsequently verified that all **90 encoder state
tensors** in this instruction checkpoint are identical to the official VQ encoder
(zero mismatched keys/tensors). The new collapsed/repaired VQ was not its encoder.

Historical published results are retained in
[the original report](../results/INSTRUCTION_FIVE_TASK_20261108.md).
The new inference entry point is `scripts/eval_downstream.py`; the frozen author
runtime remains evidence of the original experiment, not the entry point for
future corrected evaluation.

## Saved-generation parser audit

Rules were defined from generated text, without consulting the true class.
All samples remain in every classification denominator. Invalid output is `-1`,
never a one-hot array index. Accuracy, balanced accuracy, weighted F1 and kappa
use an extra invalid column in the confusion matrix. Binary ROC AUC is not
computed from generated hard labels or by dropping invalid examples.

| TUSL split | Rule | Invalid / total | Accuracy | Balanced accuracy |
|---|---|---:|---:|---:|
| Validation | Strict | 36 / 66 | 22.73% | 25.35% |
| Validation | Missing-`)` tolerance | 7 / 66 | 43.94% | 45.58% |
| Test | Strict | 42 / 60 | 10.00% | 6.44% |
| Test | Missing-`)` tolerance | 8 / 60 | 20.00% | 43.94% |

Strict parsing anchors on `Answer:` and truncates at the first EOS. This alone
does not recover the malformed TUSL answers. The separately labelled tolerance
accepts only an otherwise exact legal option prefix such as `(B` immediately
before EOS. It never maps `(E)` into the three legal classes. Of 34 recovered
test answers, 6 are correct and 28 incorrect. There are only 3 true class-B test
examples, so balanced accuracy is particularly sensitive to those predictions.

Every invalid original string must be printed and saved in an explicit private
`*-invalid.jsonl` / `invalid.jsonl` file. Aggregate reports can be public; raw EEG,
sample identifiers and raw prediction files remain private.

## Corrected constrained evaluation

`scripts/eval_downstream.py --mode constrained` derives a token trie directly
from each loader's `prompt` and complete `text[label]` tensors, including EOS.
It verifies exact prompt-prefix equality; it does not independently tokenize
answer fragments or whitelist isolated letter tokens. TUAB retains **Yes/No**;
TUSL retains **(A)/(B)/(C)**. Greedy selection is restricted at every position
until a full legal answer and EOS have been produced.

Singleton trie branches have only one possible next token and are appended
without an unnecessary model forward. TUAB/TUSL each have a different first
answer token per class, so only the actual class decision requires logits.
An adversarial multi-token/shared-prefix test verifies this optimization.

Appended attention rows inherit the existing EEG padding mask. The author's
generation recurrence instead fills the new query row with ones, exposing
previously masked EEG padding. This additional evaluation correction is explicit;
new results must not be described as a parser-only or paper-identical comparison.
For the present TUAB/TUSL candidates, the decision occurs before any appended
query, so the padding-row correction does not change that class decision.

TUAB AUC, when available, uses the conditional probability of Yes among the
legal first answer tokens. This is a modified score definition, not the original
author's AUC from generated class IDs. The evaluator records the definition.

Local CPU verification: **12 tests passed**. Coverage includes adversarial illegal
top logits, premature EOS, shared multi-token prefixes, variable answer lengths,
mask preservation, invalid raw-output logging, independent sklearn metric
agreement, rejection of NaN/Inf codebooks/features, and valid-token-only perplexity. Real-checkpoint runs are recorded below
only after completion and coverage verification.

## VQ quality tool for Agent A

Use `scripts/eval_vq_quality.py` with the model code matching the checkpoint. It
strictly loads official `VQ.pt` or training checkpoints with `VQ.*` keys and saved
`encoder_args`, `decoder_args`, and optional `vq_options`. Wrapper prefixes are
normalized, while missing/unexpected weights cause failure.

```bash
python scripts/eval_vq_quality.py \
  --author-dir "$VQ_CODE" --checkpoint "$VQ_CHECKPOINT" \
  --sample-manifest "$FIXED_SAMPLES_JSON" --save-batch "$FIXED_BATCH_PT" \
  --output "$NEW_REPORT_JSON" --device cpu

# Every later checkpoint uses identical tensors, ordering and normalization.
python scripts/eval_vq_quality.py \
  --author-dir "$MATCHING_VQ_CODE" --checkpoint "$NEXT_VQ_CHECKPOINT" \
  --batch "$FIXED_BATCH_PT" --output "$NEXT_REPORT_JSON" --device cpu
```

The sample manifest is a fixed JSON list of trusted pretraining pickle paths
(`X`, `ch_names`), or indexed-tar entries if that author's loader supports them.
The frozen batch contains `x`, `input_chans`, `input_time`, `valid_mask`.
It must stay private. The tool checks inputs, excludes padding, records all
tensor hashes, checkpoint/code hashes, histogram, active codes and
`perplexity = exp(-sum(p * ln(p)))`. It fails if parameters or codebook weights
change. Legacy usage-buffer changes are explicitly reported. It refuses an
uninitialized codebook rather than fitting k-means during evaluation.
Nonfinite codebook weights or encoder features are rejected before quantization;
they cannot silently become argmin index zero and a misleading perplexity of one.

Fixed-batch perplexity is a diagnostic, not a downstream accuracy metric or a
standalone guarantee that VQ training has succeeded. Its upper bound is the
smaller of the codebook size and the number of observed valid tokens; a tiny
batch cannot demonstrate use of all 8192 codes.

## New-checkpoint handoff

Agent A's chat is **定位预训练 ck**. The final five-epoch VQ was received and
evaluated during this task; the completed measurements are recorded below. No
matched newly pretrained/instruction-tuned NeuroLM has been provided. Only
completed immutable checkpoints are eligible for these recorded evaluations.

For a full NeuroLM checkpoint, run strict loading and full TUSL/TUAB validation
and test splits. For a VQ-only checkpoint, first run the fixed-batch quality test.
`--vq-encoder-override` permits the requested immediate downstream diagnostic,
but labels it **unaligned VQ encoder swap**: the old adapter/LLM has not been
trained with this new encoder. It does not replace a matched stage-two/three
model, and this agent must not start training to produce one.

```bash
python scripts/eval_downstream.py \
  --author-dir "$NEUROLM_CODE" --checkpoint "$NEUROLM_CHECKPOINT" \
  --dataset-dir "$DATASET_ROOT" --out-dir "$NEW_EVAL_DIRECTORY" \
  --tasks TUSL TUAB --mode constrained --device cpu
```

For allocated GPU workers, use `torchrun --standalone --nproc_per_node=2`,
`--device cuda --precision bfloat16`; retain the verified inference LM-head
small-batch padding patch in the frozen model code. Do not run on Notebook GPUs.
Output directories must be new. The runner verifies exact once-only split index
coverage, preserves all raw predictions, writes per-split metrics and hashes, and
creates `summary.json` only after every requested split finishes successfully.

## Real-checkpoint CPU checks — 2026-11-08

Official VQ SHA-256:
`e792c39a6a9e6d1bf4604cf63090730424f1d37f942597883d0c0a1375a2663a`.
The two-sample frozen batch (one TUAB and one TUSL training sample) has file hash
`4c9b3f152c3e5d39be8cf00c59d3e89d5c57d82d3c68094ac4898725563422e3`.
The official VQ used **248 codes / 460 valid tokens**, perplexity **158.155494**,
with **1588 padding tokens excluded**. Parameters/codebook weights did not change.
The original quantizer's runtime usage buffer changes in evaluation and is
reported explicitly; the checkpoint file is unchanged. This is a tiny fixed
training-batch diagnostic, not a held-out VQ quality conclusion.

The current instruction checkpoint, CPU float32, exact constrained answer trie:

| TUSL split | Coverage | Invalid | Accuracy | Balanced accuracy | Weighted F1 | Kappa |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 66 / 66 | 0 | 50.00% | 51.8053% | 49.3426% | 0.245322 |
| Test | 60 / 60 | 0 | 25.00% | 46.9697% | 25.7276% | -0.097561 |

These are real inference results from the existing instruction checkpoint, not
from Agent A's new VQ. A format guarantee does not imply good classification.

Compatibility was also checked with Agent A's **completed 500-step reset-on** VQ
checkpoint, using the identical two-sample batch: **385 active codes**, perplexity
**366.940693**, 460 valid tokens. This validates the training-checkpoint format
(three decoder layers, saved VQ options) in addition to the official format
(four decoder layers). Higher fixed-batch perplexity alone does not mean this
checkpoint is better than the official VQ. This is distinct from the final five-epoch
checkpoint's result below.

The complete 126-sample TUSL CPU run passed `scripts/audit_downstream_eval.py`:
independent strict parsing, sklearn metrics, unique full-split index coverage,
prediction hashes and exact correspondence between invalid rows and error logs.

## Full allocated-GPU baseline — 2026-11-08

TI-ONE task `train-1713043048264500992` completed **SUCCEED**, worker exit code
**0**, 05:11:14–05:28:01 UTC (1007 seconds). It used two allocated H20 GPUs,
BF16, batch size 32 per rank, and the existing instruction checkpoint above.
No training was performed. Every validation/test sample was evaluated once.

| Task | Split | Samples | Invalid | Accuracy | Balanced accuracy | Weighted F1 | Kappa | Conditional-option AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TUSL | Validation | 66 | 0 | 50.0000% | 51.8053% | 49.3851% | 0.246106 | N/A |
| TUSL | Test | 60 | 0 | 25.0000% | 46.9697% | 26.1530% | -0.075269 | N/A |
| TUAB | Validation | 73,627 | 0 | 78.8434% | 79.5218% | 78.6491% | 0.581712 | 88.5412% |
| TUAB | Test | 36,945 | 0 | 79.0472% | 78.1112% | 78.6408% | 0.571470 | 88.0366% |

The full run contains **110,698** predictions. CPU float32 and GPU BF16 TUSL
outputs differ on a few wrong-class choices while retaining the same accuracy
and balanced accuracy; preserve their separate precision/batch provenance.
Neither run demonstrates a downstream improvement from Agent A's new VQ.
The independent audit passed for all 110,698 predictions: exact index coverage,
strict raw-text parsing, error-file correspondence, prediction hashes and
sklearn recomputation of every reported classification metric and TUAB AUC.

Machine-readable evidence:
[GPU summary](../results/downstream_constrained_baseline_gpu_20261108.json),
[independent GPU audit](../results/downstream_constrained_baseline_gpu_audit_20261108.json),
[saved-output parser audit](../results/downstream_parser_audit_20261108.json),
[encoder provenance](../results/downstream_encoder_provenance_20261108.json),
[official fixed-batch VQ check](../results/vq_official_fixed_batch_quality_20261108.json).

## Agent A final five-epoch VQ received — 2026-11-08

The source completion record reports **5 complete epochs, 4485 updates**, with
both worker nodes exiting 0. The immutable epoch-5 snapshot SHA-256 is
`546e07e0b455b15ef9374a96bc77dead226bba8d00afa6e738a625fbb266de07`.

The updated quality tool ran on the exact frozen two-sample batch used above:
**241 active codes / 460 valid tokens**, perplexity **123.651409**, top-code
fraction **9.3478%**; 1588 padding tokens excluded. Encoder features and codebook
weights are finite, parameters are unchanged, and no runtime buffers changed.
This is not sufficient evidence to call the new VQ high quality.

[Final VQ fixed-batch evidence](../results/vq_fixed_e5_fixed_batch_quality_20261108.json).

At the user's request, an isolated tmux workflow now handles waiting, completion
checks, CPU quality checks, scoped TI-ONE inference submission and independent
result auditing. It uses task-name deduplication and a process lock, never starts
training, and leaves errors visible with logs. Its session is `downstream-eval`;
Agent A's training/monitoring sessions were not modified.

The resulting inference task is `train-1713078005898010496`,
`neurolm-downstream-constrained-newvq-e5-20261108-r1`. It retains the same original
instruction checkpoint, adapter, GPT weights, dataset splits, prompts and decoding
protocol, and explicitly overrides only the embedded encoder with the new VQ
encoder. **This is an unaligned encoder-swap diagnostic, not a newly pretrained
and instruction-tuned NeuroLM.** Full downstream results are recorded only after
the worker, platform and independent audit all finish successfully.

## Completed new-encoder diagnostic and independent audit — 2026-11-08

Task `train-1713078005898010496` completed **SUCCEED**, worker exit **0**,
2026-11-08T06:20:40Z–2026-11-08T06:35:47Z
(907 seconds). The tmux workflow completed its audit;
no training was started. All **110,698** predictions passed independent strict
parsing, exact split coverage, file hashing and sklearn metric recomputation.

| Task | Split | Samples | Invalid | Accuracy | Balanced accuracy | Weighted F1 | Kappa | Conditional-option AUC |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| TUSL | Validation | 66 | 0 | 25.7576% | 33.3333% | 10.5513% | 0.000000 | N/A |
| TUSL | Test | 60 | 0 | 55.0000% | 33.3333% | 39.0323% | 0.000000 | N/A |
| TUAB | Validation | 73,627 | 0 | 47.1009% | 50.0000% | 30.1629% | 0.000000 | 40.4289% |
| TUAB | Test | 36,945 | 0 | 53.8828% | 50.0000% | 37.7346% | 0.000000 | 37.8194% |

The baseline and encoder-swap runs have identical inference source hashes,
precision, two allocated H20 GPUs, per-rank batch size, sorted split-file hashes,
answer token sequences and prompts. Only the explicit VQ encoder override
differs. Every TUSL output in the swap run is class A: test accuracy rises to
55% because A is the majority class, while balanced accuracy is only 33.33%
and kappa is zero. This is not an improvement over the baseline TUSL test
balanced accuracy of 46.97%. TUAB likewise always predicts No, giving 50%
balanced accuracy on both splits. Format validity and classification quality
are separate checks.

These results do not isolate intrinsic VQ quality: the replacement encoder is
not aligned with the old adapter/LLM. A matched stage-two/three checkpoint is
needed before drawing conclusions about the new training pipeline. This task
has not started that training. Fixed-batch perplexity also cannot resolve this
confound or establish that the full codebook is healthy.

Evidence: [new-encoder summary](../results/downstream_constrained_newvq_gpu_20261108.json),
[independent audit](../results/downstream_constrained_newvq_gpu_audit_20261108.json),
[platform completion](../results/downstream_newvq_platform_20261108.json),
[matched-protocol comparison](../results/downstream_encoder_swap_comparison_20261108.json).
Raw predictions and EEG inputs remain on the authorized server; only code,
configuration, hashes and aggregate measurements are published.
