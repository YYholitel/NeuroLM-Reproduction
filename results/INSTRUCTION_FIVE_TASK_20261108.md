# Five-task NeuroLM-B instruction tuning — 2026-11-08

Completed one full training epoch with seed 42 on two H20 GPUs. All five
validation and test sets were evaluated: **201,240 predictions**, with exact
index coverage and no duplicates. The recovery evaluation worker exited with
code 0 and TI-ONE reported `SUCCEED`. The original training worker
completed all optimizer updates but subsequently failed during evaluation;
it remains recorded as `FAILED`. Total parse failures: **78**.

This is a five-task modified protocol, excluding SEED. The paper reports
six-task training for five epochs and three seeds; its means and standard
deviations below provide background, not a matched-protocol comparison.
The recorded TUAB v3.0.1, patient-disjoint split, and preprocessing differences
remain as documented in the repository's dataset and decision records.

| Task | Validation BA | Test BA | Strict test BA | Test parse failures | Paper BA (background) |
| --- | ---: | ---: | ---: | ---: | ---: |
| TUAB | 0.796144 | 0.782054 | 0.782054 | 0 | 0.7826 ± 0.0065 |
| TUEV | 0.559123 | 0.531339 | 0.531339 | 0 | 0.4560 ± 0.0048 |
| HMC | 0.657105 | 0.658599 | 0.658599 | 0 | 0.6737 ± 0.0050 |
| Workload | 0.579310 | 0.520690 | 0.520690 | 0 | 0.6172 ± 0.0113 |
| TUSL | 0.391481 | 0.258838 | 0.064394 | 42 | 0.6734 ± 0.0436 |

**TUSL is a poor result in this run.** Answers failed parsing for 36/66
validation examples and 42/60 test examples. Counting every parse failure as
an error gives validation/test BA **25.35% / 6.44%**, and test accuracy
**10.00%**. Failed generations include incomplete `(B` or `(C` before the end
token (29 validation / 34 test cases), and the unsupported option `(E)`
(7 validation / 8 test cases). The original parser and all predictions were
retained; these failures were not removed or reassigned in the strict scores.

BA means balanced accuracy. Author metrics retain the original answer parser
and hard-label AUC implementation. Binary parse failures map to class 0 for
classification; multiclass failures map to the last class. Strict BA always
counts an unparseable answer as an error. The audit recomputed every reported
author metric from the saved predictions and separately calculated strict
accuracy and BA; no predictions were dropped.

## Configuration and evidence

- Official NeuroLM-B initialization; no newly trained VQ checkpoint was used.
- Python 3.11.8, PyTorch 2.2.2 / CUDA 12.1, BF16; one worker, two H20 GPUs.
- Per GPU EEG/text batch 32/8, accumulation 8, nominal global batch 512/128.
- 7,182 microsteps and 898 optimizer updates; `drop_last` omits eight training
  examples. The final update accumulates six microsteps (global batch 384/96).
- Peak/minimum LR 5e-4/5e-5; 718 warmup microsteps, weight decay 0.1,
  gradient clipping 1. Original Appendix B prompts and answer boundaries.
- All 660,896 input examples counted; original labeled EEG files were
  transferred with SHA-256 checks, and official checkpoint/text hashes matched.
- The first full attempt stopped before its first batch because its IPC path
  exceeded the Unix socket limit. The training retry only moved ephemeral
  DataLoader IPC to a short container-local `/tmp` path. Persistent inputs,
  logs, and checkpoints stayed on shared storage. Training parameters and
  author code were unchanged between attempts.
- That retry completed the epoch, saved its checkpoint, and completed TUAB
  evaluation, then aborted during TUEV evaluation with `gmp: overflow in mpz
  type`. Its logs and failed platform state are preserved.
- A first evaluation-only retry with SymPy's Python integer backend also
  failed. Fatal-error tracing located the native abort in the final linear
  output layer. An isolated two-H20 diagnostic reproduced the abort at
  batches 7, 8, and 16; batches 1, 33, and 48 returned finite outputs.
- The final evaluation used patch `0011`: only the inference LM-head input is
  padded to at least 48 rows, and extra output rows are discarded. EEG input,
  attention masks, prompts, parser, training code, weights, and BF16 stay
  unchanged. All 16 padded shape checks passed against an FP32 CPU reference.
  The original symbolic backend was restored. The shape-dependent failure is
  isolated; its underlying native-library cause is not established.
- The checkpoint still records 898 optimizer updates and 7,182 microsteps,
  and its hash was unchanged by evaluation. The final retry reused the
  already-passing full input manifest and independently rechecked all split
  lengths recorded by the author loaders.
- All 110,572 TUAB validation/test predictions are byte-identical to those
  saved by the original worker before it failed (four matching file hashes).

Final checkpoint: **2,377,398,188 bytes**. Independently
recomputed SHA-256: `a10a23f6f4a69058c996c01096d47b0806dafd83a0d0ca6506e274a5c6372174`.

- [Full measured metrics](instruction_five_task_e1_s42_20261108.json)
- [Metric CSV and paper background values](instruction_five_task_e1_s42_metrics_20261108.csv)
- [Independent prediction and checkpoint audit](instruction_five_task_e1_s42_audit_20261108.json)
- [Full input preflight](instruction_five_task_inputs_20261108.json)
- [Logged training losses and learning rates](instruction_five_task_e1_s42_steps_20261108.jsonl)
- [Execution protocol and diagnostic](../docs/INSTRUCTION_TI_ONE_TWO_H20.md)
- [Training code manifest](../experiments/instruction_two_h20_code_manifest_20261108_r2.json)
- [Failed recovery code manifest](../experiments/instruction_two_h20_eval_code_manifest_20261108_r3.json)
- [Completed evaluation code manifest](../experiments/instruction_two_h20_eval_code_manifest_20261108_r4.json)
- [Native shape-failure diagnostic](instruction_bf16_lm_head_shape_probe_20261108.json)
- [Padded LM-head verification](instruction_lm_head_padding_20261108.json)

Raw EEG, model weights, source-identifying predictions, environments, private
server paths, and credentials are not included in this public report.
