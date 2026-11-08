# Five-task instruction tuning on two H20 GPUs

This is a modified five-task protocol: TUAB, TUEV, HMC, Workload, and TUSL;
SEED is excluded. One epoch with seed 42 is not the paper's six-task,
three-seed result. This experiment starts from the published NeuroLM-B
checkpoint, independently of the separate VQ training experiment.

## Execution contract

- One TI-ONE worker with two H20 GPUs; the Notebook's GPUs are not used.
- Python 3.11.8, PyTorch 2.2.2 / CUDA 12.1, NumPy 1.26.4.
- Original labeled downstream pickles are transferred without changing their
  bytes. VQ pretraining inputs cannot substitute for these labeled inputs.
- The author's Appendix B prompts and answer-token boundaries are retained.
- Official `NeuroLM-B.pt` and `VQ.pt` checksums are checked before execution.
  The instruction branch restores the EEG encoder contained in NeuroLM-B;
  it does not load a checkpoint produced by the separate VQ experiment.
- Per GPU: EEG batch 32, text batch 8, gradient accumulation 8. Nominal
  effective batches are EEG 512 and text 128 across two GPUs.
- Peak/minimum learning rate: 5e-4 / 5e-5; warmup ratio 0.1; weight decay
  0.1; gradient clipping 1. The author's scheduler uses the explicitly
  calculated positive `warmup_steps` in preference to `warmup_epochs`.

The complete training set contains 459,656 examples. With the existing
distributed sampler and `drop_last=True`, an epoch has 7,182 microsteps and
898 optimizer updates; eight training examples are omitted by `drop_last`.
The last update accumulates six microsteps (EEG 384,
text 96). Learning rate scheduling remains per microstep, as in the patched
author training loop. Validation and test loaders retain every example.

## Diagnostic before the full epoch

`scripts/smoke_instruction_optimizer.py` uses the actual author loaders and
official B checkpoint. It cycles through five tasks over eight microsteps,
adds next-token OpenWebText loss, runs backward, clips gradients, and applies
one AdamW update. It records both ranks' losses, supervision counts, gradient
norms, peak GPU memory, and an observed parameter change, then saves and
hashes a diagnostic checkpoint.

The diagnostic uses 64 original training examples per task, with two
validation and two test files per task only to satisfy loader construction.
It does **not** evaluate these validation/test files. Its fixed learning rate
of 5e-4, repeated small sample pool, and one optimizer update are diagnostic
choices. Its checkpoint must never initialize the full experiment.

Run inside an allocated two-GPU worker, with private paths supplied through
environment variables:

```bash
"$NEUROLM_PYTHON" -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/smoke_instruction_optimizer.py \
  --author-dir "$AUTHOR_CODE" --data-dir "$DIAGNOSTIC_DATA" \
  --checkpoint "$OFFICIAL_B" --text-bin "$OPENWEBTEXT_TRAIN_BIN" \
  --out-dir "$DIAGNOSTIC_OUTPUT"
```

Success requires task termination, exit code 0, and a passing
`diagnostic_summary.json` plus its checkpoint. Finite forward loss alone is
not sufficient. Diagnostic loss is not a task accuracy or a paper result.

## Full epoch and results

The configuration is
`experiments/neurolm_b_five_task_tione_two_h20.yaml`. Within the allocated
worker, set `NEUROLM_ROOT` to the private project directory,
`NEUROLM_CODE_ROOT` to the frozen bundle containing `author/` and
`reproduction/`, and `NEUROLM_RUN_ID` to a new run name, then execute:

```bash
bash "$NEUROLM_CODE_ROOT/reproduction/scripts/run_five_task_tione_two_h20.sh"
```

The launcher rejects existing output directories, unexpected GPU topology,
occupied GPUs, incomplete split counts, and mismatched input checksums. It
uses the published B checkpoint afresh, trains one epoch, evaluates all five
validation/test sets, and runs `summarize_instruction_run.py`.

Completion requires `exit_code.txt` equal to 0, the final checkpoint and
SHA-256, five metric records, exact per-split prediction index coverage,
matching parse-failure counts, and `summary.json`. The author's generated
answer parser and hard-label AUC behavior remain documented limitations.
Only code, configuration, aggregate measured results, and reproducibility
notes are public; datasets, weights, predictions containing source
identifiers, environments, credentials, and private paths stay off GitHub.

## Observed status, 2026-11-07

TI-ONE STS authentication succeeded in this execution. The official weights,
full OpenWebText files, tokenizer cache, 340-file diagnostic snapshot, and
relocated training environment passed their checks.

The two-H20 diagnostic finished with platform status `SUCCEED` and exit code
0. Platform runtime was 54 seconds (34.30 seconds inside the diagnostic).
Both ranks reported a finite pre-clipping gradient norm of 1627.34 and a
maximum position-embedding update of 0.00051525. Peak allocated memory was
19.13 GiB per GPU. The saved 2,377,444,354-byte diagnostic checkpoint's
independently recomputed SHA-256 is
`d27d6de366ea067452c4bce45c3b29b720c370d7fe0b3169f5ff87088fb5ac54`.

| Task | Diagnostic instruction loss, mean across two ranks |
| --- | ---: |
| TUAB | 25.04476 |
| TUEV | 25.96094 |
| HMC | 8.72998 |
| Workload | 14.86637 |
| TUSL | 29.61926 |

These losses were measured before the single optimizer update; they do not
show a learning curve or validation/test performance. The large gradient
norm is clipped to 1 before the update. Full details are in
[`instruction_five_task_optimizer_diagnostic_20261107.json`](../results/instruction_five_task_optimizer_diagnostic_20261107.json).
At that diagnostic snapshot, full input transfer and complete-epoch results
were still pending. The later completed result is recorded below.


## Completed epoch and recovered evaluation, 2026-11-08

All 660,896 original labeled inputs were verified. The full epoch completed
7,182 microsteps and 898 optimizer updates. The final checkpoint is
2,377,398,188 bytes with SHA-256
`a10a23f6f4a69058c996c01096d47b0806dafd83a0d0ca6506e274a5c6372174`.

Runtime failures and recovery are retained in the experiment history:

1. The first full attempt stopped before its first batch because the deeply
   nested DataLoader IPC path exceeded the Unix socket path limit. The retry
   used a short, container-local `/tmp` directory for ephemeral IPC only.
2. That retry completed training and TUAB evaluation, then aborted during
   TUEV evaluation with `gmp: overflow in mpz type`. Its platform status is
   `FAILED`; the trained checkpoint was kept. A separate evaluation-only
   worker with SymPy's Python integer backend also failed. Its fatal-error
   trace located the abort in the inference LM-head. An isolated diagnostic
   reproduced it for batches 7, 8, and 16, while 1, 33, and 48 passed.
3. Patch `0011` pads only the inference linear-layer input to at least 48 rows
   and discards extra output rows. All 16 shape checks passed against an FP32
   CPU reference, then complete evaluation succeeded with the original
   symbolic backend restored. EEG inputs, masks, prompts, parser, BF16,
   training code, and weights were unchanged; no optimizer steps were taken.
   The underlying native-library cause is not established.

To recover evaluation, keep the frozen author bundle, set `NEUROLM_TRAIN_RUN`
to the completed training directory, select a fresh `NEUROLM_RUN_ID`, and run:

```bash
bash "$NEUROLM_CODE_ROOT/reproduction/scripts/run_five_task_tione_eval_two_h20.sh"
```

The bundle must include `patches/0011-inference-lm-head-small-batch.patch`,
applied to the frozen author tree with `git apply`. The patch preserves the
upstream CRLF bytes, as specified by its `.gitattributes` exception. Set
`NEUROLM_VERIFIED_INPUT_RUN` to the prior run containing the passing full
input preflight; fixed inputs must remain unchanged during recovery. The
author loaders enumerate every split again and the audit checks their exact
counts. The trained checkpoint is rehashed before and after evaluation.

The evaluation task finished with `SUCCEED` and exit
code 0. All 201,240 validation/test examples were covered exactly once.
The independent audit reloaded the checkpoint to verify optimizer steps,
recomputed its SHA-256, and recomputed each metric from every saved prediction.
Author metrics and stricter metrics counting unparseable answers as errors
are both retained. See the [measured result report](../results/INSTRUCTION_FIVE_TASK_20261108.md).

This remains a five-task, single-seed, one-epoch modified protocol. It does not
establish the paper's six-task, five-epoch, three-seed results. Separate VQ
training is unrelated to the initialization used here.
