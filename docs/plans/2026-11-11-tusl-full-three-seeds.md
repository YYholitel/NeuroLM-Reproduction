# TUSL full coverage and three-seed rerun Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Retain all 112 TUSL recordings and run seeds 42, 43, 44 through continued stage-two pretraining from the author's released NeuroLM-B.pt, followed by five epochs of stage-three instruction training and full evaluation on TI-ONE.

**Architecture:** Preserve model definitions and historical experiments. Reuse unchanged non-TUSL train shards, retain the original sorted-file TUSL split, and preserve existing REF signals exactly. Recover LE recordings using their actual supported electrode names (21 or 23), preserving reference metadata. Update only TUSL loading to use sample channel names. Each seed gets a fresh optimizer, output directory, and independently chained stage-two/stage-three tasks. Official VQ.pt supplies frozen targets and must match the official NeuroLM EEG encoder exactly.

**Tech Stack:** Python, MNE 1.7.1, PyTorch/DDP, TI-ONE STS Notebook API, tmux, matplotlib.

1. Inventory every TUSL EDF and annotation; preserve train/test/eval file membership (67/22/23). Validate no records/events silently skipped and no fabricated channels. Existing 245 REF windows must remain numerically identical. Expected new split windows: train 174, eval 66, test 60.
2. Create a versioned dataset and provenance manifest. Validate both 21- and 23-channel PickleLoader and TUSLLoader paths, masks, normalization, and labels. Keep only training members in pretraining (459711 total), with validation/test excluded.
3. Strict-load official NeuroLM-B.pt and VQ.pt; verify hashes and encoder alignment. Implement explicit official-checkpoint initialization without changing model definitions. Freeze target VQ and EEG encoder; fresh optimizer for each seed.
4. Stage two: 20 epochs, 2 warmup, global EEG/text batches 480/32, peak/min LR 6e-4/6e-5, weight decay .1, clip 1. Seeds 42/43/44 run concurrently, subject to freshly checked prepaid capacity. Stage three: five epochs, existing instruction protocol, seed-matched new stage-two output, dynamic TUSL channel metadata.
5. Prepare scoped task specs, dry-run, deduplicate, launch under a tmux driver. Chain stage three only after stage-two completion and checkpoint audits. Stop the affected pipeline on OOM, data errors, nonfinite loss or failed frozen-state assertions; do not alter architecture to hide errors.
6. Check live logs, actual allocated GPUs, finite losses and frozen state. Evaluate all validation/test samples after the final instruction epoch, audit coverage and predictions, plot per-seed curves and summarize mean/sample SD across seeds. Preserve raw evidence and distinct experiment documentation.

Interpretation: This is continued pretraining of the author's released checkpoint, not pretraining from scratch. The retained TUSL split is file-level and is not claimed to be patient-independent. Old one-epoch results are historical context, not a controlled comparison for this five-epoch rerun.
