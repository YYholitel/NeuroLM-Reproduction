# VQ Codebook Fix Implementation Plan

> Execution: implement the user's authorized diagnosis and fixes in an isolated worktree; no subagents and no replacement of the completed run. Routine implementation choices do not require a new permission round.

**Goal:** Identify concrete causes of codebook underuse, fix padding contamination, add configurable synchronized dead-code replacement, and verify the changes without claiming a new successful training run.

**Architecture:** Preserve normalized EMA (decay 0.99) and beta 1.0 initially. Pass a separate valid-token mask into quantization, use only valid vectors for initialization, EMA and commitment loss, normalize reconstruction by valid elements, pool initialization/replacement candidates across ranks, and replace persistently unused codes identically on all ranks. Keep the completed r1 code/checkpoints immutable; distribute the new implementation as a patch and use an isolated runtime copy.

**Tech Stack:** Python, PyTorch, CPU Gloo distributed tests, existing indexed-tar EEG dataset, existing Tencent SSH and Notebook identity.

## Evidence and alternatives

- Existing implementation is normalized EMA with frozen codebook weights; optimizer LR does not update the dictionary directly. Beta is already 1.0. The advertised epsilon is used by an unused alternative update method; the active branch masks empty bins.
- Padding is masked for attention and reporting but not for quantizer initialization, EMA or commitment loss. Reconstruction loss is averaged over the entire padded tensor. This makes the effective loss balance depend on sample length.
- `reset_cluster_size` resets a usage statistic, not dead codebook vectors.
- First fix masks and normalization; preserve hyperparameters for attribution. A second configurable dead-code intervention can then be compared with mask-only behavior. Blind beta/LR changes are lower priority because they do not repair these data-flow problems.
- Reduced codebook size is a possible later experiment but changes the 8192-code setup and is not required for the immediate code fix.

## Task 1: Audit actual checkpoint and real tokens

Create `scripts/audit_vq_codebook.py`. Verify runtime file hashes, inspect actual EMA state and its unused legacy fields, compare epoch-10 and epoch-50 codebook rows, and call the actual tokenizer on a stratified 32-window batch. Print masked valid-token and padding-only histograms using exact integer bincounts, top-1/top-5 shares, perplexity, indices and mask. Save aggregate JSON and per-token indices without exporting EEG to Git.

## Task 2: Reproduce failures before editing

Create `scripts/test_vq_codebook_fix.py`. Quantizer-only tests must demonstrate that appending/changing invalid vectors affects the old EMA and loss. Test full VQ masking/normalization separately with a small model; do not promise whole-encoder padding-length invariance because the existing TemporalConv GroupNorm spans the token dimension.

## Task 3: Implement isolated patch

Prepare `.local/vq_codebook_fix_bundle` from the final immutable r1 bundle. Edit only `model/norm_ema_quantizer.py`, `model/model_vq.py`, `train_vq.py` and, if needed, its logging helper. Generate `patches/0017-vq-valid-token-ema-and-dead-code-reset.patch` against the exact r1 source hashes.

- Quantize valid vectors only; retain zero placeholder indices for masked positions for downstream index-range compatibility.
- Use globally weighted losses in DDP so uneven valid-token counts across ranks do not bias gradients.
- Preserve current normalized centroid EMA and avoid a dense one-hot assignment matrix.
- Synchronize k-means candidates and chosen embeddings; support a rank with zero valid vectors, reject a globally empty batch consistently.
- Add a configurable consecutive-unused-step policy with a grace period and reset interval; reset from current valid vectors and broadcast replacements. Statistics must report real assignments, not pretend that reset entries were used.
- In eval mode do not initialize an uninitialized dictionary or mutate weights/statistics; allow inference on initialized checkpoints without collectives.
- Expose decay, beta, initialization-pool size and dead-code policy as CLI configuration. Preserve all defaults unrelated to the explicit repair.

## Task 4: Verify meaningful properties

Run local/remote CPU tests: invalid-vector perturbation leaves valid assignments, EMA and gradients unchanged; invalid gradients are zero; valid losses are invariant to added masked positions; eval is pure; dead-code replacement changes only eligible entries; actual usage counts exclude padding; two Gloo ranks produce identical dictionaries and losses weighted by global valid counts, including empty-local-rank cases. Run a small actual VQ optimizer step and state round-trip. Demonstrate which regression tests fail with the old implementation.

## Task 5: Report and publish

Print the real token histogram and source settings, distinguish established bugs from unproven explanations of the full training collapse, publish patch/tests/audit to `codex/vq-codebook-fix-20261108`, and record that no new 50-epoch training was run. Recommend controlled mask-only versus mask-plus-reset training with held-out validation before beta sweeps or replacing the final checkpoint.
