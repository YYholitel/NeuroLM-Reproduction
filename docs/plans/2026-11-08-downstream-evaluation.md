# Downstream evaluation work

Scope: evaluation only; do not edit `train_vq.py` or launch training.

1. Add strict answer parsing, explicit invalid-output logs, and metrics retaining
   every example. Keep historical author results and tolerant-parser ablations separate.
2. Build legal continuation tries from the actual loader token tensors (including
   EOS), preserving Appendix B prompts and BPE boundaries. Test constrained greedy
   decoding with a deliberately adversarial fake model and the real model.
3. Add `eval_vq_quality.py`: fixed input batch, strict checkpoint loading, valid-token
   histograms/perplexity, checkpoint/input/code hashes, no weight updates.
4. Audit saved predictions offline; run inference-only evaluation on available
   checkpoints. Record complete coverage, failures, decoding protocol and provenance
   in `docs/Downstream_eval_log.md`.
5. Inspect Agent A's checkpoint handoff. A VQ-only checkpoint is not a complete
   downstream model; replacing the embedded encoder is an explicitly labelled
   compatibility diagnostic, not evidence of successful stage-two/three training.

Work in this isolated worktree. Publish only code and aggregate results; raw outputs
and EEG inputs stay private. No other agent's files or running jobs are modified.
