"""Property tests for patch 0017. CPU only; --mode ddp runs under torchrun."""
import argparse
import copy
import importlib.util
import json
import os
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F


def load_quantizer(root, name):
    spec = importlib.util.spec_from_file_location(name, Path(root) / 'model/norm_ema_quantizer.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def same_state(a, b):
    for key, value in a.state_dict().items():
        torch.testing.assert_close(value, b.state_dict()[key], rtol=0, atol=0, msg=key)


def baseline(root):
    module = load_quantizer(root, 'baseline_quantizer')
    torch.manual_seed(21)
    a = module.NormEMAVectorQuantizer(8, 4, beta=1.0)
    b = copy.deepcopy(a)
    valid = torch.randn(1, 3, 4)
    padded = torch.cat([valid, torch.tensor([[[1., 0., 0., 0.]]]).repeat(1, 13, 1)], dim=1)
    _, la, _ = a(valid)
    _, lb, _ = b(padded)
    assert not torch.allclose(la, lb)
    assert not torch.equal(a.embedding.weight, b.embedding.weight)
    return dict(reproduced_padding_changes_loss_and_ema=True,
                valid_only_commitment=float(la), padded_commitment=float(lb))


def unit(module):
    Q = module.NormEMAVectorQuantizer
    torch.manual_seed(21)
    a = Q(8, 4, beta=1.0, dead_code_patience=0)
    b = copy.deepcopy(a)
    x = torch.randn(2, 3, 4, requires_grad=True)
    padded = torch.cat([x.detach(), torch.randn(2, 13, 4) * 1000], 1).requires_grad_()
    mask = torch.arange(16).expand(2, -1) < 3
    qa, la, ia = a(x)
    qb, lb, ib = b(padded, mask)
    la.backward(); lb.backward()
    torch.testing.assert_close(qa, qb[:, :3])
    torch.testing.assert_close(la, lb)
    assert torch.equal(ia, ib.reshape(2, 16)[:, :3].reshape(-1))
    torch.testing.assert_close(x.grad, padded.grad[:, :3])
    assert not padded.grad[:, 3:].any() and not qb[:, 3:].any()
    same_state(a, b)
    assert abs(float(a.cluster_size.sum()) - 6 * (1 - a.decay)) < 1e-6
    init_a = Q(8, 4, 1., kmeans_init=True, dead_code_patience=0)
    init_b = copy.deepcopy(init_a)
    torch.manual_seed(63); init_a(x.detach())
    torch.manual_seed(63); init_b(padded.detach(), mask)
    same_state(init_a, init_b)

    # Pure evaluation, even after training; invalid values never enter quantization.
    a.eval(); snapshot = copy.deepcopy(a)
    x_eval = torch.cat([x.detach(), torch.full((2, 1, 4), float('nan'))], 1)
    valid = torch.arange(4).expand(2, -1) < 3
    assert torch.isfinite(a(x_eval, valid)[0]).all()
    same_state(a, snapshot)
    try:
        Q(8, 4, 1., kmeans_init=True).eval()(x.detach())
        raise AssertionError('uninitialized eval was allowed')
    except RuntimeError as exc:
        assert 'uninitialized' in str(exc)

    # Count only genuine assignments. Replace persistently dead rows with valid vectors.
    r = Q(4, 2, 1., decay=0., dead_code_patience=2, dead_code_interval=1)
    with torch.no_grad():
        r.embedding.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]))
    for _ in range(2):
        r(torch.tensor([[[1., 0.], [0., -100.]]]), torch.tensor([[True, False]]))
    assert int(r.last_reset_count) == 3 and int(r.total_replacements) == 3
    torch.testing.assert_close(r.embedding.weight, torch.tensor([[1., 0.]]).expand(4, -1))
    assert torch.equal(r.cluster_size, torch.tensor([1., 0., 0., 0.]))
    assert not r.code_age.any()
    restored = Q(4, 2, 1., decay=0., dead_code_patience=2, dead_code_interval=1)
    restored.load_state_dict(r.state_dict()); same_state(r, restored)

    # Outer mixed precision must not silently change dictionary/EMA calculations.
    fp32 = Q(8, 4, 1., dead_code_patience=0); mixed = copy.deepcopy(fp32)
    x32 = torch.randn(2, 3, 4)
    out1, loss1, _ = fp32(x32)
    with torch.autocast('cpu', dtype=torch.bfloat16):
        out2, loss2, _ = mixed(x32)
    torch.testing.assert_close(out1, out2, rtol=0, atol=0)
    torch.testing.assert_close(loss1, loss2, rtol=0, atol=0)
    same_state(fp32, mixed)

    # Loss normalization ignores both masked predictions and masked targets.
    values = torch.tensor([[[1., 3.], [float('nan'), float('nan')]]], requires_grad=True)
    mean = module.masked_token_mean(values, torch.tensor([[True, False]]), sync=False)
    mean.backward()
    assert float(mean) == 2. and not values.grad[:, 1:].any()
    return dict(masked_assignments_ema_loss_gradients_invariant=True,
                initialization_excludes_padding=True,
                padding_gradient_zero=True, eval_pure=True, dead_reset_valid_only=True,
                state_roundtrip=True, fp32_under_autocast=True)


def distributed_checks(module):
    rank = dist.get_rank()
    Q = module.NormEMAVectorQuantizer
    torch.manual_seed(73 + rank)
    q = Q(4, 3, 1., kmeans_init=True, init_samples_per_rank=8,
          dead_code_patience=2, dead_code_interval=1)
    # Empty local rank is legal. The other rank supplies every initialization vector.
    x = torch.randn(1, 8, 3, requires_grad=True)
    mask = torch.full((1, 8), rank == 1, dtype=torch.bool)
    q(x, mask)[1].backward()
    assert torch.isfinite(x.grad).all()
    if rank == 0:
        assert not x.grad.any()
    for tensor in q.state_dict().values():
        other = tensor.clone(); dist.broadcast(other, src=0)
        torch.testing.assert_close(tensor, other, rtol=0, atol=0)
    # Eval on rank zero alone must not perform collectives or mutate state.
    dist.barrier()
    if rank == 0:
        q.eval(); state = copy.deepcopy(q); q(x, mask); same_state(q, state); q.train()
    dist.barrier()

    # DDP's average gradients must equal the single concatenated-batch objective.
    weight = torch.nn.Parameter(torch.tensor(2.))
    local_x = torch.tensor([1.]) if rank == 0 else torch.tensor([2., 3., 4.])
    local_errors = (weight * local_x).square().reshape(1, -1, 1)
    loss = module.masked_token_mean(local_errors, torch.ones(1, len(local_x), dtype=torch.bool), sync=True)
    loss.backward()
    dist.all_reduce(weight.grad); weight.grad.div_(dist.get_world_size())
    mean_loss = loss.detach(); dist.all_reduce(mean_loss); mean_loss.div_(dist.get_world_size())
    torch.testing.assert_close(weight.grad, torch.tensor(30.))
    torch.testing.assert_close(mean_loss, torch.tensor(30.))

    # Match DDP's initial state broadcast, including unused legacy EMA parameters.
    torch.manual_seed(111)
    # All-dead candidates are synchronized and actual bins exclude padding.
    r = Q(4, 2, 1., decay=0., dead_code_patience=2, dead_code_interval=1)
    with torch.no_grad():
        r.embedding.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]))
    for _ in range(2):
        vector = [1., 0.] if rank == 0 else [0., 1.]
        r(torch.tensor([[vector, [0., -100.]]]), torch.tensor([[True, False]]))
    assert int(r.last_reset_count) == 2
    assert float(r.cluster_size.sum()) == 2.
    assert {tuple(v) for v in r.embedding.weight[2:].tolist()} == {(1., 0.), (0., 1.)}
    for tensor in r.state_dict().values():
        other = tensor.clone(); dist.broadcast(other, src=0)
        torch.testing.assert_close(tensor, other, rtol=0, atol=0)
    try:
        q(x, torch.zeros_like(mask))
        raise AssertionError('globally empty batch was accepted')
    except ValueError as exc:
        assert 'valid' in str(exc)
    return dict(world_size=dist.get_world_size(), global_initialization_synchronized=True,
                empty_local_rank_supported=True, globally_weighted_gradient=True,
                eval_without_collectives=True, synchronized_reset=True, global_empty_rejected=True)


def integration(root):
    sys.path.insert(0, str(Path(root).resolve()))
    from model.model_vq import VQ
    from model.model_neural_transformer import NTConfig
    encoder = NTConfig(n_layer=1, n_head=12, n_embd=768, block_size=8, out_chans=16, in_chans=1)
    decoder = NTConfig(n_layer=1, n_head=12, n_embd=768, block_size=8, in_chans=8)
    model = VQ(encoder, decoder, n_embed=16, embed_dim=8, dead_code_patience=0)
    x = torch.randn(2, 8, 200)
    mask = torch.arange(8).expand(2, -1) < torch.tensor([[3], [5]])
    chans = torch.zeros(2, 8, dtype=torch.long); times = chans.clone()
    yf = torch.randn(2, 8, 100); yr = torch.randn_like(x)
    optimizer = model.configure_optimizers(0.01, 5e-5, (.9, .999), 'cpu')
    assert all(p is not model.quantize.embedding.weight for g in optimizer.param_groups for p in g['params'])
    loss, _, log = model(x, yf, yr, chans, times, mask)
    assert int(log['train/valid_code_counts'].sum()) == 8
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optimizer.step(); optimizer.zero_grad()
    # Forward passes the mask all the way through; observed EMA total is 8 * .01.
    torch.testing.assert_close(model.quantize.cluster_size.sum(), torch.tensor(.08))
    model.eval(); snapshot = copy.deepcopy(model.state_dict())
    indices = model.get_codebook_indices(x, chans, times, mask)
    assert not indices[~mask].any()
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in snapshot.items())
    p = torch.randn(2, 3, 7); target = torch.randn_like(p)
    base_loss = model.calculate_rec_loss(p, target, torch.ones(2, 3, dtype=torch.bool))
    padded_loss = model.calculate_rec_loss(torch.cat([p, torch.randn(2, 5, 7)], 1),
        torch.cat([target, torch.full((2, 5, 7), 1e6)], 1), torch.arange(8).expand(2, -1) < 3)
    torch.testing.assert_close(base_loss, padded_loss)
    from vq_monitor import fixed_probe, epoch_summary
    model.train()
    snapshot = copy.deepcopy(model.state_dict())
    probe = fixed_probe(model, (x, yf, yr, chans, times, mask), 'cpu', nullcontext())
    assert model.training and all(torch.equal(v, model.state_dict()[k]) for k, v in snapshot.items())
    row = dict(reconstruction=2., frequency=1., raw=1., codebook=.001,
               adversarial=1.4, adversarial_eeg=.7, adversarial_text=.7,
               lr=5e-5, alpha=.4, reconstruction_padded_mean=1., dead_code_resets=3,
               loss_normalization='global_valid_elements')
    summary = epoch_summary(5, [row, row], log['train/valid_code_counts'],
                            log['train/valid_code_counts'], probe, probe, [])
    assert summary['dead_code_resets'] == 6
    assert summary['loss_normalization'] == 'global_valid_elements'
    assert summary['mean']['reconstruction_padded_mean'] == 1.
    import train_vq
    previous_argv = sys.argv
    try:
        sys.argv = ['train_vq.py', '--train_manifest', 'unused', '--data_ready_manifest', 'unused',
                    '--vq_dead_code_patience', '0']
        config = train_vq.vq_options(train_vq.get_args())
    finally:
        sys.argv = previous_argv
    assert config['beta'] == 1. and config['decay'] == .99 and config['dead_code_patience'] == 0
    return dict(actual_vq_optimizer_step=True, finite_gradients=True, codebook_not_in_optimizer=True,
                model_mask_reaches_quantizer=True, reconstruction_padding_invariant=True,
                code_indices_masked=True, model_eval_pure=True, fixed_probe_preserves_state=True,
                epoch_metrics_label_loss_scale=True, cli_can_disable_reset=True,
                cpu_loss=float(loss.detach()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--runtime', required=True)
    parser.add_argument('--mode', choices=['baseline', 'unit', 'ddp', 'integration'], required=True)
    parser.add_argument('--output')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.mode == 'ddp':
        dist.init_process_group('gloo')
    if args.mode == 'baseline':
        result = baseline(args.runtime)
    elif args.mode == 'integration':
        result = integration(args.runtime)
    else:
        module = load_quantizer(args.runtime, 'fixed_quantizer')
        result = unit(module) if args.mode == 'unit' else distributed_checks(module)
    result.update(ok=True, mode=args.mode)
    if not dist.is_initialized() or dist.get_rank() == 0:
        if args.output:
            Path(args.output).write_text(json.dumps(result, indent=2) + '\n')
        print('VQ_FIX_TEST=' + json.dumps(result), flush=True)
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
