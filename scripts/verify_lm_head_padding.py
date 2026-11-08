"""Check the BF16 padded LM-head operation against an FP32 CPU reference."""
import argparse
import json
from pathlib import Path
import torch
import sympy  # Match the algebra libraries loaded by the author optimizer.


def verify():
    assert torch.cuda.device_count() == 2
    torch.manual_seed(42)
    layer_cpu = torch.nn.Linear(768, 58496, bias=False).eval()
    records = []
    for device in (0, 1):
        layer = torch.nn.Linear(768, 58496, bias=False).to(device).eval()
        layer.load_state_dict(layer_cpu.state_dict())
        for batch in (1, 7, 8, 16, 22, 30, 33, 48):
            x_cpu = torch.randn(batch, 1, 768)
            with torch.no_grad():
                reference = layer_cpu(x_cpu)
                x = x_cpu.to(device)
                if batch < 48:
                    x = torch.cat((x, x[:1].expand(48 - batch, -1, -1)), dim=0)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    actual = layer(x)[:batch]
                torch.cuda.synchronize(device)
                actual = actual.float().cpu()
                assert actual.shape == reference.shape and torch.isfinite(actual).all()
                torch.testing.assert_close(actual, reference, rtol=0.03, atol=0.02)
                records.append({'gpu': device, 'batch': batch, 'shape': list(actual.shape),
                                'max_absolute_error_vs_fp32': (actual - reference).abs().max().item()})
        del layer
        torch.cuda.empty_cache()
    return {'status': 'passed', 'diagnostic_only': True, 'torch': torch.__version__,
            'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0),
            'padding_batch': 48, 'reference': 'FP32 CPU linear', 'rtol': 0.03, 'atol': 0.02,
            'cases': records}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = verify()
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'cases': len(report['cases'])}))
