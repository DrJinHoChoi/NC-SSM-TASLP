#!/usr/bin/env python3
"""
Var[y] vs σ² experiment — Empirical verification of O(σ²) vs O(σ⁶) noise amplification.

Theory (Proposition 1):
  - LTI SSM: Var[y_noise] = ||h||² · σ²  → O(σ²)
  - Selective SSM: Var[y_noise] ∝ σ⁶      → O(σ⁶) due to input-dependent params

Experiment:
  1. Feed PURE white noise at various σ² levels through trained models
  2. Measure Var[output]
  3. Log-log plot: slope=1 → O(σ²), slope=3 → O(σ⁶)

Usage:
  python var_y_experiment.py --checkpoint_dir /path/to/checkpoints
  python var_y_experiment.py --checkpoint_dir /content/drive/MyDrive/NC-SSM-checkpoints
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path

def measure_var_y(model, sigma_values, device, n_samples=200, seq_len=16000):
    """Feed pure noise at various σ levels, measure output variance."""
    model.eval()
    results = {}

    for sigma in sigma_values:
        output_vars = []
        with torch.no_grad():
            for _ in range(n_samples // 10):  # batch of 10
                noise = torch.randn(10, seq_len, device=device) * sigma
                out = model(noise)  # [B, n_classes]
                # Variance of pre-softmax logits
                output_vars.append(out.var(dim=0).mean().item())

        avg_var = np.mean(output_vars)
        sigma_sq = sigma ** 2
        results[float(sigma)] = {
            'sigma': float(sigma),
            'sigma_sq': float(sigma_sq),
            'log_sigma_sq': float(np.log10(sigma_sq)) if sigma_sq > 0 else -10,
            'var_y': float(avg_var),
            'log_var_y': float(np.log10(avg_var)) if avg_var > 0 else -10,
        }
        print(f"    σ={sigma:.4f}, σ²={sigma_sq:.6f}, Var[y]={avg_var:.4f}")

    return results


def compute_slope(results):
    """Fit log(Var[y]) = slope * log(σ²) + intercept via least squares."""
    points = sorted(results.values(), key=lambda x: x['sigma_sq'])
    # Use only points where both values are valid
    valid = [(p['log_sigma_sq'], p['log_var_y']) for p in points
             if p['log_sigma_sq'] > -9 and p['log_var_y'] > -9]
    if len(valid) < 2:
        return None, None

    x = np.array([v[0] for v in valid])
    y = np.array([v[1] for v in valid])

    # Linear regression
    A = np.vstack([x, np.ones(len(x))]).T
    slope, intercept = np.linalg.lstsq(A, y, rcond=None)[0]
    return slope, intercept


def main():
    parser = argparse.ArgumentParser(description='Var[y] vs σ² experiment')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                        help='Directory containing model checkpoints')
    parser.add_argument('--output', type=str, default='results/var_y_results.json',
                        help='Output JSON file')
    parser.add_argument('--n_samples', type=int, default=200,
                        help='Number of noise samples per σ level')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Import model factories
    from nanomamba import (
        create_nanomamba_nc_matched,
        create_nanomamba_nc_large,
        create_nanomamba_nc_12k,
        create_nanomamba_nc_15k,
        create_nanomamba_nc_20k,
    )

    models_config = [
        ('NanoMamba-NC-20K', create_nanomamba_nc_20k),
        ('NanoMamba-NC-15K', create_nanomamba_nc_15k),
        ('NanoMamba-NC-12K', create_nanomamba_nc_12k),
        ('NanoMamba-NC-Large', create_nanomamba_nc_large),
        ('NanoMamba-NC', create_nanomamba_nc_matched),
    ]

    # σ values corresponding to various noise levels
    # SNR(dB) = 10*log10(P_signal/σ²), assuming P_signal=1
    # σ² = 10^(-SNR/10)
    snr_levels = [-15, -10, -5, 0, 5, 10, 15]
    sigma_values = [np.sqrt(10 ** (-snr / 10)) for snr in snr_levels]

    print("=" * 70)
    print("  Var[y] vs σ² — Noise Amplification Order Verification")
    print("  Theory: slope=1 → O(σ²) [LTI], slope=3 → O(σ⁶) [selective]")
    print("=" * 70)
    print(f"\n  SNR levels: {snr_levels}")
    print(f"  σ values: {[f'{s:.4f}' for s in sigma_values]}")
    print(f"  σ² values: {[f'{s**2:.4f}' for s in sigma_values]}")

    all_results = {}
    ckpt_dir = Path(args.checkpoint_dir)

    for model_name, factory_fn in models_config:
        # Try multiple checkpoint locations
        ckpt_paths = [
            ckpt_dir / model_name / 'best.pt',
            ckpt_dir / model_name.replace(' ', '_') / 'best.pt',
            Path('checkpoints') / model_name / 'best.pt',
        ]

        ckpt_path = None
        for p in ckpt_paths:
            if p.exists():
                ckpt_path = p
                break

        if ckpt_path is None:
            print(f"\n  [SKIP] {model_name}: no checkpoint found")
            continue

        print(f"\n  {'=' * 60}")
        print(f"  {model_name} (checkpoint: {ckpt_path})")
        print(f"  {'=' * 60}")

        # Load model
        model = factory_fn().to(device)
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        model.eval()

        params = sum(p.numel() for p in model.parameters())
        print(f"  Params: {params:,}")

        # Measure Var[y]
        results = measure_var_y(model, sigma_values, device, args.n_samples)

        # Compute slope
        slope, intercept = compute_slope(results)
        if slope is not None:
            print(f"\n  📐 Log-log slope: {slope:.3f}")
            if slope < 1.5:
                print(f"     → O(σ^{2*slope:.1f}) ≈ O(σ²) — LTI behavior ✅")
            elif slope < 2.5:
                print(f"     → O(σ^{2*slope:.1f}) — intermediate coupling ⚠️")
            else:
                print(f"     → O(σ^{2*slope:.1f}) ≈ O(σ⁶) — selective coupling ❌")

        all_results[model_name] = {
            'params': params,
            'slope': float(slope) if slope is not None else None,
            'intercept': float(intercept) if intercept is not None else None,
            'measurements': results,
            'snr_to_sigma': {str(snr): float(sigma) for snr, sigma in zip(snr_levels, sigma_values)},
        }

        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {output_path}")

    # Print summary table
    print(f"\n  {'=' * 60}")
    print(f"  SUMMARY: Noise Amplification Order")
    print(f"  {'=' * 60}")
    print(f"  {'Model':<22} {'Params':>8} {'Slope':>8} {'Order':>10}")
    print(f"  {'-' * 52}")
    for name, r in all_results.items():
        slope = r['slope']
        if slope is not None:
            order = f"O(σ^{2*slope:.1f})"
            print(f"  {name:<22} {r['params']:>8,} {slope:>8.3f} {order:>10}")
        else:
            print(f"  {name:<22} {r['params']:>8,} {'N/A':>8} {'N/A':>10}")

    print(f"\n  Theory prediction:")
    print(f"    slope ≈ 1.0 → O(σ²) — noise-controlled (LTI fallback)")
    print(f"    slope ≈ 3.0 → O(σ⁶) — selectivity leaking noise")
    print(f"    SS should reduce slope back toward 1.0")


if __name__ == '__main__':
    main()
