"""Verify NC-SSM scaling model parameters and deployment metrics."""
import sys
sys.path.insert(0, '.')
from nanomamba import (
    create_nanomamba_nc_matched,
    create_nanomamba_nc_large,
    create_nanomamba_nc_12k,
    create_nanomamba_nc_15k,
    create_nanomamba_nc_20k,
    profile_model,
)

models = [
    ('NC-SSM-20K',  create_nanomamba_nc_20k),
    ('NC-SSM-15K',  create_nanomamba_nc_15k),
    ('NC-SSM-12K',  create_nanomamba_nc_12k),
    ('NC-SSM-Large', create_nanomamba_nc_large),
    ('NC-SSM',      create_nanomamba_nc_matched),
]

# BC-ResNet-1 reference (from profile_all.py code-verified)
bc_full = 6.15e6
bc_latency = 12.8
bc_energy = 2562
bc_ram = 30.0

print("=" * 85)
print("  NC-SSM Parameter Scaling: Code-Verified Deployment Metrics")
print("  Target: Cortex-M7 @ 480MHz, INT8, 200mW active")
print("=" * 85)

results = []
for name, factory in models:
    m = factory()
    p = profile_model(m, verbose=False)
    params = p['memory']['params']
    d_model = m.classifier.in_features
    d_state = m.blocks[0].sa_ssm.d_state
    d_inner = m.blocks[0].d_inner

    full_macs = p['deployment']['macs_M']
    # Model-only = full - preprocessing (STFT + SNR + Mel + LSG + DualPCEN + InstNorm)
    preproc_keys = ['STFT', 'SNR Estimation', 'Mel Filterbank',
                    'Learned Spectral Gate', 'DualPCEN v2', 'Instance Norm']
    preproc_macs = sum(p['breakdown'].get(k, 0) for k in preproc_keys) / 1e6
    model_macs = full_macs - preproc_macs

    latency = p['deployment']['latency_ms']
    energy = p['deployment']['energy_uJ']
    ram = p['deployment']['total_ram_kb']

    results.append({
        'name': name, 'params': params,
        'd': d_model, 'N': d_state, 'di': d_inner,
        'full': full_macs, 'model': model_macs, 'preproc': preproc_macs,
        'lat': latency, 'energy': energy, 'ram': ram,
    })

# Print comparison table
print(f"\n  {'Model':<16} {'Params':>8} {'d':>3} {'N':>3} {'d_i':>4}"
      f"  {'Full MACs':>10} {'Model':>8} {'Lat(ms)':>8} {'Energy':>8} {'RAM(K)':>7}"
      f"  {'vs BC':>6}")
print(f"  {'-'*100}")

# BC-ResNet-1 reference
print(f"  {'BC-ResNet-1':<16} {'7,464':>8} {'-':>3} {'-':>3} {'-':>4}"
      f"  {'6.15M':>10} {'4.99M':>8} {'12.8':>8} {'2,562':>8} {'30.0':>7}"
      f"  {'1.0x':>6}")
print(f"  {'-'*100}")

for r in results:
    ratio = bc_full / (r['full'] * 1e6)
    print(f"  {r['name']:<16} {r['params']:>8,} {r['d']:>3} {r['N']:>3} {r['di']:>4}"
          f"  {r['full']:>9.2f}M {r['model']:>7.2f}M {r['lat']:>8.1f} {r['energy']:>8.0f} {r['ram']:>7.1f}"
          f"  {ratio:>5.1f}x")

# Preprocessing breakdown (should be same for all)
print(f"\n  Shared preprocessing: {results[0]['preproc']:.2f}M MACs (same for all NC-SSM models)")

# Duty cycle power comparison
print(f"\n  --- 1-second Duty Cycle Average Power ---")
print(f"  {'Model':<16} {'Active':>8} {'Sleep':>8} {'Avg Power':>10}")
print(f"  {'-'*50}")
print(f"  {'BC-ResNet-1':<16} {'12.8ms':>8} {'987ms':>8} {'2.56mW':>10}")
for r in results:
    sleep = 1000 - r['lat']
    avg_mw = r['energy'] / 1000  # uJ per second = mW
    print(f"  {r['name']:<16} {r['lat']:>7.1f}ms {sleep:>7.1f}ms {avg_mw:>9.2f}mW")
