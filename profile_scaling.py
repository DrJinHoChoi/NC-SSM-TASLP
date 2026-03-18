"""Profile NC-SSM scaling models for paper Table VII."""
import torch, sys
sys.path.insert(0, '.')
from nanomamba import (
    create_nanomamba_nc_matched, create_nanomamba_nc_large,
    create_nanomamba_nc_12k, create_nanomamba_nc_15k, create_nanomamba_nc_20k,
    profile_model
)

models = [
    ('NC-SSM-20K', create_nanomamba_nc_20k),
    ('NC-SSM-15K', create_nanomamba_nc_15k),
    ('NC-SSM-12K', create_nanomamba_nc_12k),
    ('NC-SSM-Large', create_nanomamba_nc_large),
    ('NC-SSM', create_nanomamba_nc_matched),
]

print(f"{'Model':<18} {'Params':>8} {'Full(M)':>8} {'Model(M)':>9} {'Lat(ms)':>8} {'Energy':>8} {'RAM(KB)':>8}")
print("-" * 75)

for name, fn in models:
    m = fn()
    p = sum(x.numel() for x in m.parameters())
    r = profile_model(m, verbose=False)
    total = r['total_macs']
    # Calculate model-only MACs from breakdown
    pre = 0
    bd = r['breakdown']
    for k in ['stft', 'snr_estimator', 'mel_filterbank', 'dual_pcen', 'lsg', 'nano_se', 'instance_norm']:
        if k in bd:
            pre += bd[k]
    model_only = total - pre
    d = r['deployment']
    lat = total / 480e6 * 1000  # ms at 480MHz
    energy = lat * 200  # uJ at 200mW
    ram = d.get('ram_kb', 0)
    print(f"{name:<18} {p:>8,} {total/1e6:>8.2f} {model_only/1e6:>9.2f} {lat:>8.1f} {energy:>8.0f} {ram:>8.1f}")
