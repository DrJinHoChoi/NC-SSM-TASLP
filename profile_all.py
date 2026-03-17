"""Complete deployment profiling: MACs, Latency, Energy, RAM.

Usage:
    python profile_all.py
"""
import torch
import torch.nn as nn
import math
import sys
sys.path.insert(0, '.')

from nanomamba import (
    create_nanomamba_nc_matched,
    create_nanomamba_nc_large,
    create_nanomamba_nc_nanose_v3,
    create_nanomamba_nc_nanose,
    create_nanomamba_nc_matched_nanose,
    profile_model,
)
from train_colab import BCResNet, DSCNN_S


# ============================================================
# Hook-based profiler for CNN models
# ============================================================
_total_macs = [0]
_peak_act = [0]


def _count_hook(module, inp, output):
    if isinstance(module, nn.Conv2d):
        out = output.shape
        k = module.kernel_size
        in_c = module.in_channels // module.groups
        macs = out[2] * out[3] * out[1] * (k[0] * k[1] * in_c)
        if module.bias is not None:
            macs += out[1] * out[2] * out[3]
        _total_macs[0] += macs
    elif isinstance(module, nn.Linear):
        macs = module.in_features * module.out_features
        if module.bias is not None:
            macs += module.out_features
        _total_macs[0] += macs
    elif isinstance(module, nn.BatchNorm2d):
        out = output.shape
        _total_macs[0] += out[1] * out[2] * out[3] * 4
    if isinstance(output, torch.Tensor):
        _peak_act[0] = max(_peak_act[0], output.numel())


def profile_cnn(model, name, mel_input):
    model.eval()
    params = sum(p.numel() for p in model.parameters())

    hooks = []
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d)):
            hooks.append(m.register_forward_hook(_count_hook))

    _total_macs[0] = 0
    _peak_act[0] = 0
    with torch.no_grad():
        model(mel_input)

    cnn_macs = _total_macs[0]
    cnn_peak = _peak_act[0]
    for h in hooks:
        h.remove()

    # Preprocessing (STFT + Mel + log) - same front-end as NanoMamba
    n_fft, n_freq, n_mels, T = 512, 257, 40, 97
    stft = T * (n_fft // 2) * int(math.log2(n_fft)) + T * n_fft + T * n_freq * 3
    mel = n_mels * n_freq * T
    log = n_mels * T
    preproc = stft + mel + log

    total = cnn_macs + preproc
    int8_kb = params / 1024
    peak_act_kb = cnn_peak / 1024
    total_ram_kb = int8_kb + peak_act_kb

    clock_hz = 480e6
    power_w = 0.200
    lat_ms = total / clock_hz * 1000
    energy_uj = total / clock_hz * power_w * 1e6

    return dict(
        name=name, params=params,
        total_macs=total, preproc_macs=preproc, model_macs=cnn_macs,
        int8_kb=int8_kb, peak_act_kb=peak_act_kb, total_ram_kb=total_ram_kb,
        latency_ms=lat_ms, energy_uj=energy_uj, avg_power_mw=energy_uj / 1e3,
    )


def main():
    mel_input = torch.randn(1, 40, 97)

    # CNN baselines
    bc = profile_cnn(BCResNet(n_classes=12, scale=1), 'BC-ResNet-1', mel_input)
    ds = profile_cnn(DSCNN_S(n_classes=12), 'DS-CNN-S', mel_input)

    # SSM models
    ssm_cfgs = [
        ('NC-SSM', create_nanomamba_nc_matched),
        ('NC-SSM-Large', create_nanomamba_nc_large),
        ('NC-SSM+NanoSE-v3', create_nanomamba_nc_nanose_v3),
        ('NC-SSM+NanoSE-v1', create_nanomamba_nc_nanose),
    ]
    ssm_results = []
    for name, fn in ssm_cfgs:
        m = fn()
        m.eval()
        r = profile_model(m, verbose=False)
        d = r['deployment']
        mem = r['memory']
        model_keys = ('SSM Blocks', 'Patch Projection', 'Classifier')
        model_macs = sum(v for k, v in r['breakdown'].items() if k in model_keys)
        ssm_results.append(dict(
            name=name, params=mem['params'],
            total_macs=d['macs'], model_macs=model_macs,
            preproc_macs=d['macs'] - model_macs,
            int8_kb=mem['int8_kb'], peak_act_kb=mem['peak_activation_kb'],
            total_ram_kb=mem['total_ram_kb'],
            latency_ms=d['latency_ms'], energy_uj=d['energy_uJ'],
            avg_power_mw=d['avg_power_mW'],
        ))

    all_r = [ds, bc] + ssm_results

    # ================================================================
    # Table 1: MACs
    # ================================================================
    print()
    print('=' * 90)
    print('  COMPLETE DEPLOYMENT PROFILE - Cortex-M7 @ 480 MHz, INT8, 200mW')
    print('=' * 90)
    print()
    print('  [1] MACs Breakdown')
    hdr = f'  {"Model":<22} {"Params":>8} {"Preproc":>9} {"Model":>9} {"Total":>9} {"Total(M)":>9}'
    print(hdr)
    print(f'  {"-" * 70}')
    for r in all_r:
        print(f'  {r["name"]:<22} {r["params"]:>8,} {r["preproc_macs"]:>9,} '
              f'{r["model_macs"]:>9,} {r["total_macs"]:>9,} {r["total_macs"]/1e6:>8.2f}M')

    # ================================================================
    # Table 2: Latency & Energy
    # ================================================================
    print()
    print('  [2] Latency & Energy (Full Pipeline)')
    print(f'  {"Model":<22} {"MACs(M)":>8} {"Lat(ms)":>8} {"E(uJ)":>8} {"P(mW)":>8} {"vs BC":>8}')
    print(f'  {"-" * 56}')
    bc_lat = bc['latency_ms']
    for r in all_r:
        is_bc = (r['name'] == 'BC-ResNet-1')
        ratio = '-' if is_bc else f'{bc_lat / r["latency_ms"]:.1f}x'
        print(f'  {r["name"]:<22} {r["total_macs"]/1e6:>8.2f} {r["latency_ms"]:>8.2f} '
              f'{r["energy_uj"]:>8.0f} {r["avg_power_mw"]:>8.2f} {ratio:>8}')

    # ================================================================
    # Table 3: Memory
    # ================================================================
    print()
    print('  [3] Memory (INT8 Deployment)')
    print(f'  {"Model":<22} {"Weights":>9} {"Peak Act":>9} {"Total RAM":>10} {"vs BC":>8}')
    print(f'  {"-" * 52}')
    bc_ram = bc['total_ram_kb']
    for r in all_r:
        is_bc = (r['name'] == 'BC-ResNet-1')
        ratio = '-' if is_bc else f'{bc_ram / r["total_ram_kb"]:.1f}x'
        print(f'  {r["name"]:<22} {r["int8_kb"]:>8.1f}K {r["peak_act_kb"]:>8.1f}K '
              f'{r["total_ram_kb"]:>9.1f}K {ratio:>8}')

    # ================================================================
    # Paper-ready summary
    # ================================================================
    print()
    print('=' * 90)
    print('  PAPER TABLE VII - Full Pipeline, Cortex-M7 @ 480 MHz')
    print('=' * 90)
    print(f'  {"Model":<22} {"Params":>8} {"MACs":>9} {"Lat.":>9} '
          f'{"Energy":>9} {"RAM":>9}')
    print(f'  {"-" * 60}')
    for r in all_r:
        if r['name'] == 'NC-SSM+NanoSE-v1':
            continue
        print(f'  {r["name"]:<22} {r["params"]:>8,} {r["total_macs"]/1e6:>8.2f}M '
              f'{r["latency_ms"]:>8.1f}ms {r["energy_uj"]:>8.0f}uJ '
              f'{r["total_ram_kb"]:>8.1f}K')

    # ================================================================
    # Key ratios
    # ================================================================
    print()
    print('  * Key ratios vs BC-ResNet-1 (full pipeline):')
    show = ('NC-SSM', 'NC-SSM-Large', 'NC-SSM+NanoSE-v3')
    for r in ssm_results:
        if r['name'] not in show:
            continue
        mac_r = bc['total_macs'] / r['total_macs']
        lat_r = bc['latency_ms'] / r['latency_ms']
        ram_r = bc['total_ram_kb'] / r['total_ram_kb']
        e_r = bc['energy_uj'] / r['energy_uj']
        print(f'    {r["name"]:<22} MACs {mac_r:.1f}x | '
              f'Lat {lat_r:.1f}x | Energy {e_r:.1f}x | RAM {ram_r:.1f}x')

    print()
    print('  * Key ratios vs BC-ResNet-1 (model-only, excl. preprocessing):')
    bc_model = bc['model_macs']
    for r in ssm_results:
        if r['name'] not in show:
            continue
        mac_r = bc_model / r['model_macs']
        print(f'    {r["name"]:<22} Model MACs {mac_r:.1f}x fewer')
    print()


if __name__ == '__main__':
    main()
