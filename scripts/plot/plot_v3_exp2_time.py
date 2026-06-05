"""生成v3实验2（纯时间扰动）的论文级图表."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['mathtext.fontset'] = 'cm'

with open(ROOT / "results" / "v3_exp2_time_raw.json", "r", encoding="utf-8") as f:
    data = json.load(f)

time_perturbs = [-300, -200, -100, 0, 100, 200, 300]
t_labels = ['-300', '-200', '-100', '0', '+100', '+200', '+300']

tube_rates, notube_rates = [], []
tube_means, notube_means = [], []
tube_stds, notube_stds = [], []

for t_ms in time_perturbs:
    for mode, rates, means, stds in [
        ('tube', tube_rates, tube_means, tube_stds),
        ('no_tube', notube_rates, notube_means, notube_stds),
    ]:
        key = f'{mode}/t={t_ms}'
        runs = [r['result'] for r in data[key] if r['result'] is not None]
        if runs:
            hits = [r for r in runs if r.get('hit_type') in ('active', 'passive')]
            rates.append(len(hits) / len(runs) * 100)
            errs = [r['pos_error'] * 100 for r in runs]
            means.append(np.mean(errs))
            stds.append(np.std(errs))
        else:
            rates.append(0); means.append(0); stds.append(0)

# ===== Figure: 成功率折线图 =====
fig, ax1 = plt.subplots(1, 1, figsize=(6, 3.8))
x = np.arange(len(time_perturbs))

ax1.plot(x, tube_rates, 'o-', color='#2ca02c', linewidth=2.2, markersize=8,
         label='Tube (Softmin)', zorder=3)
ax1.plot(x, notube_rates, 's--', color='#d62728', linewidth=2.2, markersize=8,
         label='No-Tube', zorder=3)

for i in range(len(time_perturbs)):
    if tube_rates[i] < 100:
        ax1.annotate(f'{tube_rates[i]:.0f}%', (x[i], tube_rates[i]),
                     textcoords="offset points", xytext=(0, 10),
                     ha='center', fontsize=8, color='#2ca02c', fontweight='bold')
    if notube_rates[i] < 100:
        ax1.annotate(f'{notube_rates[i]:.0f}%', (x[i], notube_rates[i]),
                     textcoords="offset points", xytext=(0, -14),
                     ha='center', fontsize=8, color='#d62728', fontweight='bold')

ax1.fill_between(x, tube_rates, notube_rates, where=[t > n for t, n in zip(tube_rates, notube_rates)],
                 alpha=0.15, color='#2ca02c', label='Tube advantage')

ax1.set_xticks(x)
ax1.set_xticklabels(t_labels, fontsize=10)
ax1.set_xlabel(r'Time prediction error $\Delta t$ (ms)', fontsize=11)
ax1.set_ylabel('Success rate (%)', fontsize=11)
ax1.set_ylim(-5, 112)
ax1.legend(fontsize=9, loc='lower left', framealpha=0.9)
ax1.grid(axis='y', alpha=0.3, zorder=0)
ax1.axhline(y=100, color='gray', linestyle=':', alpha=0.4, zorder=0)

fig.tight_layout()
out1 = ROOT / "results" / "v3_exp2_time_success.pdf"
fig.savefig(out1, dpi=300, bbox_inches='tight')
fig.savefig(str(out1).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
print(f"成功率图已保存: {out1}")

# ===== Figure: 位置误差柱状图 =====
fig2, ax2 = plt.subplots(1, 1, figsize=(6, 3.8))
width = 0.3

ax2.bar(x - width/2, tube_means, width, yerr=tube_stds,
        color='#2ca02c', alpha=0.85, capsize=3, label='Tube', zorder=2)
ax2.bar(x + width/2, notube_means, width, yerr=notube_stds,
        color='#d62728', alpha=0.85, capsize=3, label='No-Tube', zorder=2)

ax2.set_xticks(x)
ax2.set_xticklabels(t_labels, fontsize=10)
ax2.set_xlabel(r'Time prediction error $\Delta t$ (ms)', fontsize=11)
ax2.set_ylabel('Position error (cm)', fontsize=11)
ax2.legend(fontsize=9, loc='upper left', framealpha=0.9)
ax2.grid(axis='y', alpha=0.3, zorder=0)

fig2.tight_layout()
out2 = ROOT / "results" / "v3_exp2_time_error.pdf"
fig2.savefig(out2, dpi=300, bbox_inches='tight')
fig2.savefig(str(out2).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
print(f"位置误差图已保存: {out2}")
