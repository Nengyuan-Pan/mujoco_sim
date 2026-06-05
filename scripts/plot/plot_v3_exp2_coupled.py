"""生成v3实验2耦合扰动的科研格式图表."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] =10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['mathtext.fontset'] = 'cm'

with open(ROOT / "results" / "v3_exp2_coupled_raw.json", "r", encoding="utf-8") as f:
    data = json.load(f)

pcts = [-20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0]
pct_labels = ['-20%', '-15%', '-10%', '-5%', '0%', '+5%', '+10%', '+15%', '+20%']
offsets = [p/100*2.0 for p in pcts]  # 实际偏移距离 (m)

tube_rates = []
notube_rates = []
tube_means = []
notube_means = []
tube_stds = []
notube_stds = []

for pct in pcts:
    for mode, rates, means, stds in [
        ('tube', tube_rates, tube_means, tube_stds),
        ('no_tube', notube_rates, notube_means, notube_stds),
    ]:
        key = f'{mode}/pct={pct}'
        runs = [r['result'] for r in data[key] if r['result'] is not None]
        if runs:
            hits = [r for r in runs if r.get('hit_type') in ('active', 'passive')]
            rates.append(len(hits) / len(runs) * 100)
            errs = [r['pos_error'] * 100 for r in runs]
            means.append(np.mean(errs))
            stds.append(np.std(errs))
        else:
            rates.append(0)
            means.append(0)
            stds.append(0)

# ===== Figure: 成功率折线图 =====
fig, ax1 = plt.subplots(1, 1, figsize=(6, 3.5))

x = np.arange(len(pcts))

ax1.plot(x, tube_rates, 'o-', color='#2ca02c', linewidth=2, markersize=7,
         label='Tube', zorder=3)
ax1.plot(x, notube_rates, 's--', color='#d62728', linewidth=2, markersize=7,
         label='No-Tube', zorder=3)

for i in range(len(pcts)):
    if tube_rates[i] < 100:
        ax1.text(x[i], tube_rates[i] + 2, f'{tube_rates[i]:.0f}%', ha='center',
                 fontsize=8, color='#2ca02c')
    if notube_rates[i] < 100 and abs(tube_rates[i] - notube_rates[i]) > 1:
        ax1.text(x[i], notube_rates[i] - 5, f'{notube_rates[i]:.0f}%', ha='center',
                 fontsize=8, color='#d62728')

ax1.set_xticks(x)
ax1.set_xticklabels(pct_labels, fontsize=9)
ax1.set_xlabel('Launch position offset along flight direction', fontsize=10)
ax1.set_ylabel('Success rate (%)', fontsize=10)
ax1.set_ylim(55, 108)
ax1.set_yticks([60, 70, 80, 90, 100])
ax1.legend(fontsize=9, loc='lower left', framealpha=0.9)
ax1.grid(axis='y', alpha=0.3, zorder=0)
ax1.axhline(y=100, color='gray', linestyle=':', alpha=0.5, zorder=0)

fig.tight_layout()
out1 = ROOT / "results" / "v3_exp2_success_rate.pdf"
fig.savefig(out1, dpi=300, bbox_inches='tight')
fig.savefig(str(out1).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
print(f"成功率图已保存: {out1}")

# ===== Figure: 位置误差柱状图 =====
fig2, ax2 = plt.subplots(1, 1, figsize=(6, 3.5))

width = 0.3
bars1 = ax2.bar(x - width/2, tube_means, width, yerr=tube_stds,
                color='#2ca02c', alpha=0.85, capsize=3, label='Tube', zorder=2)
bars2 = ax2.bar(x + width/2, notube_means, width, yerr=notube_stds,
                color='#d62728', alpha=0.85, capsize=3, label='No-Tube', zorder=2)

ax2.set_xticks(x)
ax2.set_xticklabels(pct_labels, fontsize=10)
ax2.set_xlabel('Launch position offset along flight direction', fontsize=10)
ax2.set_ylabel('Position error (cm)', fontsize=10)
ax2.legend(fontsize=9, loc='upper left', framealpha=0.9)
ax2.grid(axis='y', alpha=0.3, zorder=0)
ax2.set_ylim(0, 16)

fig2.tight_layout()
out2 = ROOT / "results" / "v3_exp2_position_error.pdf"
fig2.savefig(out2, dpi=300, bbox_inches='tight')
fig2.savefig(str(out2).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
print(f"位置误差图已保存: {out2}")

# ===== 打印论文表格 =====
print("\n=== 论文表格 ===")
print("| Offset | Tube Rate | No-Tube Rate | Tube Error (cm) | No-Tube Error (cm) |")
print("|:---:|:---:|:---:|:---:|:---:|")
for i, pct in enumerate(pcts):
    print(f"| {pct_labels[i]} | {tube_rates[i]:.0f}% | {notube_rates[i]:.0f}% "
          f"| {tube_means[i]:.1f} ± {tube_stds[i]:.1f} "
          f"| {notube_means[i]:.1f} ± {notube_stds[i]:.1f} |")
