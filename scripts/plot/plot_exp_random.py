"""生成实验A和实验B的论文级图表."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DATE = "20260602"
OUT_DIR = ROOT / "results" / f"exp_random_robustness_{DATE}"

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['mathtext.fontset'] = 'cm'

# ===== 实验A: 散点图 =====
with open(OUT_DIR / "expA_raw.json", "r", encoding="utf-8") as f:
    data_a = json.load(f)

perturb = np.load(OUT_DIR / "perturbations.npz")
t_all = perturb['t_perturbs']
s_all = perturb['s_perturbs'] * 100  # cm

tube_hit = []
notube_hit = []
tube_pos = []
notube_pos = []
t_plot = []
s_plot = []

for i in range(len(t_all)):
    for mode, hit_list, pos_list in [
        ('tube', tube_hit, tube_pos),
        ('no_tube', notube_hit, notube_pos),
    ]:
        r = data_a[mode][i]['result']
        if r is not None:
            is_hit = r.get('hit_type', '') in ('active', 'passive')
            hit_list.append(1 if is_hit else 0)
            pos_list.append(r['pos_error'] * 100)
        else:
            hit_list.append(0)
            pos_list.append(-1)

# 散点图: 时间扰动 vs 成功/失败
fig, axes = plt.subplots(1, 2, figsize=(10, 4))

for ax, mode, hits, label in [
    (axes[0], 'tube', tube_hit, 'Tube'),
    (axes[1], 'no_tube', notube_hit, 'No-Tube'),
]:
    hits_arr = np.array(hits)
    t_arr = t_all[:len(hits)]
    s_arr = s_all[:len(hits)]
    
    hit_mask = hits_arr == 1
    miss_mask = hits_arr == 0
    
    ax.scatter(t_arr[hit_mask], s_arr[hit_mask], c='#2ca02c', s=60, alpha=0.7,
               label=f'Hit ({hit_mask.sum()})', edgecolors='white', linewidth=0.5, zorder=3)
    ax.scatter(t_arr[miss_mask], s_arr[miss_mask], c='#d62728', s=60, alpha=0.7,
               marker='x', label=f'Miss ({miss_mask.sum()})', zorder=3)
    
    ax.set_xlabel(r'Time prediction error $\Delta t$ (ms)', fontsize=10)
    ax.set_ylabel(r'Space prediction error $\Delta s$ (cm)', fontsize=10)
    ax.set_title(f'{label}', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='lower left', framealpha=0.9)
    ax.set_xlim(-350, 350)
    ax.set_ylim(-20, 20)
    ax.axhline(0, color='gray', linestyle=':', alpha=0.3)
    ax.axvline(0, color='gray', linestyle=':', alpha=0.3)
    ax.grid(alpha=0.2)

fig.tight_layout(w_pad=2)
fig.savefig(OUT_DIR / "expA_scatter.pdf", dpi=300, bbox_inches='tight')
fig.savefig(OUT_DIR / "expA_scatter.png", dpi=300, bbox_inches='tight')
print(f"散点图已保存")

# 实验A 统计摘要
fig2, ax2 = plt.subplots(1, 1, figsize=(3, 3))
tube_rate = sum(tube_hit) / len(tube_hit) * 100
notube_rate = sum(notube_hit) / len(notube_hit) * 100
bars = ax2.bar(['Tube', 'No-Tube'], [tube_rate, notube_rate],
               color=['#2ca02c', '#d62728'], alpha=0.85, width=0.5, zorder=2)
for bar, rate in zip(bars, [tube_rate, notube_rate]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f'{rate:.0f}%', ha='center', fontsize=11, fontweight='bold')
ax2.set_ylabel('Success rate (%)', fontsize=10)
ax2.set_ylim(0, 115)
ax2.grid(axis='y', alpha=0.3, zorder=0)
fig2.tight_layout()
fig2.savefig(OUT_DIR / "expA_bar.pdf", dpi=300, bbox_inches='tight')
fig2.savefig(OUT_DIR / "expA_bar.png", dpi=300, bbox_inches='tight')
print(f"柱状图已保存")

# ===== 实验B: 衰减因子扫描折线图 =====
with open(OUT_DIR / "expB_raw.json", "r", encoding="utf-8") as f:
    data_b = json.load(f)

alphas = [round(a * 0.1, 1) for a in range(0, 11)]
tube_rates_b = []
notube_rates_b = []
tube_errs_b = []
notube_errs_b = []

for alpha in alphas:
    for mode, rates, errs in [
        ('tube', tube_rates_b, tube_errs_b),
        ('no_tube', notube_rates_b, notube_errs_b),
    ]:
        key = f'{mode}/alpha={alpha}'
        runs = [r for r in data_b[key] if r['result'] is not None]
        hits = [r for r in runs if r['result'].get('hit_type') in ('active', 'passive')]
        rates.append(len(hits) / len(runs) * 100 if runs else 0)
        pe = [r['result']['pos_error'] * 100 for r in runs]
        errs.append(np.mean(pe) if pe else 0)

fig3, ax3 = plt.subplots(1, 1, figsize=(6, 3.8))
ax3.plot(alphas, tube_rates_b, 'o-', color='#2ca02c', linewidth=2.2, markersize=8,
         label='Tube', zorder=3)
ax3.plot(alphas, notube_rates_b, 's--', color='#d62728', linewidth=2.2, markersize=8,
         label='No-Tube', zorder=3)
ax3.fill_between(alphas, tube_rates_b, notube_rates_b,
                 where=[t > n for t, n in zip(tube_rates_b, notube_rates_b)],
                 alpha=0.15, color='#2ca02c', label='Tube advantage')
ax3.set_xlabel(r'Perturbation retention $\alpha_{\min}$', fontsize=11)
ax3.set_ylabel('Success rate (%)', fontsize=11)
ax3.set_ylim(20, 110)
ax3.legend(fontsize=9, loc='lower left', framealpha=0.9)
ax3.grid(axis='y', alpha=0.3, zorder=0)
ax3.set_xticks(alphas)
ax3.set_xticklabels([f'{a:.1f}' for a in alphas], fontsize=9)
fig3.tight_layout()
fig3.savefig(OUT_DIR / "expB_alpha_sweep.pdf", dpi=300, bbox_inches='tight')
fig3.savefig(OUT_DIR / "expB_alpha_sweep.png", dpi=300, bbox_inches='tight')
print(f"衰减因子扫描图已保存")

# ===== 打印论文表格 =====
print("\n=== 实验A 论文表格 ===")
print(f"| Metric | Tube | No-Tube | Difference |")
print(f"|:---:|:---:|:---:|:---:|")
print(f"| Success rate | {sum(tube_hit)}/{len(tube_hit)} ({tube_rate:.0f}%) "
      f"| {sum(notube_hit)}/{len(notube_hit)} ({notube_rate:.0f}%) "
      f"| {tube_rate - notube_rate:+.0f}pp |")
tube_pos_valid = [p for p in tube_pos if p > 0]
notube_pos_valid = [p for p in notube_pos if p > 0]
print(f"| Pos. error (cm) | {np.mean(tube_pos_valid):.1f}±{np.std(tube_pos_valid):.1f} "
      f"| {np.mean(notube_pos_valid):.1f}±{np.std(notube_pos_valid):.1f} "
      f"| -{np.mean(notube_pos_valid) - np.mean(tube_pos_valid):.1f} |")

print("\n=== 实验B 论文表格 ===")
print(f"| {'α_min':>5} | {'Tube':>6} | {'No-Tube':>7} | {'Δ':>5} |")
print(f"|:---:|:---:|:---:|:---:|")
for i, alpha in enumerate(alphas):
    diff = tube_rates_b[i] - notube_rates_b[i]
    print(f"| {alpha:.1f} | {tube_rates_b[i]:.0f}% | {notube_rates_b[i]:.0f}% | {diff:+.0f}pp |")
