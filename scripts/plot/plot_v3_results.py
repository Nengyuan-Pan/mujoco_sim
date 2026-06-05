"""生成v3实验的科研格式图表: 热力图 + 对比图."""
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

# 加载数据
with open(ROOT / "results" / "v3_experiment_raw.json", "r", encoding="utf-8") as f:
    data = json.load(f)

time_perturbs = [-200, -100, 0, 100, 200]
space_perturbs = [-20, -10, 0, 10, 20]

# 构建矩阵
tube_rate = np.zeros((5, 5))
notube_rate = np.zeros((5, 5))
tube_err = np.zeros((5, 5))
notube_err = np.zeros((5, 5))

for i, t_ms in enumerate(time_perturbs):
    for j, s_cm in enumerate(space_perturbs):
        s_m = s_cm / 100.0
        for mi, (mode, mat_rate, mat_err) in enumerate([
            ('tube', tube_rate, tube_err),
            ('no_tube', notube_rate, notube_err),
        ]):
            key = f'{mode}/t={float(t_ms)}/s={s_m}'
            runs = data['exp2'].get(key, [])
            results = [r['result'] for r in runs if r['result'] is not None]
            if results:
                hits = [r for r in results if r.get('hit_type') in ('active', 'passive')]
                mat_rate[i, j] = len(hits) / len(results) * 100
                mat_err[i, j] = np.mean([r['pos_error'] for r in results]) * 100  # cm
            else:
                mat_rate[i, j] = 0
                mat_err[i, j] = 0

diff_rate = tube_rate - notube_rate

# ===== Figure: 三图并排，手动 GridSpec 保证等大 =====
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig = plt.figure(figsize=(14, 3.8))
# 9列: [矩阵0][cb0][间隔][矩阵1][cb1][间隔][矩阵2][cb2]
gs = GridSpec(1, 9, width_ratios=[1, 0.02, 0.2, 1, 0.02, 0.2, 1, 0.02, 0.01],
              wspace=0.0, left=0.05, right=0.97)

cmap_rate = matplotlib.colors.LinearSegmentedColormap.from_list(
    'rate', ['#d62728', '#ffbb78', '#98df8a', '#2ca02c'], N=256)

t_labels = [f'{t:+d}' for t in time_perturbs]
s_labels = [f'{s:+d}' for s in space_perturbs]

def draw_heatmap(ax, mat, title, cmap, vmin, vmax, show_ylabel=True):
    im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, aspect='equal')
    ax.set_xticks(range(5))
    ax.set_xticklabels(s_labels, fontsize=9)
    ax.set_yticks(range(5))
    if show_ylabel:
        ax.set_yticklabels(t_labels, fontsize=9)
        ax.set_ylabel(r'$\Delta t$ (ms)', fontsize=10)
    else:
        ax.set_yticklabels([])
    ax.set_xlabel(r'$\Delta s$ (cm)', fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    for ii in range(5):
        for jj in range(5):
            v = mat[ii, jj]
            if cmap == cmap_rate:
                label = f'{v:.0f}'
                color = 'white' if v < 40 or v > 85 else 'black'
            else:
                if abs(v) < 0.5:
                    label = '0'
                else:
                    label = f'{v:+.0f}'
                color = 'white' if abs(v) > 25 else 'black'
            ax.text(jj, ii, label, ha='center', va='center', fontsize=9, color=color)
    return im

ax0 = fig.add_subplot(gs[0, 0])
im0 = draw_heatmap(ax0, tube_rate, 'Tube', cmap_rate, 0, 100, show_ylabel=True)
cax0 = fig.add_subplot(gs[0, 1])
cb0 = fig.colorbar(im0, cax=cax0)
cb0.set_label('Success rate (%)', fontsize=9)
cb0.ax.tick_params(labelsize=8)

ax1 = fig.add_subplot(gs[0, 3])
im1 = draw_heatmap(ax1, notube_rate, 'No-Tube', cmap_rate, 0, 100, show_ylabel=True)
cax1 = fig.add_subplot(gs[0, 4])
cb1 = fig.colorbar(im1, cax=cax1)
cb1.set_label('Success rate (%)', fontsize=9)
cb1.ax.tick_params(labelsize=8)

cmap_diff = matplotlib.colors.LinearSegmentedColormap.from_list(
    'diff', ['#2166ac', '#67a9cf', '#f7f7f7', '#ef8a62', '#b2182b'], N=256)
max_abs = max(abs(diff_rate.min()), abs(diff_rate.max()), 1)
ax2_hm = fig.add_subplot(gs[0, 6])
im2 = draw_heatmap(ax2_hm, diff_rate, 'Tube $-$ No-Tube', cmap_diff, -max_abs, max_abs, show_ylabel=True)
cax2 = fig.add_subplot(gs[0, 7])
cb2 = fig.colorbar(im2, cax=cax2)
cb2.set_label(r'$\Delta$ Success rate (pp)', fontsize=9)
cb2.ax.tick_params(labelsize=8)

out_fig = ROOT / "results" / "v3_perturbation_heatmap.pdf"
fig.savefig(out_fig, dpi=300, bbox_inches='tight')
fig.savefig(str(out_fig).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
print(f"热力图已保存: {out_fig}")

# ===== Figure 2: 实验1 柱状图 =====
# 从最新rerun结果加载
try:
    with open(ROOT / "results" / "v3_exp1_raw.json", "r", encoding="utf-8") as f:
        exp1 = json.load(f)
except FileNotFoundError:
    exp1 = data.get('exp1', {})

fig2, ax2 = plt.subplots(1, 1, figsize=(5, 3.6))
speeds_plot = [5, 6, 7, 8]
distances = [5.7, 6.8, 8.0, 9.5]
rates = []
means = []
stds = []
for sp in speeds_plot:
    runs = [r['result'] for r in exp1.get(str(sp), []) if r['result'] is not None]
    if runs:
        hits = [r for r in runs if r.get('hit_type') in ('active', 'passive')]
        rates.append(len(hits) / len(runs) * 100)
        errs = [r['pos_error'] * 100 for r in runs]  # cm
        means.append(np.mean(errs))
        stds.append(np.std(errs))
    else:
        rates.append(0)
        means.append(0)
        stds.append(0)

x = np.arange(len(speeds_plot))
width = 0.35

bars = ax2.bar(x - width/2, rates, width, color='#2ca02c', alpha=0.85, label='Success rate (%)', zorder=2)
for bar, rate in zip(bars, rates):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f'{rate:.0f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax2_err = ax2.twinx()
ax2_err.bar(x + width/2, means, width, yerr=stds, color='#1f77b4', alpha=0.85,
            capsize=3, label='Position error (cm)', zorder=2)

ax2.set_xlabel('Ball speed (m/s)', fontsize=10)
ax2.set_ylabel('Success rate (%)', fontsize=10, color='#2ca02c')
ax2_err.set_ylabel('Position error (cm)', fontsize=10, color='#1f77b4')
ax2.set_xticks(x)
ax2.set_xticklabels([f'{s} m/s\n(d={d}m)' for s, d in zip(speeds_plot, distances)], fontsize=9)
ax2.set_ylim(0, 115)
ax2_err.set_ylim(0, 20)

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_err.get_legend_handles_labels()
fig2.legend(lines1 + lines2, labels1 + labels2, loc='lower center', ncol=2,
            fontsize=8, framealpha=0.9, handlelength=1.5, borderpad=0.3,
            bbox_to_anchor=(0.5, -0.02))
ax2.grid(axis='y', alpha=0.3, zorder=0)

fig2.subplots_adjust(bottom=0.2, right=0.88)
out_fig2 = ROOT / "results" / "v3_ball_speed_bar.pdf"
fig2.savefig(out_fig2, dpi=300, bbox_inches='tight')
fig2.savefig(str(out_fig2).replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
print(f"柱状图已保存: {out_fig2}")
