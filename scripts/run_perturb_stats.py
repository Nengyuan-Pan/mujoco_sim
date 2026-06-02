"""Tube vs No-Tube 扰动对比 — 多试次统计版本。

用法:
  python scripts/run_perturb_stats.py              # 运行所有测试 + 绘图
  python scripts/run_perturb_stats.py --skip-run   # 只从缓存绘图
  python scripts/run_perturb_stats.py --n-seeds 20 --seed-start 100
"""
import argparse
import subprocess
import sys
import re
import time
from pathlib import Path
from collections import defaultdict

import numpy as np


def run_one(perturb_type: str, value: float, use_tube: bool, seed: int) -> dict:
    """运行单次测试并返回指标字典。"""
    flag = "--time-perturb-ms" if perturb_type == "time" else "--space-perturb-m"
    tube_str = "true" if use_tube else "false"
    cmd = [
        sys.executable, "scripts/rm65_mpc_tube.py",
        "--use_tube", tube_str, "--seed", str(seed),
        "--window-ms", "50", "--no-plot",
        flag, str(value),
    ]
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=300,
        cwd=Path(__file__).resolve().parent.parent,
    )
    stdout = result.stdout
    d = {}
    for pattern, key in [
        (r"最小球拍-球距离:\s+([\d.]+)\s+m", "min_dist"),
        (r"ball_near\s+步数:\s+(\d+)", "ball_near"),
        (r"tube_ready\s+步数:\s+(\d+)", "tube_ready"),
        (r"位置误差:\s+([\d.]+)\s+m", "pos_err"),
        (r"最长连续 tube_ready:\s+(\d+)", "longest_tr"),
        (r"击打后球速:\s+([\d.]+)\s+m/s", "ball_speed"),
    ]:
        m = re.search(pattern, stdout)
        d[key] = float(m.group(1)) if m else np.nan
    d["hit"] = 1.0 if d.get("min_dist", 999) < 0.15 else 0.0
    return d


def run_trials(perturb_type: str, values: list, seeds: list[int]) -> dict:
    """跑完一个扰动类型的所有试次。

    Returns:
        {value: {"no": [dict, ...], "tube": [dict, ...]}}
    """
    total = len(values) * len(seeds) * 2
    results = {v: {"no": [], "tube": []} for v in values}
    count = 0
    t_start = time.perf_counter()
    for v in values:
        for seed in seeds:
            for use_tube in [False, True]:
                tag = "TUBE" if use_tube else "NO"
                key = "tube" if use_tube else "no"
                d = run_one(perturb_type, v, use_tube, seed)
                results[v][key].append(d)
                count += 1
                elapsed = time.perf_counter() - t_start
                eta = elapsed / count * (total - count) if count > 0 else 0
                h = "HIT" if d["hit"] else "MISS"
                print(f"  [{count}/{total}] {perturb_type} v={v:+8.3f} {tag:>4} seed={seed:>3} "
                      f"min_dist={d['min_dist']:.3f} {h:>4}  ETA={eta/60:.0f}min", flush=True)
    return results


def compute_stats(results: dict, key: str = "min_dist") -> dict:
    """计算均值和标准差。"""
    stats = {}
    for v, data in results.items():
        no_arr = np.array([d[key] for d in data["no"]])
        tube_arr = np.array([d[key] for d in data["tube"]])
        stats[v] = {
            "no_mean": float(np.nanmean(no_arr)),
            "no_std": float(np.nanstd(no_arr)),
            "tube_mean": float(np.nanmean(tube_arr)),
            "tube_std": float(np.nanstd(tube_arr)),
            "no_hit_rate": float(np.nanmean([d["hit"] for d in data["no"]])),
            "tube_hit_rate": float(np.nanmean([d["hit"] for d in data["tube"]])),
        }
    return stats


def plot_results(time_stats, space_stats, time_values, space_values,
                 n_seeds, seed_start, out_dir: Path):
    """绘制带误差棒的多试次对比图。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    HIT = 0.15

    # ==== 图 1: 时间偏移 min_dist ====
    ax = axes[0, 0]
    tv = np.array(time_values)
    no_m = [time_stats[v]["no_mean"] for v in time_values]
    no_s = [time_stats[v]["no_std"] for v in time_values]
    tu_m = [time_stats[v]["tube_mean"] for v in time_values]
    tu_s = [time_stats[v]["tube_std"] for v in time_values]
    ax.errorbar(tv, no_m, yerr=no_s, fmt="s--", color="tab:orange",
                capsize=5, capthick=1.5, linewidth=1.5, markersize=7, label="No-Tube")
    ax.errorbar(tv, tu_m, yerr=tu_s, fmt="o-", color="tab:blue",
                capsize=5, capthick=1.5, linewidth=2, markersize=8, label="Tube")
    ax.axhline(y=HIT, color="red", linestyle=":", linewidth=1.5, alpha=0.7, label=f"hit={HIT}m")
    ax.fill_between(tv, 0, HIT, alpha=0.06, color="green")
    ax.set_xlabel("Time perturbation (ms)", fontsize=12)
    ax.set_ylabel("Min racket-ball distance (m)", fontsize=12)
    ax.set_title(f"Time Perturb: min_dist  (N={n_seeds} seeds, err=1 std)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(0.5, ax.get_ylim()[1]))

    # ==== 图 2: 空间偏移 min_dist ====
    ax = axes[0, 1]
    sv = np.array(space_values) * 100
    no_m = [space_stats[v]["no_mean"] for v in space_values]
    no_s = [space_stats[v]["no_std"] for v in space_values]
    tu_m = [space_stats[v]["tube_mean"] for v in space_values]
    tu_s = [space_stats[v]["tube_std"] for v in space_values]
    ax.errorbar(sv, no_m, yerr=no_s, fmt="s--", color="tab:orange",
                capsize=5, capthick=1.5, linewidth=1.5, markersize=7, label="No-Tube")
    ax.errorbar(sv, tu_m, yerr=tu_s, fmt="o-", color="tab:blue",
                capsize=5, capthick=1.5, linewidth=2, markersize=8, label="Tube")
    ax.axhline(y=HIT, color="red", linestyle=":", linewidth=1.5, alpha=0.7, label=f"hit={HIT}m")
    ax.fill_between(sv, 0, HIT, alpha=0.06, color="green")
    ax.set_xlabel("Space offset (cm)", fontsize=12)
    ax.set_ylabel("Min racket-ball distance (m)", fontsize=12)
    ax.set_title(f"Space Perturb: min_dist  (N={n_seeds} seeds, err=1 std)", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(0.25, ax.get_ylim()[1]))

    # ==== 图 3: 命中率 ====
    ax = axes[1, 0]
    width = 0.35
    # 时间
    n_time = len(time_values)
    x_t = np.arange(n_time)
    ax.bar(x_t - width/2, [time_stats[v]["no_hit_rate"]*100 for v in time_values],
           width, color="tab:orange", alpha=0.8, label="No-Tube (time)")
    ax.bar(x_t + width/2, [time_stats[v]["tube_hit_rate"]*100 for v in time_values],
           width, color="tab:blue", alpha=0.8, label="Tube (time)")
    # 空间
    n_space = len(space_values)
    x_s = np.arange(n_time + 1, n_time + 1 + n_space)
    ax.bar(x_s - width/2, [space_stats[v]["no_hit_rate"]*100 for v in space_values],
           width, color="tab:orange", alpha=0.5, hatch="//", label="No-Tube (space)")
    ax.bar(x_s + width/2, [space_stats[v]["tube_hit_rate"]*100 for v in space_values],
           width, color="tab:blue", alpha=0.5, hatch="//", label="Tube (space)")
    ax.set_xticks(list(x_t) + list(x_s))
    labels_t = [f"{v:+d}ms" for v in time_values]
    labels_s = [f"{v*100:+5.0f}cm" for v in space_values]
    ax.set_xticklabels(labels_t + labels_s, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Hit rate (%)", fontsize=12)
    ax.set_title(f"Hit Rate (min_dist < 0.15m, N={n_seeds})", fontsize=13)
    ax.legend(fontsize=7, loc="lower left")
    ax.axhline(y=100, color="green", linestyle=":", alpha=0.3)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.2, axis="y")

    # ==== 图 4: 汇总表 ====
    ax = axes[1, 1]
    ax.axis("off")
    lines = []
    lines.append("SUMMARY  (mean +/- std over N seeds)")
    lines.append("=" * 55)
    lines.append(f"{'Perturb':>10} | {'No-Tube min_d':>18} | {'Tube min_d':>18}")
    lines.append("-" * 55)
    for v in time_values:
        ns = time_stats[v]
        ts = time_stats[v]
        lines.append(f"{v:+8d}ms | {ns['no_mean']:>7.3f}+/-{ns['no_std']:.3f} | {ts['tube_mean']:>7.3f}+/-{ts['tube_std']:.3f}")
    lines.append("-" * 55)
    for v in space_values:
        ns = space_stats[v]
        ts = space_stats[v]
        lines.append(f"{v:+8.3f}m | {ns['no_mean']:>7.3f}+/-{ns['no_std']:.3f} | {ts['tube_mean']:>7.3f}+/-{ts['tube_std']:.3f}")
    text = "\n".join(lines)
    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=7.5,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    out_path = out_dir / f"perturb_stats_N{n_seeds}_s{seed_start}.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"\nChart saved: {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Tube vs No-Tube 多试次扰动对比")
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=100)
    parser.add_argument("--skip-run", action="store_true")
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    time_vals = [-30, -20, -10, 0, 10, 20, 30]
    space_vals = [-0.10, -0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.10]

    results_dir = Path(__file__).resolve().parent.parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    cache = results_dir / f"perturb_stats_N{args.n_seeds}_s{args.seed_start}.npz"

    if not args.skip_run:
        print(f"Time perturbation: {len(time_vals)} values x {args.n_seeds} seeds x 2 modes"
              f" = {len(time_vals)*args.n_seeds*2} runs")
        time_results = run_trials("time", time_vals, seeds)

        print(f"\nSpace perturbation: {len(space_vals)} values x {args.n_seeds} seeds x 2 modes"
              f" = {len(space_vals)*args.n_seeds*2} runs")
        space_results = run_trials("space", space_vals, seeds)

        # 保存缓存 (拆开存避免 pickle 兼容问题)
        time_dict = {}
        for v in time_vals:
            time_dict[f"v{v}_no_min_dist"] = np.array([d["min_dist"] for d in time_results[v]["no"]])
            time_dict[f"v{v}_tube_min_dist"] = np.array([d["min_dist"] for d in time_results[v]["tube"]])
            time_dict[f"v{v}_no_hit"] = np.array([d["hit"] for d in time_results[v]["no"]])
            time_dict[f"v{v}_tube_hit"] = np.array([d["hit"] for d in time_results[v]["tube"]])
        space_dict = {}
        for v in space_vals:
            space_dict[f"v{v}_no_min_dist"] = np.array([d["min_dist"] for d in space_results[v]["no"]])
            space_dict[f"v{v}_tube_min_dist"] = np.array([d["min_dist"] for d in space_results[v]["tube"]])
            space_dict[f"v{v}_no_hit"] = np.array([d["hit"] for d in space_results[v]["no"]])
            space_dict[f"v{v}_tube_hit"] = np.array([d["hit"] for d in space_results[v]["tube"]])
        np.savez(cache, time_vals=np.array(time_vals), space_vals=np.array(space_vals),
                 seeds=np.array(seeds), **time_dict, **space_dict)
    else:
        if not cache.exists():
            print("No cache found, run without --skip-run first")
            return
        data = np.load(cache)
        time_vals = data["time_vals"].tolist()
        space_vals = data["space_vals"].tolist()
        seeds = data["seeds"].tolist()
        time_results = {}
        for v in time_vals:
            time_results[v] = {
                "no": [{"min_dist": x, "hit": h} for x, h in
                       zip(data[f"v{v}_no_min_dist"], data[f"v{v}_no_hit"])],
                "tube": [{"min_dist": x, "hit": h} for x, h in
                         zip(data[f"v{v}_tube_min_dist"], data[f"v{v}_tube_hit"])],
            }
        space_results = {}
        for v in space_vals:
            space_results[v] = {
                "no": [{"min_dist": x, "hit": h} for x, h in
                       zip(data[f"v{v}_no_min_dist"], data[f"v{v}_no_hit"])],
                "tube": [{"min_dist": x, "hit": h} for x, h in
                         zip(data[f"v{v}_tube_min_dist"], data[f"v{v}_tube_hit"])],
            }

    # 计算统计
    time_stats = compute_stats(time_results)
    space_stats = compute_stats(space_results)

    # 绘图
    plot_results(time_stats, space_stats, time_vals, space_vals,
                 args.n_seeds, args.seed_start, results_dir)

    # 打印统计表
    HIT = 0.15
    print("\n" + "=" * 80)
    print(f"  Time Perturbation  (mean +/- 1 std, N={args.n_seeds} seeds)")
    print("=" * 80)
    print(f"{'pert':>7} | {'No-Tube':>21} | {'Tube':>21} | hit rate")
    print(f"{'':>7} | {'min_d':>9} {'+/-':>5} {'hit%':>5} | {'min_d':>9} {'+/-':>5} {'hit%':>5} |")
    print("-" * 70)
    for v in time_vals:
        ns = time_stats[v]
        ts = time_stats[v]
        print(f"{v:+6d}ms | {ns['no_mean']:>7.3f} {ns['no_std']:>5.3f} {ns['no_hit_rate']*100:>4.0f}% | "
              f"{ts['tube_mean']:>7.3f} {ts['tube_std']:>5.3f} {ts['tube_hit_rate']*100:>4.0f}% |")

    print("\n" + "=" * 80)
    print(f"  Space Perturbation  (mean +/- 1 std, N={args.n_seeds} seeds)")
    print("=" * 80)
    print(f"{'offset':>8} | {'No-Tube':>21} | {'Tube':>21} | hit rate")
    print(f"{'':>8} | {'min_d':>9} {'+/-':>5} {'hit%':>5} | {'min_d':>9} {'+/-':>5} {'hit%':>5} |")
    print("-" * 70)
    for v in space_vals:
        ns = space_stats[v]
        ts = space_stats[v]
        print(f"{v:+7.3f}m | {ns['no_mean']:>7.3f} {ns['no_std']:>5.3f} {ns['no_hit_rate']*100:>4.0f}% | "
              f"{ts['tube_mean']:>7.3f} {ts['tube_std']:>5.3f} {ts['tube_hit_rate']*100:>4.0f}% |")

    print(f"\nHit threshold = {HIT:.2f}m")


if __name__ == "__main__":
    main()
