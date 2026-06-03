"""时间偏移 + 空间偏移 综合可视化脚本。

用法:
  python scripts/plot_perturb_results.py
  python scripts/plot_perturb_results.py --seed 42
"""
import argparse
import subprocess
import sys
import re
from pathlib import Path

import numpy as np

def run_one(perturb_type: str, value: float, use_tube: bool, seed: int) -> dict:
    flag = "--time-perturb-ms" if perturb_type == "time" else "--space-perturb-m"
    tube_str = "true" if use_tube else "false"
    cmd = [
        sys.executable, "scripts/rm65_mpc_tube.py",
        "--use_tube", tube_str, "--seed", str(seed),
        "--window-ms", "50", "--no-plot",
        flag, str(value),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                            cwd=Path(__file__).resolve().parent.parent.parent)
    stdout = result.stdout
    d = {}
    for pattern, key in [
        (r"最小球拍-球距离:\s+([\d.]+)\s+m", "min_dist"),
        (r"ball_near\s+步数:\s+(\d+)", "ball_near"),
        (r"tube_ready\s+步数:\s+(\d+)", "tube_ready"),
        (r"位置误差:\s+([\d.]+)\s+m", "pos_err"),
        (r"击打后球速:\s+([\d.]+)\s+m/s", "ball_speed"),
        (r"最长连续 tube_ready:\s+(\d+)", "longest_tr"),
    ]:
        m = re.search(pattern, stdout)
        d[key] = float(m.group(1)) if m else np.nan
    d["hit"] = 1.0 if d.get("min_dist", 999) < 0.15 else 0.0
    return d


def main() -> None:
    parser = argparse.ArgumentParser(description="Tube 扰动测试可视化")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-run", action="store_true",
                        help="跳过运行，直接绘图（需已有 .npz 缓存）")
    args = parser.parse_args()

    cache_path = Path(__file__).resolve().parent.parent.parent / "results" / f"perturb_cache_s{args.seed}.npz"

    # ── 运行测试 ──
    if not args.skip_run or not cache_path.exists():
        print("运行时间偏移测试...")
        time_vals = [-30, -20, -10, 0, 10, 20, 30]
        time_no  = {v: run_one("time", v, False, args.seed) for v in time_vals}
        time_tube = {v: run_one("time", v, True, args.seed) for v in time_vals}

        print("运行空间偏移测试...")
        space_vals = [-0.10, -0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.10]
        space_no  = {v: run_one("space", v, False, args.seed) for v in space_vals}
        space_tube = {v: run_one("space", v, True, args.seed) for v in space_vals}

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path,
                 time_vals=np.array(time_vals), space_vals=np.array(space_vals),
                 time_no=time_no, time_tube=time_tube,
                 space_no=space_no, space_tube=space_tube)
    else:
        print(f"从缓存加载: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        time_vals = data["time_vals"].tolist()
        space_vals = data["space_vals"].tolist()
        time_no = data["time_no"].item()
        time_tube = data["time_tube"].item()
        space_no = data["space_no"].item()
        space_tube = data["space_tube"].item()

    # ── 绘图 ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib 未安装")
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f"Tube vs No-Tube 扰动对比 (seed={args.seed})", fontsize=16, fontweight="bold")

    HIT_LINE = 0.15  # 命中阈值 (racket_radius + ball_radius)

    # ===== 图 1: 时间偏移 — min_dist =====
    ax = axes[0, 0]
    tv = np.array(time_vals)
    ax.plot(tv, [time_no[v]["min_dist"] for v in time_vals],
            "s--", color="tab:orange", linewidth=1.8, markersize=8, label="No-Tube")
    ax.plot(tv, [time_tube[v]["min_dist"] for v in time_vals],
            "o-", color="tab:blue", linewidth=2.2, markersize=9, label="Tube")
    ax.axhline(y=HIT_LINE, color="red", linestyle=":", linewidth=1.5, alpha=0.7, label=f"Hit threshold ({HIT_LINE:.2f}m)")
    ax.fill_between(tv, 0, HIT_LINE, alpha=0.06, color="green")
    ax.set_xlabel("Time perturbation (ms)", fontsize=12)
    ax.set_ylabel("Min racket-ball distance (m)", fontsize=12)
    ax.set_title("Time Perturbation: min_dist", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(0.45, ax.get_ylim()[1]))

    # ===== 图 2: 空间偏移 — min_dist =====
    ax = axes[0, 1]
    sv = np.array(space_vals) * 100  # 转 cm
    ax.plot(sv, [space_no[v]["min_dist"] for v in space_vals],
            "s--", color="tab:orange", linewidth=1.8, markersize=8, label="No-Tube")
    ax.plot(sv, [space_tube[v]["min_dist"] for v in space_vals],
            "o-", color="tab:blue", linewidth=2.2, markersize=9, label="Tube")
    ax.axhline(y=HIT_LINE, color="red", linestyle=":", linewidth=1.5, alpha=0.7, label=f"Hit threshold ({HIT_LINE:.2f}m)")
    ax.fill_between(sv, 0, HIT_LINE, alpha=0.06, color="green")
    # 标记 tube 救回的点
    for v in space_vals:
        nd = space_no[v]["min_dist"]
        td = space_tube[v]["min_dist"]
        if td < HIT_LINE and nd >= HIT_LINE:
            ax.annotate("SAVED", xy=(v*100, td), fontsize=11, fontweight="bold",
                        color="darkgreen", ha="center", va="bottom",
                        xytext=(0, 15), textcoords="offset points")
    ax.set_xlabel("Space offset (cm)", fontsize=12)
    ax.set_ylabel("Min racket-ball distance (m)", fontsize=12)
    ax.set_title("Space Perturbation: min_dist", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, max(0.22, ax.get_ylim()[1]))

    # ===== 图 3: Tube 专有指标 — tube_ready =====
    ax = axes[1, 0]
    # 时间偏移
    tr_time = [time_tube[v].get("tube_ready", 0) for v in time_vals]
    ax.bar(tv - 3, tr_time, width=5, color="tab:blue", alpha=0.7, label="Time perturb: tube_ready (steps)")
    # 空间偏移
    tr_space = [space_tube[v].get("tube_ready", 0) for v in space_vals]
    ax2 = ax.twiny()
    ax2.bar(np.array(space_vals)*100 - 2, tr_space, width=3.5, color="tab:red", alpha=0.7, label="Space perturb: tube_ready (steps)")
    ax.set_xlabel("Time perturbation (ms)", fontsize=12, color="tab:blue")
    ax2.set_xlabel("Space offset (cm)", fontsize=12, color="tab:red")
    ax.set_ylabel("tube_ready (steps)", fontsize=12)
    ax.set_title("Tube-active steps in spatial corridor", fontsize=13)
    ax.tick_params(axis="x", colors="tab:blue")
    ax2.tick_params(axis="x", colors="tab:red")
    ax.grid(True, alpha=0.2, axis="y")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="upper right")

    # ===== 图 4: 命中判定矩阵（彩色块）=====
    ax = axes[1, 1]
    # 构建数据矩阵
    time_hit_no  = [1 if time_no[v]["min_dist"] < HIT_LINE else 0 for v in time_vals]
    time_hit_tube = [1 if time_tube[v]["min_dist"] < HIT_LINE else 0 for v in time_vals]
    space_hit_no  = [1 if space_no[v]["min_dist"] < HIT_LINE else 0 for v in space_vals]
    space_hit_tube = [1 if space_tube[v]["min_dist"] < HIT_LINE else 0 for v in space_vals]

    # 时间部分
    y_labels_t = [f"{v:+d}ms" for v in time_vals]
    data_t = np.column_stack([time_hit_no, time_hit_tube])
    # 空间部分
    y_labels_s = [f"{v*100:+5.0f}cm" for v in space_vals]
    data_s = np.column_stack([space_hit_no, space_hit_tube])

    # 拼合
    all_labels = y_labels_t + y_labels_s
    all_data = np.vstack([data_t, np.zeros((1, 2)) * np.nan, data_s])  # 空行分隔

    im = ax.imshow(all_data, aspect="auto", cmap=plt.cm.RdYlGn, vmin=0, vmax=1,
                   interpolation="nearest")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["No-Tube", "Tube"], fontsize=11)
    ax.set_yticks(range(len(all_labels)))
    ax.set_yticklabels(all_labels, fontsize=9)
    ax.set_title("Hit / Miss Matrix  (green=HIT, red=MISS)", fontsize=13)
    # 添加分隔线
    n_time = len(time_vals)
    ax.axhline(y=n_time - 0.5, color="black", linewidth=1.5)
    ax.text(-0.5, n_time / 2 - 0.5, "TIME\nPERTURB", ha="center", va="center",
            fontsize=9, fontweight="bold", rotation=90, color="gray")
    ax.text(-0.5, n_time + 1 + len(space_vals)/2 - 0.5, "SPACE\nPERTURB",
            ha="center", va="center", fontsize=9, fontweight="bold",
            rotation=90, color="gray")

    plt.tight_layout()
    out_path = Path(__file__).resolve().parent.parent.parent / "results" / f"perturb_compare_s{args.seed}.png"
    fig.savefig(str(out_path), dpi=150)
    plt.close(fig)
    print(f"图表已保存到: {out_path}")

    # ── 打印文本表格 ──
    print("\n" + "=" * 80)
    print("  Time Perturbation")
    print("=" * 80)
    print(f"{'pert':>6} | {'NO min_d':>9} | {'TUBE min_d':>10} | {'t_ready':>7} | {'result':>12}")
    print("-" * 60)
    for v in time_vals:
        nd = time_no[v]["min_dist"]
        td = time_tube[v]["min_dist"]
        tr = time_tube[v].get("tube_ready", 0)
        nh = "HIT" if nd < 0.15 else "MISS"
        th = "HIT" if td < 0.15 else "MISS"
        better = "TUBE" if td < nd - 0.002 else "NO" if nd < td - 0.002 else "=="
        star = " ***" if th == "HIT" and nh == "MISS" else ""
        print(f"{v:+6d}ms | {nd:>8.3f}m | {td:>9.3f}m | {tr:>5.0f}st | {nh:>5} {th:>5} {better}{star}")

    print("\n" + "=" * 80)
    print("  Space Perturbation")
    print("=" * 80)
    print(f"{'offset':>7} | {'NO min_d':>9} | {'TUBE min_d':>10} | {'t_ready':>7} | {'result':>12}")
    print("-" * 60)
    for v in space_vals:
        nd = space_no[v]["min_dist"]
        td = space_tube[v]["min_dist"]
        tr = space_tube[v].get("tube_ready", 0)
        nh = "HIT" if nd < 0.15 else "MISS"
        th = "HIT" if td < 0.15 else "MISS"
        better = "TUBE" if td < nd - 0.002 else "NO" if nd < td - 0.002 else "=="
        star = " ***" if th == "HIT" and nh == "MISS" else ""
        print(f"{v:+7.3f}m | {nd:>8.3f}m | {td:>9.3f}m | {tr:>5.0f}st | {nh:>5} {th:>5} {better}{star}")

    print(f"\n*** = TUBE saved a case that NO-TUBE missed")
    print(f"Hit threshold = {HIT_LINE:.2f}m (racket radius + ball radius)")


if __name__ == "__main__":
    main()
