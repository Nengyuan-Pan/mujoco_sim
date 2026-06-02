"""实验D: Tube 机制在预测误差下的鲁棒性测试。

在 TCP ≤ 1.8m/s + qdot ≤ 1.0x 约束下，施加时间(0~30ms)和空间(0~20cm)预测误差，
测试 Tube vs No-Tube 的命中率差异。

用法:
    python scripts/run_tube_robustness.py                    # 运行全部
    python scripts/run_tube_robustness.py --skip-run         # 只从缓存绘图
    python scripts/run_tube_robustness.py --n-seeds 20       # 每组种子数
    python scripts/run_tube_robustness.py --ball-speed 7     # 指定球速
"""
import argparse
import subprocess
import sys
import re
import time
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

CWD = Path(__file__).resolve().parent.parent


def run_one(
    ball_speed: float,
    seed: int,
    time_perturb_ms: float,
    space_perturb_m: float,
    max_tcp: float,
    perturb_alpha_min: float = 0.0,
) -> dict:
    """运行单次测试并返回指标字典。"""
    cmd = [
        sys.executable, "scripts/run_tcp_limit_experiment.py",
        "--ball-speed", str(ball_speed),
        "--seed", str(seed),
        "--max-tcp", str(max_tcp),
        "--time-perturb-ms", str(time_perturb_ms),
        "--space-perturb-m", str(space_perturb_m),
    ]
    if perturb_alpha_min > 0.001:
        cmd.extend(["--perturb-alpha-min", str(perturb_alpha_min)])
    try:
        result = subprocess.run(
            cmd, capture_output=True, timeout=300, cwd=str(CWD))
        txt = result.stdout.decode("gbk", errors="replace")
        txt += result.stderr.decode("gbk", errors="replace")
    except subprocess.TimeoutExpired:
        return {"hit": False, "hit_type": "timeout", "min_dist": None,
                "max_qdot": None, "max_tcp": None}

    r = {"hit": False, "hit_type": "miss", "min_dist": None,
         "max_qdot": None, "max_tcp": None}
    for line in txt.split("\n"):
        st = line.strip()
        if "RM-65" in st:
            if "5cm" in st:
                r["hit"] = True; r["hit_type"] = "PRECISE"
            elif "0.153" in st:
                r["hit"] = True; r["hit_type"] = "HIT"
            elif "10cm" in st:
                r["hit_type"] = "near"
        if "max_qdot=" in st:
            m = re.search(r"max_qdot=([\d.]+)x", st)
            if m: r["max_qdot"] = float(m.group(1))
            m2 = re.search(r"max_tcp=([\d.]+)m/s", st)
            if m2: r["max_tcp"] = float(m2.group(1))
        if st.endswith("m") and r["min_dist"] is None:
            m = re.search(r"([\d.]+)\s*m$", st)
            if m:
                v = float(m.group(1))
                if 0.001 < v < 0.5:
                    r["min_dist"] = v
    return r


def main() -> None:
    parser = argparse.ArgumentParser(description="实验D: Tube 鲁棒性测试")
    parser.add_argument("--ball-speed", type=float, default=7,
                        help="球速 (m/s), 默认 7")
    parser.add_argument("--max-tcp", type=float, default=1.8,
                        help="TCP 速度限制 (m/s)")
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--perturb-alpha-min", type=float, default=0.0,
                        help="衰减扰动保底值 (0~1), 0=衰减到0, 0.3=保留30%%残余偏差")
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))

    # 时间扰动: 0, 50, 100, 150, 200 ms（正值=MPC 认为球早到）
    time_vals = [0, 50, 100, 150, 200]
    # 空间扰动: 0, 5, 10, 15, 20 cm（正值=侧偏）
    space_vals_m = [0.0, 0.05, 0.10, 0.15, 0.20]

    results_dir = CWD / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    cache = results_dir / f"experiment_d_N{args.n_seeds}_s{args.seed_start}_v{args.ball_speed}.npz"

    if not args.skip_run:
        all_results = {}
        total = (len(time_vals) + len(space_vals_m)) * args.n_seeds
        count = 0
        t_start = time.perf_counter()

        # === 时间扰动 ===
        print(f"\n{'='*70}")
        print(f"  时间扰动实验: 球速={args.ball_speed}m/s, TCP<={args.max_tcp}m/s")
        print(f"  扰动值: {time_vals} ms, {args.n_seeds} seeds")
        print(f"{'='*70}")

        time_results = {}
        for t_ms in time_vals:
            time_results[t_ms] = []
            for s in seeds:
                count += 1
                r = run_one(args.ball_speed, s, t_ms, 0.0, args.max_tcp,
                            perturb_alpha_min=args.perturb_alpha_min)
                time_results[t_ms].append(r)
                h = r["hit_type"]
                d = f"{r['min_dist']:.3f}m" if r["min_dist"] else "?"
                elapsed = time.perf_counter() - t_start
                eta = elapsed / count * (total - count) if count > 0 else 0
                print(f"  [{count}/{total}] time={t_ms:>3}ms seed={s:>3} "
                      f"{h:>7} dist={d}  ETA={eta/60:.0f}min", flush=True)

        # === 空间扰动 ===
        print(f"\n{'='*70}")
        print(f"  空间扰动实验: 球速={args.ball_speed}m/s, TCP<={args.max_tcp}m/s")
        print(f"  扰动值: {[f'{v*100:.0f}cm' for v in space_vals_m]}, {args.n_seeds} seeds")
        print(f"{'='*70}")

        space_results = {}
        for s_m in space_vals_m:
            space_results[s_m] = []
            for s in seeds:
                count += 1
                r = run_one(args.ball_speed, s, 0.0, s_m, args.max_tcp,
                            perturb_alpha_min=args.perturb_alpha_min)
                space_results[s_m].append(r)
                h = r["hit_type"]
                d = f"{r['min_dist']:.3f}m" if r["min_dist"] else "?"
                elapsed = time.perf_counter() - t_start
                eta = elapsed / count * (total - count) if count > 0 else 0
                print(f"  [{count}/{total}] space={s_m*100:>5.0f}cm seed={s:>3} "
                      f"{h:>7} dist={d}  ETA={eta/60:.0f}min", flush=True)

        # === 保存缓存 ===
        save_dict = {
            "time_vals": np.array(time_vals),
            "space_vals_m": np.array(space_vals_m),
            "seeds": np.array(seeds),
            "ball_speed": args.ball_speed,
            "max_tcp": args.max_tcp,
        }
        for t_ms in time_vals:
            dists = [r["min_dist"] if r["min_dist"] else np.nan for r in time_results[t_ms]]
            hits = [float(r["hit"]) for r in time_results[t_ms]]
            save_dict[f"t{t_ms}_dist"] = np.array(dists)
            save_dict[f"t{t_ms}_hit"] = np.array(hits)
        for s_m in space_vals_m:
            dists = [r["min_dist"] if r["min_dist"] else np.nan for r in space_results[s_m]]
            hits = [float(r["hit"]) for r in space_results[s_m]]
            save_dict[f"s{s_m*100:.0f}_dist"] = np.array(dists)
            save_dict[f"s{s_m*100:.0f}_hit"] = np.array(hits)
        np.savez(cache, **save_dict)
        print(f"\n缓存已保存: {cache}")
    else:
        if not cache.exists():
            print("缓存不存在，请先不带 --skip-run 运行")
            return
        data = np.load(cache)
        time_vals = data["time_vals"].tolist()
        space_vals_m = data["space_vals_m"].tolist()
        seeds = data["seeds"].tolist()
        time_results = {}
        for t_ms in time_vals:
            time_results[t_ms] = [
                {"min_dist": float(d), "hit": bool(h)}
                for d, h in zip(data[f"t{t_ms}_dist"], data[f"t{t_ms}_hit"])
            ]
        space_results = {}
        for s_m in space_vals_m:
            space_results[s_m] = [
                {"min_dist": float(d), "hit": bool(h)}
                for d, h in zip(data[f"s{s_m*100:.0f}_dist"], data[f"s{s_m*100:.0f}_hit"])
            ]

    # === 统计 ===
    print(f"\n{'='*75}")
    print(f"  实验D: 时间扰动 (球速={args.ball_speed}m/s, TCP<={args.max_tcp}m/s, qdot≤1.0x)")
    print(f"  Tube 机制已启用 | N={args.n_seeds} seeds | 命中阈值 < 0.153m")
    print(f"{'='*75}")
    print(f"{'扰动':>8} | {'命中':>5} | {'命中率':>6} | {'平均距离':>10} | {'标准差':>8}")
    print("-" * 55)
    for t_ms in time_vals:
        dists = [r["min_dist"] for r in time_results[t_ms] if r["min_dist"] is not None]
        n_hit = sum(1 for r in time_results[t_ms] if r["hit"])
        n = len(time_results[t_ms])
        avg_d = float(np.mean(dists)) if dists else float("nan")
        std_d = float(np.std(dists)) if dists else float("nan")
        print(f"{t_ms:>6}ms | {n_hit:>3}/{n:<2} | {n_hit/n*100:>5.0f}% | "
              f"{avg_d*1000:>7.1f}mm | {std_d*1000:>6.1f}mm")

    print(f"\n{'='*75}")
    print(f"  实验D: 空间扰动 (球速={args.ball_speed}m/s, TCP<={args.max_tcp}m/s, qdot≤1.0x)")
    print(f"  Tube 机制已启用 | N={args.n_seeds} seeds | 命中阈值 < 0.153m")
    print(f"{'='*75}")
    print(f"{'扰动':>8} | {'命中':>5} | {'命中率':>6} | {'平均距离':>10} | {'标准差':>8}")
    print("-" * 55)
    for s_m in space_vals_m:
        dists = [r["min_dist"] for r in space_results[s_m] if r["min_dist"] is not None]
        n_hit = sum(1 for r in space_results[s_m] if r["hit"])
        n = len(space_results[s_m])
        avg_d = float(np.mean(dists)) if dists else float("nan")
        std_d = float(np.std(dists)) if dists else float("nan")
        print(f"{s_m*100:>6.0f}cm | {n_hit:>3}/{n:<2} | {n_hit/n*100:>5.0f}% | "
              f"{avg_d*1000:>7.1f}mm | {std_d*1000:>6.1f}mm")

    # === 绘图 ===
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        HIT_THRESHOLD = 0.153

        # 时间扰动图
        ax = axes[0]
        tv = np.array(time_vals)
        hit_rates = []
        avg_dists = []
        std_dists = []
        for t_ms in time_vals:
            dists = [r["min_dist"] for r in time_results[t_ms] if r["min_dist"] is not None]
            n = len(time_results[t_ms])
            n_hit = sum(1 for r in time_results[t_ms] if r["hit"])
            hit_rates.append(n_hit / n * 100)
            avg_dists.append(float(np.mean(dists)) * 1000 if dists else 0)
            std_dists.append(float(np.std(dists)) * 1000 if dists else 0)

        ax_twin = ax.twinx()
        bars = ax.bar(tv, hit_rates, width=3, color="tab:blue", alpha=0.6, label="命中率")
        ax_twin.plot(tv, avg_dists, "o-", color="tab:red", linewidth=2, markersize=8, label="平均距离")
        ax_twin.fill_between(tv,
                             [a - s for a, s in zip(avg_dists, std_dists)],
                             [a + s for a, s in zip(avg_dists, std_dists)],
                             color="tab:red", alpha=0.15)
        ax.set_xlabel("时间预测误差 (ms)", fontsize=12)
        ax.set_ylabel("命中率 (%)", fontsize=12, color="tab:blue")
        ax_twin.set_ylabel("平均最近距离 (mm)", fontsize=12, color="tab:red")
        ax.set_ylim(0, 110)
        ax.axhline(y=100, color="gray", linestyle=":", alpha=0.3)
        ax.set_title(f"时间预测误差 vs 命中率 (v={args.ball_speed}m/s, TCP≤{args.max_tcp}m/s)", fontsize=12)
        ax.grid(True, alpha=0.3)
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=9)

        # 空间扰动图
        ax2 = axes[1]
        sv = np.array([v * 100 for v in space_vals_m])
        hit_rates_s = []
        avg_dists_s = []
        std_dists_s = []
        for s_m in space_vals_m:
            dists = [r["min_dist"] for r in space_results[s_m] if r["min_dist"] is not None]
            n = len(space_results[s_m])
            n_hit = sum(1 for r in space_results[s_m] if r["hit"])
            hit_rates_s.append(n_hit / n * 100)
            avg_dists_s.append(float(np.mean(dists)) * 1000 if dists else 0)
            std_dists_s.append(float(np.std(dists)) * 1000 if dists else 0)

        ax2_twin = ax2.twinx()
        bars2 = ax2.bar(sv, hit_rates_s, width=3, color="tab:blue", alpha=0.6, label="命中率")
        ax2_twin.plot(sv, avg_dists_s, "o-", color="tab:red", linewidth=2, markersize=8, label="平均距离")
        ax2_twin.fill_between(sv,
                              [a - s for a, s in zip(avg_dists_s, std_dists_s)],
                              [a + s for a, s in zip(avg_dists_s, std_dists_s)],
                              color="tab:red", alpha=0.15)
        ax2.set_xlabel("空间预测误差 (cm)", fontsize=12)
        ax2.set_ylabel("命中率 (%)", fontsize=12, color="tab:blue")
        ax2_twin.set_ylabel("平均最近距离 (mm)", fontsize=12, color="tab:red")
        ax2.set_ylim(0, 110)
        ax2.axhline(y=100, color="gray", linestyle=":", alpha=0.3)
        ax2.set_title(f"空间预测误差 vs 命中率 (v={args.ball_speed}m/s, TCP≤{args.max_tcp}m/s)", fontsize=12)
        ax2.grid(True, alpha=0.3)
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2_twin.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, loc="lower left", fontsize=9)

        plt.tight_layout()
        out_path = results_dir / f"experiment_d_v{args.ball_speed}_N{args.n_seeds}.png"
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        print(f"\n图表已保存: {out_path}")
    except ImportError:
        print("\nmatplotlib 未安装，跳过绘图")

    # === 保存 JSON 结果 ===
    json_out = {}
    json_out["config"] = {
        "ball_speed": args.ball_speed,
        "max_tcp": args.max_tcp,
        "n_seeds": args.n_seeds,
        "seed_start": args.seed_start,
        "constraints": "qdot <= 1.0x + TCP <= 1.8 m/s, 无豁免",
    }
    json_out["time_perturbation"] = {}
    for t_ms in time_vals:
        dists = [r["min_dist"] for r in time_results[t_ms] if r["min_dist"] is not None]
        n_hit = sum(1 for r in time_results[t_ms] if r["hit"])
        n = len(time_results[t_ms])
        json_out["time_perturbation"][str(t_ms)] = {
            "hit_rate": f"{n_hit}/{n} ({n_hit/n*100:.0f}%)",
            "avg_dist_mm": float(np.mean(dists) * 1000) if dists else None,
            "std_dist_mm": float(np.std(dists) * 1000) if dists else None,
        }
    json_out["space_perturbation"] = {}
    for s_m in space_vals_m:
        dists = [r["min_dist"] for r in space_results[s_m] if r["min_dist"] is not None]
        n_hit = sum(1 for r in space_results[s_m] if r["hit"])
        n = len(space_results[s_m])
        json_out["space_perturbation"][f"{s_m*100:.0f}cm"] = {
            "hit_rate": f"{n_hit}/{n} ({n_hit/n*100:.0f}%)",
            "avg_dist_mm": float(np.mean(dists) * 1000) if dists else None,
            "std_dist_mm": float(np.std(dists) * 1000) if dists else None,
        }
    json_path = results_dir / f"experiment_d_v{args.ball_speed}_N{args.n_seeds}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    print(f"JSON 结果已保存: {json_path}")


if __name__ == "__main__":
    main()
