"""实验E: No-Tube vs Tube 在预测误差下的对比。

与实验D相同条件，但禁用 Tube 机制，对比 MPC 重规划单独 vs Tube+重规划的鲁棒性。

用法:
    python scripts/run_no_tube_robustness.py
    python scripts/run_no_tube_robustness.py --skip-run
    python scripts/run_no_tube_robustness.py --ball-speed 7 --n-seeds 20
"""
import argparse
import subprocess
import sys
import re
import time
import json
from pathlib import Path

import numpy as np

CWD = Path(__file__).resolve().parent.parent.parent


def run_one(
    ball_speed: float,
    seed: int,
    time_perturb_ms: float,
    space_perturb_m: float,
    max_tcp: float,
    use_tube: str,
    perturb_alpha_min: float = 0.0,
) -> dict:
    cmd = [
        sys.executable, "scripts/exp/run_tcp_limit_experiment.py",
        "--ball-speed", str(ball_speed),
        "--seed", str(seed),
        "--max-tcp", str(max_tcp),
        "--use-tube", use_tube,
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
        return {"hit": False, "hit_type": "timeout", "min_dist": None}

    r = {"hit": False, "hit_type": "miss", "min_dist": None}
    for line in txt.split("\n"):
        st = line.strip()
        if "RM-65" in st:
            if "5cm" in st:
                r["hit"] = True; r["hit_type"] = "PRECISE"
            elif "0.153" in st:
                r["hit"] = True; r["hit_type"] = "HIT"
            elif "10cm" in st:
                r["hit_type"] = "near"
        if st.endswith("m") and r["min_dist"] is None:
            m = re.search(r"([\d.]+)\s*m$", st)
            if m:
                v = float(m.group(1))
                if 0.001 < v < 0.5:
                    r["min_dist"] = v
    return r


def main() -> None:
    parser = argparse.ArgumentParser(description="实验E: No-Tube 鲁棒性对比")
    parser.add_argument("--ball-speed", type=float, default=7)
    parser.add_argument("--max-tcp", type=float, default=1.8)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--skip-run", action="store_true")
    parser.add_argument("--perturb-alpha-min", type=float, default=0.0,
                        help="衰减扰动保底值 (0~1)")
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    time_vals = [0, 50, 100, 150, 200]
    space_vals_m = [0.0, 0.05, 0.10, 0.15, 0.20]

    results_dir = CWD / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    cache = results_dir / f"experiment_e_notube_N{args.n_seeds}_s{args.seed_start}_v{args.ball_speed}.npz"

    # 加载实验D (Tube) 的缓存用于对比
    tube_cache = results_dir / f"experiment_d_N{args.n_seeds}_s{args.seed_start}_v{args.ball_speed}.npz"
    tube_time = {}
    tube_space = {}
    if tube_cache.exists():
        td = np.load(tube_cache)
        tube_time_vals = td["time_vals"].tolist()
        tube_space_vals = td["space_vals_m"].tolist()
        for t_ms in tube_time_vals:
            tube_time[t_ms] = {
                "dist": td[f"t{t_ms}_dist"],
                "hit": td[f"t{t_ms}_hit"],
            }
        for s_m in tube_space_vals:
            tube_space[s_m] = {
                "dist": td[f"s{s_m*100:.0f}_dist"],
                "hit": td[f"s{s_m*100:.0f}_hit"],
            }

    if not args.skip_run:
        total = (len(time_vals) + len(space_vals_m)) * args.n_seeds
        count = 0
        t_start = time.perf_counter()

        # === 时间扰动 ===
        print(f"\n{'='*70}")
        print(f"  实验E (No-Tube): 时间扰动, 球速={args.ball_speed}m/s, TCP<={args.max_tcp}m/s")
        print(f"{'='*70}")

        time_results = {}
        for t_ms in time_vals:
            time_results[t_ms] = []
            for s in seeds:
                count += 1
                r = run_one(args.ball_speed, s, t_ms, 0.0, args.max_tcp, "false",
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
        print(f"  实验E (No-Tube): 空间扰动, 球速={args.ball_speed}m/s, TCP<={args.max_tcp}m/s")
        print(f"{'='*70}")

        space_results = {}
        for s_m in space_vals_m:
            space_results[s_m] = []
            for s in seeds:
                count += 1
                r = run_one(args.ball_speed, s, 0.0, s_m, args.max_tcp, "false",
                            perturb_alpha_min=args.perturb_alpha_min)
                space_results[s_m].append(r)
                h = r["hit_type"]
                d = f"{r['min_dist']:.3f}m" if r["min_dist"] else "?"
                elapsed = time.perf_counter() - t_start
                eta = elapsed / count * (total - count) if count > 0 else 0
                print(f"  [{count}/{total}] space={s_m*100:>5.0f}cm seed={s:>3} "
                      f"{h:>7} dist={d}  ETA={eta/60:.0f}min", flush=True)

        # 保存
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

    # === Tube vs No-Tube 对比表 ===
    has_tube = len(tube_time) > 0

    print(f"\n{'='*80}")
    print(f"  Tube vs No-Tube: 时间扰动 (球速={args.ball_speed}m/s, TCP<={args.max_tcp}m/s, qdot<=1.0x)")
    print(f"  N={args.n_seeds} seeds | 命中阈值 < 0.153m")
    print(f"{'='*80}")
    header = f"{'扰动':>8} | {'No-Tube':>18} |"
    if has_tube:
        header += f" {'Tube':>18} |"
    print(header)
    sub = f"{'':>8} | {'命中':>5} {'距离':>12} |"
    if has_tube:
        sub += f" {'命中':>5} {'距离':>12} |"
    print(sub)
    print("-" * (48 if not has_tube else 80))

    for t_ms in time_vals:
        nt_dists = [r["min_dist"] for r in time_results[t_ms] if r["min_dist"] is not None]
        nt_hit = sum(1 for r in time_results[t_ms] if r["hit"])
        n = len(time_results[t_ms])
        nt_avg = float(np.mean(nt_dists)) * 1000 if nt_dists else float("nan")
        line = f"{t_ms:>6}ms | {nt_hit:>3}/{n:<2} {nt_avg:>7.1f}mm    |"
        if has_tube and t_ms in tube_time:
            t_dists = tube_time[t_ms]["dist"]
            t_hits = tube_time[t_ms]["hit"]
            t_hit = int(np.sum(t_hits))
            t_avg = float(np.nanmean(t_dists)) * 1000
            line += f" {t_hit:>3}/{n:<2} {t_avg:>7.1f}mm    |"
        print(line)

    print(f"\n{'='*80}")
    print(f"  Tube vs No-Tube: 空间扰动 (球速={args.ball_speed}m/s, TCP<={args.max_tcp}m/s, qdot<=1.0x)")
    print(f"  N={args.n_seeds} seeds | 命中阈值 < 0.153m")
    print(f"{'='*80}")
    header = f"{'扰动':>8} | {'No-Tube':>18} |"
    if has_tube:
        header += f" {'Tube':>18} |"
    print(header)
    sub = f"{'':>8} | {'命中':>5} {'距离':>12} |"
    if has_tube:
        sub += f" {'命中':>5} {'距离':>12} |"
    print(sub)
    print("-" * (48 if not has_tube else 80))

    for s_m in space_vals_m:
        nt_dists = [r["min_dist"] for r in space_results[s_m] if r["min_dist"] is not None]
        nt_hit = sum(1 for r in space_results[s_m] if r["hit"])
        n = len(space_results[s_m])
        nt_avg = float(np.mean(nt_dists)) * 1000 if nt_dists else float("nan")
        line = f"{s_m*100:>6.0f}cm | {nt_hit:>3}/{n:<2} {nt_avg:>7.1f}mm    |"
        if has_tube and s_m in tube_space:
            t_dists = tube_space[s_m]["dist"]
            t_hits = tube_space[s_m]["hit"]
            t_hit = int(np.sum(t_hits))
            t_avg = float(np.nanmean(t_dists)) * 1000
            line += f" {t_hit:>3}/{n:<2} {t_avg:>7.1f}mm    |"
        print(line)

    # === 绘图 ===
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # 时间扰动对比图
        ax = axes[0]
        tv = np.array(time_vals)

        nt_hr = []
        nt_ad = []
        nt_sd = []
        for t_ms in time_vals:
            dists = [r["min_dist"] for r in time_results[t_ms] if r["min_dist"] is not None]
            n = len(time_results[t_ms])
            nt_hr.append(sum(1 for r in time_results[t_ms] if r["hit"]) / n * 100)
            nt_ad.append(float(np.mean(dists)) * 1000 if dists else 0)
            nt_sd.append(float(np.std(dists)) * 1000 if dists else 0)

        ax.bar(tv - 1.5, nt_hr, width=3, color="tab:orange", alpha=0.7, label="No-Tube")

        if has_tube:
            t_hr = []
            t_ad = []
            for t_ms in time_vals:
                if t_ms in tube_time:
                    t_hr.append(float(np.mean(tube_time[t_ms]["hit"])) * 100)
                    t_ad.append(float(np.nanmean(tube_time[t_ms]["dist"])) * 1000)
                else:
                    t_hr.append(0); t_ad.append(0)
            ax.bar(tv + 1.5, t_hr, width=3, color="tab:blue", alpha=0.7, label="Tube")

        ax.set_xlabel("Time perturbation (ms)", fontsize=12)
        ax.set_ylabel("Hit rate (%)", fontsize=12)
        ax.set_ylim(0, 110)
        ax.axhline(y=100, color="gray", linestyle=":", alpha=0.3)
        ax.set_title(f"Time Error: Tube vs No-Tube (v={args.ball_speed}m/s, TCP<={args.max_tcp}m/s)", fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        # 空间扰动对比图
        ax2 = axes[1]
        sv = np.array([v * 100 for v in space_vals_m])

        nt_hr_s = []
        for s_m in space_vals_m:
            n = len(space_results[s_m])
            nt_hr_s.append(sum(1 for r in space_results[s_m] if r["hit"]) / n * 100)

        ax2.bar(sv - 1.5, nt_hr_s, width=3, color="tab:orange", alpha=0.7, label="No-Tube")

        if has_tube:
            t_hr_s = []
            for s_m in space_vals_m:
                if s_m in tube_space:
                    t_hr_s.append(float(np.mean(tube_space[s_m]["hit"])) * 100)
                else:
                    t_hr_s.append(0)
            ax2.bar(sv + 1.5, t_hr_s, width=3, color="tab:blue", alpha=0.7, label="Tube")

        ax2.set_xlabel("Space perturbation (cm)", fontsize=12)
        ax2.set_ylabel("Hit rate (%)", fontsize=12)
        ax2.set_ylim(0, 110)
        ax2.axhline(y=100, color="gray", linestyle=":", alpha=0.3)
        ax2.set_title(f"Space Error: Tube vs No-Tube (v={args.ball_speed}m/s, TCP<={args.max_tcp}m/s)", fontsize=11)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = results_dir / f"experiment_e_notube_v{args.ball_speed}_N{args.n_seeds}.png"
        fig.savefig(str(out_path), dpi=150)
        plt.close(fig)
        print(f"\n图表已保存: {out_path}")
    except ImportError:
        print("\nmatplotlib not installed, skip plot")

    # JSON
    json_out = {
        "config": {
            "ball_speed": args.ball_speed,
            "max_tcp": args.max_tcp,
            "n_seeds": args.n_seeds,
            "tube": False,
            "constraints": "qdot <= 1.0x + TCP <= 1.8 m/s, no tube, no exemption",
        },
        "time_perturbation": {},
        "space_perturbation": {},
    }
    for t_ms in time_vals:
        dists = [r["min_dist"] for r in time_results[t_ms] if r["min_dist"] is not None]
        n_hit = sum(1 for r in time_results[t_ms] if r["hit"])
        n = len(time_results[t_ms])
        json_out["time_perturbation"][str(t_ms)] = {
            "hit_rate": f"{n_hit}/{n} ({n_hit/n*100:.0f}%)",
            "avg_dist_mm": float(np.mean(dists) * 1000) if dists else None,
        }
    for s_m in space_vals_m:
        dists = [r["min_dist"] for r in space_results[s_m] if r["min_dist"] is not None]
        n_hit = sum(1 for r in space_results[s_m] if r["hit"])
        n = len(space_results[s_m])
        json_out["space_perturbation"][f"{s_m*100:.0f}cm"] = {
            "hit_rate": f"{n_hit}/{n} ({n_hit/n*100:.0f}%)",
            "avg_dist_mm": float(np.mean(dists) * 1000) if dists else None,
        }
    json_path = results_dir / f"experiment_e_notube_v{args.ball_speed}_N{args.n_seeds}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
