"""V2 Tube 改进版鲁棒性批量实验脚本。

在 TCP <= 1.8 m/s + 关节严格约束下，测试时间和空间预测扰动的击球成功率。
每组 (time_perturb, space_perturb) 组合重复 N 次（不同 seed）。

实验网格：
  时间扰动: -200, -100, -50, 0, +50, +100, +200 ms
  空间扰动: -20, -10, -5, 0, +5, +10, +20 cm
  每组 5 次

用法:
    python scripts/run_robustness_batch_v2.py
    python scripts/run_robustness_batch_v2.py --seeds 5 --ball-speed 9
    python scripts/run_robustness_batch_v2.py --quick   # 快速模式（仅 3x3 网格）
"""
import subprocess
import sys
import re
import time
import json
import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
RUNNER = SCRIPT_DIR / "run_tcp_limit_experiment_v2.py"


def run_single_experiment(
    time_perturb_ms: float,
    space_perturb_m: float,
    seed: int,
    ball_speed: float = 9.0,
    max_tcp: float = 1.8,
) -> dict:
    """运行单次实验，解析输出返回结果字典。"""
    cmd = [
        sys.executable, str(RUNNER),
        "--ball-speed", str(ball_speed),
        "--seed", str(seed),
        "--max-tcp", str(max_tcp),
        "--time-perturb-ms", str(time_perturb_ms),
        "--space-perturb-m", str(space_perturb_m),
    ]

    result = {
        "time_perturb_ms": time_perturb_ms,
        "space_perturb_m": space_perturb_m,
        "seed": seed,
        "success": False,
        "pos_error": float("inf"),
        "vel_error": float("inf"),
        "ball_near_ms": 0.0,
        "tube_ready_ms": 0.0,
        "min_dist": float("inf"),
        "max_tcp_speed": 0.0,
        "max_qdot_ratio": 0.0,
        "error": None,
    }

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(PROJECT_DIR),
            encoding="utf-8",
            errors="replace",
        )
        output = proc.stdout + proc.stderr

        if proc.returncode != 0:
            result["error"] = f"exit_code={proc.returncode}"
            err_lines = output.strip().split("\n")[-5:]
            result["error_detail"] = "\n".join(err_lines)
            return result

        # 优先解析 __RESULT__ 结构化行（英文，无编码问题）
        m = re.search(r"__RESULT__:\s*(.+)", output)
        if m:
            result_str = m.group(1)
            pairs = result_str.split()
            parsed = {}
            for pair in pairs:
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    parsed[k] = v

            result["pos_error"] = float(parsed.get("pos_error", "inf"))
            result["vel_error"] = float(parsed.get("vel_error", "inf"))
            result["min_dist"] = float(parsed.get("min_dist", "inf"))
            result["ball_near_ms"] = float(parsed.get("ball_near_ms", "0"))
            result["tube_ready_ms"] = float(parsed.get("tube_ready_ms", "0"))
            result["max_tcp_speed"] = float(parsed.get("max_tcp", "0"))
            result["max_qdot_ratio"] = float(parsed.get("max_qdot", "0"))
            result["hit_type"] = parsed.get("hit_type", "unknown")
            result["hit_time_error_ms"] = float(parsed.get("hit_time_error_ms", "0"))
            result["hit_pos_error"] = float(parsed.get("hit_pos_error", "inf"))
        else:
            # 回退：从非结构化输出解析
            m = re.search(r"max_tcp=([\d.]+)m/s", output)
            if m:
                result["max_tcp_speed"] = float(m.group(1))
            m = re.search(r"max_qdot=([\d.]+)x", output)
            if m:
                result["max_qdot_ratio"] = float(m.group(1))

        # 成功判定：球-拍距离 < 0.153m（物理接触阈值）
        ball_racket_threshold = 0.033 + 0.12  # 0.153m
        result["success"] = result["pos_error"] < ball_racket_threshold

        # 精确命中
        result["precise"] = result["pos_error"] < 0.05

    except subprocess.TimeoutExpired:
        result["error"] = "timeout"
    except Exception as e:
        result["error"] = str(e)

    return result


def main():
    parser = argparse.ArgumentParser(description="V2 Tube 鲁棒性批量实验")
    parser.add_argument("--ball-speed", type=float, default=9.0)
    parser.add_argument("--max-tcp", type=float, default=1.8)
    parser.add_argument("--seeds", type=int, default=5, help="每组实验重复次数")
    parser.add_argument("--quick", action="store_true", help="快速模式：3x3 网格")
    parser.add_argument("--workers", type=int, default=1, help="并行进程数")
    parser.add_argument("--output", type=str, default=None, help="结果输出 JSON 路径")
    args = parser.parse_args()

    # 实验网格
    if args.quick:
        time_perturbs_ms = [-200.0, 0.0, 200.0]
        space_perturbs_m = [-0.20, 0.0, 0.20]
    else:
        time_perturbs_ms = [-200.0, -100.0, -50.0, 0.0, 50.0, 100.0, 200.0]
        space_perturbs_m = [-0.20, -0.10, -0.05, 0.0, 0.05, 0.10, 0.20]

    seeds = list(range(42, 42 + args.seeds))
    total = len(time_perturbs_ms) * len(space_perturbs_m) * len(seeds)

    print(f"{'='*70}")
    print(f"  V2 Tube 鲁棒性批量实验")
    print(f"  TCP <= {args.max_tcp} m/s | 关节 qdot <= 1.0x | ball_speed={args.ball_speed} m/s")
    print(f"  时间扰动: {time_perturbs_ms} ms")
    print(f"  空间扰动: {[f'{x*100:.0f}cm' for x in space_perturbs_m]}")
    print(f"  每组 {args.seeds} 次 (seeds={seeds})")
    print(f"  总实验数: {total}")
    print(f"  并行进程: {args.workers}")
    print(f"{'='*70}")

    # 构建所有实验任务
    tasks = []
    for tp in time_perturbs_ms:
        for sp in space_perturbs_m:
            for seed in seeds:
                tasks.append((tp, sp, seed))

    # 运行实验
    all_results = []
    t_start = time.time()

    if args.workers > 1:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {}
            for tp, sp, seed in tasks:
                fut = executor.submit(
                    run_single_experiment,
                    tp, sp, seed,
                    args.ball_speed, args.max_tcp,
                )
                futures[fut] = (tp, sp, seed)

            for i, fut in enumerate(as_completed(futures)):
                res = fut.result()
                all_results.append(res)
                tp, sp, seed = futures[fut]
                done = i + 1
                elapsed = time.time() - t_start
                eta = elapsed / done * (total - done) if done > 0 else 0
                status = "OK" if res["success"] else "MISS"
                if res["error"]:
                    status = f"ERR({res['error']})"
                print(f"  [{done}/{total}] tp={tp:+6.0f}ms sp={sp:+5.2f}m seed={seed} "
                      f"pos_err={res['pos_error']:.4f}m {status} "
                      f"({elapsed:.0f}s, ETA {eta:.0f}s)")
    else:
        for i, (tp, sp, seed) in enumerate(tasks):
            res = run_single_experiment(tp, sp, seed, args.ball_speed, args.max_tcp)
            all_results.append(res)
            done = i + 1
            elapsed = time.time() - t_start
            eta = elapsed / done * (total - done) if done > 0 else 0
            status = "OK" if res["success"] else "MISS"
            if res["error"]:
                status = f"ERR({res['error']})"
            print(f"  [{done}/{total}] tp={tp:+6.0f}ms sp={sp:+5.2f}m seed={seed} "
                  f"pos_err={res['pos_error']:.4f}m {status} "
                  f"({elapsed:.0f}s, ETA {eta:.0f}s)")

    t_total = time.time() - t_start
    print(f"\n总耗时: {t_total:.1f}s ({t_total/60:.1f}min)")

    # ===== 汇总结果 =====
    print(f"\n{'='*70}")
    print(f"  成功率汇总（球-拍距离 < 15.3cm 为命中）")
    print(f"{'='*70}")

    # 表头
    header = f"{'时间\\空间':>10s}"
    for sp in space_perturbs_m:
        header += f" | {sp*100:+6.1f}cm"
    header += " | 平均"
    print(header)
    print("-" * len(header))

    time_success_rates = {}
    for tp in time_perturbs_ms:
        row = f"{tp:+8.0f}ms"
        row_rates = []
        for sp in space_perturbs_m:
            # 收集该组合的所有结果
            group = [r for r in all_results
                     if abs(r["time_perturb_ms"] - tp) < 0.1
                     and abs(r["space_perturb_m"] - sp) < 0.001]
            n_success = sum(1 for r in group if r["success"])
            n_total = len(group)
            rate = n_success / n_total if n_total > 0 else 0.0
            row_rates.append(rate)
            row += f" | {rate*100:5.1f}%({n_success}/{n_total})"
        avg_rate = np.mean(row_rates) if row_rates else 0.0
        row += f" | {avg_rate*100:5.1f}%"
        time_success_rates[tp] = avg_rate
        print(row)

    # 空间平均行
    print("-" * len(header))
    avg_row = f"{'平均':>10s}"
    for sp in space_perturbs_m:
        group = [r for r in all_results
                 if abs(r["space_perturb_m"] - sp) < 0.001]
        n_success = sum(1 for r in group if r["success"])
        n_total = len(group)
        rate = n_success / n_total if n_total > 0 else 0.0
        avg_row += f" | {rate*100:5.1f}%({n_success}/{n_total})"
    overall_success = sum(1 for r in all_results if r["success"])
    avg_row += f" | {overall_success/len(all_results)*100:5.1f}%"
    print(avg_row)

    # ===== 精确命中率 =====
    print(f"\n{'='*70}")
    print(f"  精确命中率（球-拍距离 < 5cm）")
    print(f"{'='*70}")

    header = f"{'时间\\空间':>10s}"
    for sp in space_perturbs_m:
        header += f" | {sp*100:+6.1f}cm"
    header += " | 平均"
    print(header)
    print("-" * len(header))

    for tp in time_perturbs_ms:
        row = f"{tp:+8.0f}ms"
        row_rates = []
        for sp in space_perturbs_m:
            group = [r for r in all_results
                     if abs(r["time_perturb_ms"] - tp) < 0.1
                     and abs(r["space_perturb_m"] - sp) < 0.001]
            n_precise = sum(1 for r in group if r.get("precise", False))
            n_total = len(group)
            rate = n_precise / n_total if n_total > 0 else 0.0
            row_rates.append(rate)
            row += f" | {rate*100:5.1f}%({n_precise}/{n_total})"
        avg_rate = np.mean(row_rates) if row_rates else 0.0
        row += f" | {avg_rate*100:5.1f}%"
        print(row)

    # ===== 平均位置误差矩阵 =====
    print(f"\n{'='*70}")
    print(f"  平均位置误差 (m)")
    print(f"{'='*70}")

    header = f"{'时间\\空间':>10s}"
    for sp in space_perturbs_m:
        header += f" | {sp*100:+6.1f}cm"
    print(header)
    print("-" * len(header))

    for tp in time_perturbs_ms:
        row = f"{tp:+8.0f}ms"
        for sp in space_perturbs_m:
            group = [r for r in all_results
                     if abs(r["time_perturb_ms"] - tp) < 0.1
                     and abs(r["space_perturb_m"] - sp) < 0.001]
            errors = [r["pos_error"] for r in group if r["pos_error"] < float("inf")]
            avg_err = np.mean(errors) if errors else float("inf")
            row += f" | {avg_err:7.4f}"
        print(row)

    # ===== 保存 JSON =====
    output_path = args.output or str(PROJECT_DIR / "results" / "v2_robustness_results.json")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "config": {
            "ball_speed": args.ball_speed,
            "max_tcp": args.max_tcp,
            "time_perturbs_ms": time_perturbs_ms,
            "space_perturbs_m": space_perturbs_m,
            "seeds": seeds,
            "total_time_s": t_total,
        },
        "results": all_results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存到: {output_path}")

    # 总体统计
    n_ok = sum(1 for r in all_results if r["success"])
    n_err = sum(1 for r in all_results if r["error"] is not None)
    avg_pos_err = np.mean([r["pos_error"] for r in all_results if r["pos_error"] < float("inf")])
    print(f"\n总体: {n_ok}/{len(all_results)} 命中 ({n_ok/len(all_results)*100:.1f}%) | "
          f"错误: {n_err} | 平均位置误差: {avg_pos_err:.4f}m")


if __name__ == "__main__":
    main()
