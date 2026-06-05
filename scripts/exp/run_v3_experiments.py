"""v3 全程硬约束批量实验脚本。

实验1: 球速 5/6/7/8 m/s 下的成功率和位置误差 (每速20个seed)
实验2: 球速 7 m/s 下 tube vs no-tube 时间/空间扰动对比 (20 seeds)
"""
import subprocess
import sys
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT = ROOT / "scripts" / "exp" / "run_tcp_limit_experiment_v3.py"
HIT_THRESHOLD = 0.12


def run_one(args_list):
    """运行单个实验，返回解析后的结果字典。"""
    cmd = [sys.executable, str(SCRIPT)] + args_list
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            cwd=str(ROOT),
        )
        output = result.stdout + result.stderr
        for line in output.splitlines():
            if "__RESULT__:" in line:
                parts = line.split("__RESULT__:")[1].strip()
                kv = {}
                for token in parts.split():
                    if "=" in token:
                        k, v = token.split("=", 1)
                        try:
                            kv[k] = float(v)
                        except ValueError:
                            kv[k] = v
                return kv
    except subprocess.TimeoutExpired:
        pass
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
    return None


def exp1_ball_speeds():
    """实验1: 不同球速下的成功率。"""
    speeds = [5, 6, 7, 8]
    n_seeds = 20

    tasks = []
    for speed in speeds:
        for seed in range(n_seeds):
            args = ["--ball-speed", str(speed), "--seed", str(seed)]
            tasks.append((speed, seed, args))

    print(f"=== 实验1: 球速 {speeds} m/s, 每速 {n_seeds} seeds ===")
    all_results = {}
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(run_one, t[2]): (t[0], t[1]) for t in tasks}
        for fut in as_completed(futures):
            speed, seed = futures[fut]
            r = fut.result()
            if speed not in all_results:
                all_results[speed] = []
            all_results[speed].append((seed, r))
            status = "HIT" if r and r.get("hit_type") in ("active", "passive") else "MISS"
            pos = r.get("pos_error", -1) if r else -1
            tcp = r.get("max_tcp", -1) if r else -1
            print(f"  speed={speed} seed={seed}: {status} pos_err={pos:.4f}m tcp={tcp:.2f}m/s")

    return all_results


def exp2_perturbation():
    """实验2: 球速7, tube vs no-tube, 时间/空间扰动。"""
    speed = 7.0
    n_seeds = 20
    time_perturbs = [-200.0, -100.0, 0.0, 100.0, 200.0]
    space_perturbs = [-0.20, -0.10, 0.0, 0.10, 0.20]
    modes = [("tube", "true"), ("no_tube", "false")]

    tasks = []
    for mode_label, use_tube in modes:
        for t_ms in time_perturbs:
            for s_m in space_perturbs:
                for seed in range(n_seeds):
                    args = [
                        "--ball-speed", str(speed), "--seed", str(seed),
                        "--use-tube", use_tube,
                        "--time-perturb-ms", str(t_ms),
                        "--space-perturb-m", str(s_m),
                    ]
                    tasks.append((mode_label, t_ms, s_m, seed, args))

    total = len(tasks)
    print(f"\n=== 实验2: 球速={speed}m/s, tube vs no-tube, "
          f"时间{time_perturbs}ms × 空间{[s*100 for s in space_perturbs]}cm, "
          f"{n_seeds} seeds, 共{total}次 ===")

    all_results = {}
    done = 0
    with ProcessPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(run_one, t[4]): (t[0], t[1], t[2], t[3]) for t in tasks}
        for fut in as_completed(futures):
            mode, t_ms, s_m, seed = futures[fut]
            key = (mode, t_ms, s_m)
            if key not in all_results:
                all_results[key] = []
            r = fut.result()
            all_results[key].append((seed, r))
            done += 1
            status = "HIT" if r and r.get("hit_type") in ("active", "passive") else "MISS"
            pos = r.get("pos_error", -1) if r else -1
            if done % 50 == 0 or done == total:
                print(f"  [{done}/{total}] {mode} t={t_ms:.0f}ms s={s_m*100:.0f}cm seed={seed}: {status} pos_err={pos:.4f}m")

    return all_results, time_perturbs, space_perturbs


def summarize_exp1(results):
    lines = []
    lines.append("## 实验1: 不同球速下的击球性能\n")
    lines.append("| 球速 (m/s) | 成功率 | 平均位置误差 (m) | 最小位置误差 (m) | 最大TCP速度 (m/s) | max_qdot |")
    lines.append("|---|---|---|---|---|---|")
    for speed in sorted(results.keys()):
        runs = [r for _, r in results[speed] if r is not None]
        if not runs:
            continue
        hits = [r for r in runs if r.get("hit_type") in ("active", "passive")]
        hit_rate = len(hits) / len(runs) * 100
        pos_errors = [r["pos_error"] for r in runs]
        avg_pos = sum(pos_errors) / len(pos_errors)
        min_pos = min(pos_errors)
        tcp_speeds = [r.get("max_tcp", 0) for r in runs]
        avg_tcp = sum(tcp_speeds) / len(tcp_speeds)
        qdots = [r.get("max_qdot", 0) for r in runs]
        avg_qdot = sum(qdots) / len(qdots)
        lines.append(f"| {speed} | {hit_rate:.0f}% ({len(hits)}/{len(runs)}) | {avg_pos:.4f} | {min_pos:.4f} | {avg_tcp:.2f} | {avg_qdot:.2f}x |")
    lines.append("")
    return "\n".join(lines)


def summarize_exp2(results, time_perturbs, space_perturbs):
    lines = []
    lines.append("## 实验2: 球速 7m/s 下 Tube vs No-Tube 扰动对比\n")

    for mode_label in ["tube", "no_tube"]:
        label = "Tube" if mode_label == "tube" else "No-Tube"
        lines.append(f"### {label}\n")
        lines.append("| 时间扰动 | " + " | ".join(f"空间{s*100:+.0f}cm" for s in space_perturbs) + " |")
        lines.append("|---|" + "|".join(["---"] * len(space_perturbs)) + "|")
        for t_ms in time_perturbs:
            cells = []
            for s_m in space_perturbs:
                key = (mode_label, t_ms, s_m)
                runs = [r for _, r in results.get(key, []) if r is not None]
                if not runs:
                    cells.append("N/A")
                    continue
                hits = [r for r in runs if r.get("hit_type") in ("active", "passive")]
                hit_rate = len(hits) / len(runs) * 100
                avg_pos = sum(r["pos_error"] for r in runs) / len(runs)
                cells.append(f"{hit_rate:.0f}% ({avg_pos:.3f}m)")
            lines.append(f"| t={t_ms:+.0f}ms | " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("### 汇总对比\n")
    lines.append("| 模式 | 总成功率 | 空间扰动成功率 | 时间扰动成功率 |")
    lines.append("|---|---|---|---|")
    for mode_label in ["tube", "no_tube"]:
        label = "Tube" if mode_label == "tube" else "No-Tube"
        total_hits = 0
        total_runs = 0
        space_hits = 0
        space_runs = 0
        time_hits = 0
        time_runs = 0
        for t_ms in time_perturbs:
            for s_m in space_perturbs:
                key = (mode_label, t_ms, s_m)
                runs = [r for _, r in results.get(key, []) if r is not None]
                hits = [r for r in runs if r.get("hit_type") in ("active", "passive")]
                total_hits += len(hits)
                total_runs += len(runs)
                if s_m != 0.0:
                    space_hits += len(hits)
                    space_runs += len(runs)
                if t_ms != 0.0:
                    time_hits += len(hits)
                    time_runs += len(runs)
        tr = f"{total_hits/total_runs*100:.0f}% ({total_hits}/{total_runs})" if total_runs > 0 else "N/A"
        sr = f"{space_hits/space_runs*100:.0f}% ({space_hits}/{space_runs})" if space_runs > 0 else "N/A"
        tmr = f"{time_hits/time_runs*100:.0f}% ({time_hits}/{time_runs})" if time_runs > 0 else "N/A"
        lines.append(f"| {label} | {tr} | {sr} | {tmr} |")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    r1 = exp1_ball_speeds()
    r2, t_perturbs, s_perturbs = exp2_perturbation()

    out_path = ROOT / "results" / "v3_experiment_raw.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = {
        "exp1": {str(k): [{"seed": s, "result": r} for s, r in v] for k, v in r1.items()},
        "exp2": {
            f"{k[0]}/t={k[1]}/s={k[2]}": [{"seed": s, "result": r} for s, r in v]
            for k, v in r2.items()
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)
    print(f"\n原始数据已保存到 {out_path}")

    s1 = summarize_exp1(r1)
    s2 = summarize_exp2(r2, t_perturbs, s_perturbs)
    print("\n" + s1)
    print(s2)
