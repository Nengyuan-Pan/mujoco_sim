"""双向扰动实验：测试正/负时间和空间扰动下 Tube vs No-Tube 的差异。

时间扰动正值 = MPC 认为球早到（挥拍加速），负值 = MPC 认为球晚到（挥拍减速）。
空间扰动正值 = 瞄准点偏一侧，负值 = 偏另一侧。

用法:
    python scripts/run_bidirectional_perturb.py --n-seeds 20
    python scripts/run_bidirectional_perturb.py --n-seeds 10 --perturb-alpha-min 0.3
"""
import argparse
import subprocess
import sys
import re
import time
import json
import numpy as np
from pathlib import Path

CWD = Path(__file__).resolve().parent.parent.parent


def run_one(
    ball_speed: float,
    seed: int,
    time_perturb_ms: float,
    space_perturb_m: float,
    max_tcp: float,
    use_tube: str,
    perturb_alpha_min: float = 0.0,
    hit_threshold: float = 0.12,
    script: str = "run_tcp_limit_experiment.py",
    ablation: str | None = None,
) -> dict:
    cmd = [
        sys.executable, f"scripts/{script}",
        "--ball-speed", str(ball_speed),
        "--seed", str(seed),
        "--max-tcp", str(max_tcp),
        "--use-tube", use_tube,
        "--time-perturb-ms", str(time_perturb_ms),
        "--space-perturb-m", str(space_perturb_m),
    ]
    if perturb_alpha_min > 0.001:
        cmd.extend(["--perturb-alpha-min", str(perturb_alpha_min)])
    if ablation:
        cmd.extend(["--ablation", ablation])
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
        if st.endswith("m") and r["min_dist"] is None:
            m = re.search(r"([\d.]+)\s*m$", st)
            if m:
                v = float(m.group(1))
                if 0.001 < v < 0.5:
                    r["min_dist"] = v
    if r["min_dist"] is not None:
        if r["min_dist"] < 0.05:
            r["hit"] = True; r["hit_type"] = "PRECISE"
        elif r["min_dist"] < hit_threshold:
            r["hit"] = True; r["hit_type"] = "HIT"
    return r


def run_batch(
    ball_speed: float,
    time_vals_ms: list[float],
    space_vals_m: list[float],
    seeds: list[int],
    max_tcp: float,
    use_tube: str,
    perturb_alpha_min: float,
    label: str,
    hit_threshold: float = 0.12,
    script: str = "run_tcp_limit_experiment.py",
    ablation: str | None = None,
) -> dict:
    results = {"time": {}, "space": {}}
    n_time = len(time_vals_ms)
    n_space = len(space_vals_m)
    total = (n_time + n_space) * len(seeds)
    count = 0
    t_start = time.perf_counter()

    for t_ms in time_vals_ms:
        results["time"][t_ms] = []
        for s in seeds:
            count += 1
            r = run_one(ball_speed, s, t_ms, 0.0, max_tcp, use_tube, perturb_alpha_min,
                        hit_threshold=hit_threshold, script=script, ablation=ablation)
            results["time"][t_ms].append(r)
            h = r["hit_type"]
            d = f"{r['min_dist']:.3f}m" if r["min_dist"] else "?"
            elapsed = time.perf_counter() - t_start
            eta = elapsed / count * (total - count) if count > 0 else 0
            print(f"  [{label}][{count}/{total}] time={t_ms:>+5.0f}ms seed={s:>3} "
                  f"{h:>7} dist={d}  ETA={eta/60:.0f}min", flush=True)

    for s_m in space_vals_m:
        results["space"][s_m] = []
        for s in seeds:
            count += 1
            r = run_one(ball_speed, s, 0.0, s_m, max_tcp, use_tube, perturb_alpha_min,
                        hit_threshold=hit_threshold, script=script, ablation=ablation)
            results["space"][s_m].append(r)
            h = r["hit_type"]
            d = f"{r['min_dist']:.3f}m" if r["min_dist"] else "?"
            elapsed = time.perf_counter() - t_start
            eta = elapsed / count * (total - count) if count > 0 else 0
            print(f"  [{label}][{count}/{total}] space={s_m:>+6.2f}m seed={s:>3} "
                  f"{h:>7} dist={d}  ETA={eta/60:.0f}min", flush=True)

    return results


def summarize(results: dict, seeds: list[int]) -> dict:
    summary = {"time": {}, "space": {}}
    for key, vals in results["time"].items():
        hits = [r["hit"] for r in vals]
        dists = [r["min_dist"] for r in vals if r["min_dist"] is not None]
        summary["time"][key] = {
            "hit_rate": sum(hits) / len(hits),
            "hits": sum(hits),
            "total": len(hits),
            "avg_dist_mm": float(np.mean(dists) * 1000) if dists else None,
        }
    for key, vals in results["space"].items():
        hits = [r["hit"] for r in vals]
        dists = [r["min_dist"] for r in vals if r["min_dist"] is not None]
        summary["space"][key] = {
            "hit_rate": sum(hits) / len(hits),
            "hits": sum(hits),
            "total": len(hits),
            "avg_dist_mm": float(np.mean(dists) * 1000) if dists else None,
        }
    return summary


def print_comparison(
    tube_sum: dict, notube_sum: dict,
    time_vals: list[float], space_vals: list[float],
    hit_threshold: float = 0.12,
):
    print(f"\n{'='*80}")
    print(f"  双向时间扰动: Tube vs No-Tube (命中阈值={hit_threshold*100:.0f}cm)")
    print(f"{'='*80}")
    print(f"  {'扰动':>8s} | {'No-Tube':^20s} | {'Tube':^20s} | {'Δ命中率':>8s} | {'Δ距离':>8s}")
    print(f"  {'':>8s} | {'命中率':>8s} {'距离':>10s} | {'命中率':>8s} {'距离':>10s} | {'':>8s} | {'':>8s}")
    print(f"  {'-'*76}")
    for t in time_vals:
        nt = notube_sum["time"][t]
        tb = tube_sum["time"][t]
        d_hr = tb["hit_rate"] - nt["hit_rate"]
        d_dist = (tb["avg_dist_mm"] or 0) - (nt["avg_dist_mm"] or 0)
        nt_s = f"{nt['hits']}/{nt['total']} {nt['hit_rate']:.0%}"
        tb_s = f"{tb['hits']}/{tb['total']} {tb['hit_rate']:.0%}"
        nt_d = f"{nt['avg_dist_mm']:.1f}mm" if nt['avg_dist_mm'] else "?"
        tb_d = f"{tb['avg_dist_mm']:.1f}mm" if tb['avg_dist_mm'] else "?"
        print(f"  {t:>+6.0f}ms | {nt_s:>8s} {nt_d:>10s} | {tb_s:>8s} {tb_d:>10s} | {d_hr:>+7.0%} | {d_dist:>+7.1f}mm")

    print(f"\n{'='*80}")
    print(f"  双向空间扰动: Tube vs No-Tube (命中阈值={hit_threshold*100:.0f}cm)")
    print(f"{'='*80}")
    print(f"  {'扰动':>8s} | {'No-Tube':^20s} | {'Tube':^20s} | {'Δ命中率':>8s} | {'Δ距离':>8s}")
    print(f"  {'':>8s} | {'命中率':>8s} {'距离':>10s} | {'命中率':>8s} {'距离':>10s} | {'':>8s} | {'':>8s}")
    print(f"  {'-'*76}")
    for s in space_vals:
        nt = notube_sum["space"][s]
        tb = tube_sum["space"][s]
        d_hr = tb["hit_rate"] - nt["hit_rate"]
        d_dist = (tb["avg_dist_mm"] or 0) - (nt["avg_dist_mm"] or 0)
        nt_s = f"{nt['hits']}/{nt['total']} {nt['hit_rate']:.0%}"
        tb_s = f"{tb['hits']}/{tb['total']} {tb['hit_rate']:.0%}"
        nt_d = f"{nt['avg_dist_mm']:.1f}mm" if nt['avg_dist_mm'] else "?"
        tb_d = f"{tb['avg_dist_mm']:.1f}mm" if tb['avg_dist_mm'] else "?"
        s_label = f"{s*100:>+6.0f}cm"
        print(f"  {s_label:>8s} | {nt_s:>8s} {nt_d:>10s} | {tb_s:>8s} {tb_d:>10s} | {d_hr:>+7.0%} | {d_dist:>+7.1f}mm")


def main() -> None:
    parser = argparse.ArgumentParser(description="双向扰动实验: Tube vs No-Tube")
    parser.add_argument("--ball-speed", type=float, default=7)
    parser.add_argument("--max-tcp", type=float, default=1.8)
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--perturb-alpha-min", type=float, default=0.3)
    parser.add_argument("--hit-threshold", type=float, default=0.12,
                        help="命中判定阈值 (m), 默认 0.12m=12cm")
    parser.add_argument("--script", type=str, default="run_tcp_limit_experiment.py",
                        help="实验入口脚本 (v1 or v2)")
    parser.add_argument("--ablation", type=str, default=None,
                        help="消融模式: 'sigma-only'(禁用softmin)")
    args = parser.parse_args()

    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    time_vals = [-200, -100, 0, 100, 200]
    space_vals = [-0.20, -0.10, 0.0, 0.10, 0.20]

    scr = args.script
    ht = args.hit_threshold
    print(f"双向扰动实验: ball_speed={args.ball_speed}m/s, TCP<={args.max_tcp}m/s")
    print(f"  脚本: {scr}")
    print(f"  命中阈值: {ht*100:.0f}cm ({ht}m)")
    print(f"  时间扰动: {time_vals} ms")
    print(f"  空间扰动: {[f'{v*100:.0f}cm' for v in space_vals]}")
    print(f"  Seeds: {args.n_seeds}, alpha_min={args.perturb_alpha_min}")
    if args.ablation:
        print(f"  消融模式: {args.ablation}")
    print(f"  总次数: {len(time_vals)*args.n_seeds*2 + len(space_vals)*args.n_seeds*2} = "
           f"(5×{args.n_seeds}×2 时间 + 5×{args.n_seeds}×2 空间)")

    tube_results = run_batch(
        args.ball_speed, time_vals, space_vals, seeds,
        args.max_tcp, "true", args.perturb_alpha_min, "Tube",
        hit_threshold=ht, script=scr, ablation=args.ablation,
    )

    notube_results = run_batch(
        args.ball_speed, time_vals, space_vals, seeds,
        args.max_tcp, "false", args.perturb_alpha_min, "NoTube",
        hit_threshold=ht, script=scr, ablation=args.ablation,
    )

    tube_sum = summarize(tube_results, seeds)
    notube_sum = summarize(notube_results, seeds)

    print_comparison(tube_sum, notube_sum, time_vals, space_vals, hit_threshold=ht)

    out_dir = CWD / "results"
    out_dir.mkdir(exist_ok=True)
    tag = f"bidir_N{args.n_seeds}_amin{args.perturb_alpha_min}"
    np.savez(
        str(out_dir / f"experiment_d_{tag}.npz"),
        time_vals=time_vals, space_vals_m=space_vals, seeds=seeds,
        ball_speed=args.ball_speed, perturb_alpha_min=args.perturb_alpha_min,
        allow_pickle=True,
    )
    for prefix, res in [("tube", tube_results), ("notube", notube_results)]:
        for kind in ["time", "space"]:
            for key, vals in res[kind].items():
                tag_key = f"{kind}_{key}".replace(".", "p").replace("-", "m")
                out_dir.mkdir(exist_ok=True)

    json_out = {
        "ball_speed": args.ball_speed,
        "max_tcp": args.max_tcp,
        "perturb_alpha_min": args.perturb_alpha_min,
        "hit_threshold": ht,
        "n_seeds": args.n_seeds,
        "tube": tube_sum,
        "notube": notube_sum,
    }
    ver_tag = "v2" if "v2" in scr else "v1"
    abl_tag = f"_{args.ablation}" if args.ablation else ""
    json_path = out_dir / f"experiment_bidir_{ver_tag}{abl_tag}_v{args.ball_speed}_N{args.n_seeds}_ht{int(ht*100)}cm.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_out, f, indent=2, ensure_ascii=False)
    print(f"\nJSON: {json_path}")


if __name__ == "__main__":
    main()
