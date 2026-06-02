"""运行 TCP 限制实验并输出结果。"""
import subprocess, re, sys, json
import numpy as np

SPEEDS = [7, 8, 9, 10]
N_SEEDS = 20
MAX_TCP = 1.8

results = {}

for speed in SPEEDS:
    print(f"=== ball speed {speed} m/s, TCP <= {MAX_TCP} m/s ===")
    seeds = []
    for s in range(N_SEEDS):
        try:
            out = subprocess.run(
                [sys.executable, "scripts/run_tcp_limit_experiment.py",
                 "--ball-speed", str(speed), "--seed", str(s), "--max-tcp", str(MAX_TCP)],
                capture_output=True, timeout=300)
            txt = out.stdout.decode("gbk", errors="replace") + out.stderr.decode("gbk", errors="replace")
        except subprocess.TimeoutExpired:
            seeds.append({"seed": s, "hit": False, "hit_type": "timeout"})
            print(f"  seed={s:2d} timeout")
            continue

        r = {"seed": s, "hit": False, "hit_type": "miss", "max_qdot": None, "max_tcp": None, "min_dist": None}
        for line in txt.split("\n"):
            st = line.strip()
            if "RM-65" in st:
                if "5cm" in st:
                    r["hit"] = True; r["hit_type"] = "PRECISE"
                elif "0.153" in st:
                    r["hit"] = True; r["hit_type"] = "HIT"
                elif "10cm" in st:
                    r["hit_type"] = "near"
                else:
                    r["hit_type"] = "miss"
            if "max_qdot=" in st:
                m = re.search(r"max_qdot=([\d.]+)x", st)
                if m: r["max_qdot"] = float(m.group(1))
                m2 = re.search(r"max_tcp=([\d.]+)m/s", st)
                if m2: r["max_tcp"] = float(m2.group(1))
            if st.endswith("m") and r["min_dist"] is None:
                m = re.search(r"([\d.]+)\s*m$", st)
                if m:
                    v = float(m.group(1))
                    if 0.001 < v < 0.3:
                        r["min_dist"] = v

        seeds.append(r)
        h = r["hit_type"]
        d = f"{r['min_dist']:.3f}m" if r["min_dist"] else "?"
        q = f"{r['max_qdot']:.2f}x" if r["max_qdot"] else "?"
        t = f"{r['max_tcp']:.1f}" if r["max_tcp"] else "?"
        print(f"  seed={s:2d} {h:7s} dist={d} qdot={q} tcp={t}m/s")

    n_hit = sum(1 for r in seeds if r["hit"])
    dists = [r["min_dist"] for r in seeds if r["min_dist"]]
    qdots = [r["max_qdot"] for r in seeds if r["max_qdot"]]
    tcps = [r["max_tcp"] for r in seeds if r["max_tcp"]]
    results[speed] = {
        "n_hit": n_hit, "n_seeds": N_SEEDS,
        "hit_rate": n_hit / N_SEEDS,
        "avg_dist": float(np.mean(dists)) if dists else None,
        "avg_qdot": float(np.mean(qdots)) if qdots else None,
        "peak_qdot": float(np.max(qdots)) if qdots else None,
        "avg_tcp": float(np.mean(tcps)) if tcps else None,
        "peak_tcp": float(np.max(tcps)) if tcps else None,
    }

    ad = f"{results[speed]['avg_dist']*1000:.1f}mm" if results[speed]["avg_dist"] else "?"
    aq = f"{results[speed]['avg_qdot']:.2f}x" if results[speed]["avg_qdot"] else "?"
    at = f"{results[speed]['avg_tcp']:.1f}m/s" if results[speed]["avg_tcp"] else "?"
    print(f"  >>> {n_hit}/{N_SEEDS} ({n_hit/N_SEEDS*100:.0f}%) avg_dist={ad} avg_qdot={aq} avg_tcp={at}")
    print()

with open("results/experiment_tcp_limit.json", "w") as f:
    json.dump(results, f, indent=2, default=str)

print("=== SUMMARY (TCP <= 1.8 m/s + qdot <= 1.0x) ===")
for sp in SPEEDS:
    r = results[sp]
    ad = f"{r['avg_dist']*1000:.1f}mm" if r["avg_dist"] else "?"
    aq = f"{r['avg_qdot']:.2f}x" if r["avg_qdot"] else "?"
    at = f"{r['avg_tcp']:.1f}m/s" if r["avg_tcp"] else "?"
    print(f"  {sp} m/s: {r['n_hit']}/{r['n_seeds']} ({r['hit_rate']*100:.0f}%) dist={ad} qdot={aq} tcp={at}")
