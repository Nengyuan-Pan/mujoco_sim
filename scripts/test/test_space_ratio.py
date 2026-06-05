"""Space perturbation test with custom tube_cost_ratio."""
import subprocess, sys, re

offsets = [-0.10, -0.06, -0.02, 0.0, 0.04, 0.08, 0.10]
ratios = [0.3, 0.5, 0.7]
print("{:>8} | {:>5} | {:>7} | {:>8} | {:>6} | {:>7} | {:>6}".format(
    "offset", "ratio", "mode", "min_dist", "bnear", "t_ready", "result"))
print("-" * 75)
for offset_m in offsets:
    for ratio in ratios:
        for use_tube in ['false', 'true']:
            if use_tube == 'false' and ratio != ratios[0]:
                continue  # NO-TUBE same for all ratios
            tag = 'TUBE' if use_tube == 'true' else 'NO-TUBE'
            cmd = [sys.executable, 'scripts/rm65_mpc_tube.py',
                   '--use_tube', use_tube, '--seed', '42',
                   '--window-ms', '50', '--no-plot',
                   '--space-perturb-m', str(offset_m),
                   '--tube-cost-ratio', str(ratio)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd='E:/tennis_robot/mujoco_sim')
            stdout = result.stdout
            m = re.search(r'最小球拍-球距离:\s+([\d.]+)\s+m', stdout)
            min_dist = float(m.group(1)) if m else -1
            m2 = re.search(r'ball_near\s+步数:\s+(\d+)', stdout)
            ball_near = int(m2.group(1)) if m2 else -1
            m3 = re.search(r'tube_ready\s+步数:\s+(\d+)', stdout)
            tube_ready = int(m3.group(1)) if m3 else -1

            hit = 'HIT' if min_dist < 0.15 else 'MISS'
            ratio_str = f"{ratio:.1f}" if use_tube == 'true' else "N/A"
            print("{:+8.3f} | {:>5} | {:>7} | {:>7.3f}m | {:>4}st | {:>5}st | {:>6}".format(
                offset_m, ratio_str, tag, min_dist, ball_near, tube_ready, hit))
