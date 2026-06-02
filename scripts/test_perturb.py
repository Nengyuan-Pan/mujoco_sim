"""Tube vs no-tube perturbation test."""
import subprocess, sys, re

perturbs = [-30, -20, -10, 0, 10, 20, 30]
print("{:>6} | {:>7} | {:>8} | {:>6} | {:>7} | {:>6} | {:>10}".format(
    "pert", "mode", "min_dist", "bnear", "t_ready", "result", "pos_err"))
print("-" * 75)
for perturb_ms in perturbs:
    for use_tube in ['false', 'true']:
        tag = 'TUBE' if use_tube == 'true' else 'NO-TUBE'
        cmd = [sys.executable, 'scripts/rm65_mpc_tube.py',
               '--use_tube', use_tube, '--seed', '42',
               '--window-ms', '50', '--no-plot',
               '--time-perturb-ms', str(perturb_ms)]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd='E:/tennis_robot/mujoco_sim')
        stdout = result.stdout
        m = re.search(r'最小球拍-球距离:\s+([\d.]+)\s+m', stdout)
        min_dist = float(m.group(1)) if m else -1
        m2 = re.search(r'ball_near\s+步数:\s+(\d+)', stdout)
        ball_near = int(m2.group(1)) if m2 else -1
        m3 = re.search(r'tube_ready\s+步数:\s+(\d+)', stdout)
        tube_ready = int(m3.group(1)) if m3 else -1
        m4 = re.search(r'位置误差:\s+([\d.]+)\s+m', stdout)
        pos_err = float(m4.group(1)) if m4 else -1

        hit = 'HIT' if min_dist < 0.15 else 'MISS'
        print("{:+6.0f} | {:>7} | {:>7.3f}m | {:>4}st | {:>5}st | {:>6} | {:>9.3f}m".format(
            perturb_ms, tag, min_dist, ball_near, tube_ready, hit, pos_err))
