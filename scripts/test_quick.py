"""Quick baseline test."""
import subprocess, sys, re
cmd = [sys.executable, 'scripts/rm65_mpc_tube.py', '--use_tube', 'true', '--seed', '42', '--window-ms', '50', '--no-plot']
result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd='E:/tennis_robot/mujoco_sim')
for line in result.stdout.split('\n'):
    m = re.search(r'位置误差:\s+([\d.]+)\s+m', line)
    if m: print('pos_err:', m.group(1), 'm')
    m2 = re.search(r'最小球拍-球距离:\s+([\d.]+)\s+m', line)
    if m2: print('min_dist:', m2.group(1), 'm')
    m3 = re.search(r'ball_near\s+步数:\s+(\d+)', line)
    if m3: print('ball_near:', m3.group(1))
    m4 = re.search(r'tube_ready\s+步数:\s+(\d+)', line)
    if m4: print('tube_ready:', m4.group(1))
