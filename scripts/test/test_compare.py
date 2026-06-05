"""Time vs Space perturbation comparison table."""
import subprocess, sys, re

def run_test(perturb_type, value, use_tube):
    flag = '--time-perturb-ms' if perturb_type == 'time' else '--space-perturb-m'
    tube_str = 'true' if use_tube else 'false'
    cmd = [sys.executable, 'scripts/rm65_mpc_tube.py',
           '--use_tube', tube_str, '--seed', '42',
           '--window-ms', '50', '--no-plot',
           flag, str(value)]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd='E:/tennis_robot/mujoco_sim')
    stdout = result.stdout
    m = re.search(r'最小球拍-球距离:\s+([\d.]+)\s+m', stdout)
    min_dist = float(m.group(1)) if m else -1
    m2 = re.search(r'ball_near\s+步数:\s+(\d+)', stdout)
    ball_near = int(m2.group(1)) if m2 else -1
    m3 = re.search(r'tube_ready\s+步数:\s+(\d+)', stdout)
    tube_ready = int(m3.group(1)) if m3 else -1
    return min_dist, ball_near, tube_ready

# ============ Time perturbation test ============
print("=" * 85)
print("  Time perturbation (--time-perturb-ms): MPC thinks ball arrives earlier/later")
print("=" * 85)
print("{:>8} | {:>6} {:>7} | {:>6} {:>7} | {:>12} | {:>12}".format(
    "perturb", "NO", "min_d", "TUBE", "min_d", "tube_ready", "result"))
print("-" * 85)
for perturb_ms in [-30, -20, -10, 0, 10, 20, 30]:
    nd, nb, _ = run_test('time', perturb_ms, False)
    td, tb, tr = run_test('time', perturb_ms, True)
    nh = 'HIT' if nd < 0.15 else 'MISS'
    th = 'HIT' if td < 0.15 else 'MISS'
    better = 'TUBE>' if td < nd - 0.002 else 'NO>' if nd < td - 0.002 else '=='
    print("{:+8.0f}ms | {:>4} {:>5.3f}m | {:>4} {:>5.3f}m | {:>5}st {:>5} | {:>5} {:>5} {}".format(
        perturb_ms, ' ', nd, ' ', td, tr, better, nh, th, 
        '***' if th == 'HIT' and nh == 'MISS' else ''))

# ============ Space perturbation test ============
print()
print("=" * 85)
print("  Space perturbation (--space-perturb-m): p_hit is offset laterally")
print("=" * 85)
print("{:>8} | {:>6} {:>7} | {:>6} {:>7} | {:>12} | {:>12}".format(
    "offset", "NO", "min_d", "TUBE", "min_d", "tube_ready", "result"))
print("-" * 85)
for offset_m in [-0.10, -0.08, -0.06, -0.04, -0.02, 0.0, 0.02, 0.04, 0.06, 0.08, 0.10]:
    nd, nb, _ = run_test('space', offset_m, False)
    td, tb, tr = run_test('space', offset_m, True)
    nh = 'HIT' if nd < 0.15 else 'MISS'
    th = 'HIT' if td < 0.15 else 'MISS'
    better = 'TUBE>' if td < nd - 0.002 else 'NO>' if nd < td - 0.002 else '=='
    star = '***' if th == 'HIT' and nh == 'MISS' else ''
    print("{:+8.3f}m | {:>4} {:>5.3f}m | {:>4} {:>5.3f}m | {:>5}st {:>5} | {:>5} {:>5} {}".format(
        offset_m, ' ', nd, ' ', td, tr, better, nh, th, star))

print()
print("*** = TUBE saved a case that NO-TUBE missed")
