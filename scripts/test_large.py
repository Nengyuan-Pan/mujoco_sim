"""Test 30 random seeds for hit rate."""
import subprocess, sys, re, random

random.seed(12345)
seeds = random.sample(range(10000), 30)

hits_tube = 0
hits_notube = 0
total_tube = 0
total_notube = 0

for seed in seeds:
    for use_tube in ['true', 'false']:
        tag = 'TUBE' if use_tube == 'true' else 'NONE'
        cmd = [sys.executable, 'scripts/rm65_mpc_tube.py',
               '--use_tube', use_tube, '--seed', str(seed),
               '--window-ms', '50', '--no-plot']
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd='E:/tennis_robot/mujoco_sim')
        m = re.search(r'最小球拍-球距离:\s+([\d.]+)', result.stdout)
        min_dist = float(m.group(1)) if m else 999
        hits = 1 if min_dist < 0.15 else 0
        if use_tube == 'true':
            hits_tube += hits
            total_tube += 1
        else:
            hits_notube += hits
            total_notube += 1
        status = 'HIT' if hits else 'MISS'
        if not hits:
            print(f"seed={seed:>5}  {tag:>4}  {status:>4}  min_dist={min_dist:.3f}m")

print(f"\nTUBE:   {hits_tube}/{total_tube} ({100*hits_tube/total_tube:.0f}%)")
print(f"NO-TUBE: {hits_notube}/{total_notube} ({100*hits_notube/total_notube:.0f}%)")
