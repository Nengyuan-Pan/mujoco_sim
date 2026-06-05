"""对比 v3 vs v6 的 per-seed 关键指标差异"""
import json, numpy as np

d6 = json.load(open("results/exp_ablation_v6_20260604/raw_data.json"))
d3 = json.load(open("results/exp_ablation_corridor_20260602/raw_data.json"))

# 统计 v3_only seeds 上的关键差异
v3_only_seeds = []
for i in range(50):
    r3 = d3["softmin_only"][i]
    r6 = d6["full_v6"][i]
    h3 = r3["result"] is not None and r3["result"].get("hit_type", "") in ("active", "passive")
    h6 = r6["result"] is not None and r6["result"].get("hit_type", "") in ("active", "passive")
    if h3 and not h6:
        v3_only_seeds.append(i)

# 分析 1: ball_near_ms 差异（球到达附近的时间）
bn3 = [d3["softmin_only"][i]["result"].get("ball_near_ms", 0) for i in v3_only_seeds]
bn6 = [d6["full_v6"][i]["result"].get("ball_near_ms", 0) for i in v3_only_seeds]
print("=== v3_only seeds 分析 ===")
print(f"ball_near_ms:  v3={np.mean(bn3):.0f}±{np.std(bn3):.0f}  v6={np.mean(bn6):.0f}±{np.std(bn6):.0f}")

# 分析 2: tube_ready_ms 差异
tr3 = [d3["softmin_only"][i]["result"].get("tube_ready_ms", 0) for i in v3_only_seeds]
tr6 = [d6["full_v6"][i]["result"].get("tube_ready_ms", 0) for i in v3_only_seeds]
print(f"tube_ready_ms: v3={np.mean(tr3):.0f}±{np.std(tr3):.0f}  v6={np.mean(tr6):.0f}±{np.std(tr6):.0f}")

# 分析 3: max_qdot 差异（关节速度利用率）
mq3 = [d3["softmin_only"][i]["result"].get("max_qdot", 0) for i in v3_only_seeds]
mq6 = [d6["full_v6"][i]["result"].get("max_qdot", 0) for i in v3_only_seeds]
print(f"max_qdot:      v3={np.mean(mq3):.2f}±{np.std(mq3):.2f}  v6={np.mean(mq6):.2f}±{np.std(mq6):.2f}")

# 分析 4: max_tcp 差异
mt3 = [d3["softmin_only"][i]["result"].get("max_tcp", 0) for i in v3_only_seeds]
mt6 = [d6["full_v6"][i]["result"].get("max_tcp", 0) for i in v3_only_seeds]
print(f"max_tcp:       v3={np.mean(mt3):.2f}±{np.std(mt3):.2f}  v6={np.mean(mt6):.2f}±{np.std(mt6):.2f}")

# 分析 5: v6 tube_ready=0 的 seed 有多少个
tube_zero_v6 = sum(1 for i in v3_only_seeds if d6["full_v6"][i]["result"].get("tube_ready_ms", 0) == 0)
tube_zero_v3 = sum(1 for i in v3_only_seeds if d3["softmin_only"][i]["result"].get("tube_ready_ms", 0) == 0)
print(f"\nv6 tube_ready=0: {tube_zero_v6}/{len(v3_only_seeds)} seeds")
print(f"v3 tube_ready=0: {tube_zero_v3}/{len(v3_only_seeds)} seeds")

# 分析 6: v6 ball_near=0 的 seed 有多少个
bn_zero_v6 = sum(1 for i in v3_only_seeds if d6["full_v6"][i]["result"].get("ball_near_ms", 0) == 0)
bn_zero_v3 = sum(1 for i in v3_only_seeds if d3["softmin_only"][i]["result"].get("ball_near_ms", 0) == 0)
print(f"v6 ball_near=0:  {bn_zero_v6}/{len(v3_only_seeds)} seeds")
print(f"v3 ball_near=0:  {bn_zero_v3}/{len(v3_only_seeds)} seeds")

# 分析 7: 看 v3 全部 seeds 的 tube_ready_ms 分布
all_tr3 = [d3["softmin_only"][i]["result"].get("tube_ready_ms", 0)
           for i in range(50) if d3["softmin_only"][i]["result"] is not None]
all_tr6 = [d6["full_v6"][i]["result"].get("tube_ready_ms", 0)
           for i in range(50) if d6["full_v6"][i]["result"] is not None]
print(f"\n全部 seeds tube_ready_ms 分布:")
print(f"  v3: {np.mean(all_tr3):.0f}±{np.std(all_tr3):.0f}ms  range=[{min(all_tr3):.0f}, {max(all_tr3):.0f}]  zero={sum(1 for x in all_tr3 if x==0)}/{len(all_tr3)}")
print(f"  v6: {np.mean(all_tr6):.0f}±{np.std(all_tr6):.0f}ms  range=[{min(all_tr6):.0f}, {max(all_tr6):.0f}]  zero={sum(1 for x in all_tr6 if x==0)}/{len(all_tr6)}")

# 分析 8: 看 v6 的 terminal exempt 和 follow_through PD 是否在扰动下导致问题
# 看 v6 miss seed 的 pos_error — 是否因为 arm 被 PD 随挥拉走了？
print(f"\nv6 miss seeds pos_error 分布:")
pe6 = [d6["full_v6"][i]["result"]["pos_error"] * 100 for i in v3_only_seeds]
for i, pe in zip(v3_only_seeds, pe6):
    print(f"  seed {i:2d}: {pe:5.1f}cm")
