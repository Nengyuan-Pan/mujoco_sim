"""对比 v3 vs v6 在 v3_only seed 上的详细指标"""
import json, numpy as np

d6 = json.load(open("results/exp_ablation_v6_20260604/raw_data.json"))
d3 = json.load(open("results/exp_ablation_corridor_20260602/raw_data.json"))

v3_only_seeds = []
for i in range(50):
    r3 = d3["softmin_only"][i]
    r6 = d6["full_v6"][i]
    h3 = r3["result"] is not None and r3["result"].get("hit_type", "") in ("active", "passive")
    h6 = r6["result"] is not None and r6["result"].get("hit_type", "") in ("active", "passive")
    if h3 and not h6:
        v3_only_seeds.append(i)

print("seed  | v3: tube_ready ball_near  pos   hte  | v6: tube_ready ball_near  pos   hte  | t_pert    s_pert")
for i in v3_only_seeds:
    r3 = d3["softmin_only"][i]["result"]
    r6 = d6["full_v6"][i]["result"]
    t = d3["softmin_only"][i].get("time_perturb_ms", 0)
    s = d3["softmin_only"][i].get("space_perturb_m", 0) * 100
    print(f"  {i:2d}  |     {r3.get('tube_ready_ms',0):5.0f}    {r3.get('ball_near_ms',0):5.0f}   {r3['pos_error']*100:5.1f}  {r3.get('hit_time_error_ms',0):4.0f}  |     {r6.get('tube_ready_ms',0):5.0f}    {r6.get('ball_near_ms',0):5.0f}   {r6['pos_error']*100:5.1f}  {r6.get('hit_time_error_ms',0):4.0f}  | {t:+7.1f}ms {s:+6.1f}cm")

# 关键：v3 的 tube_ready 明显更短？
tr3 = [d3["softmin_only"][i]["result"].get("tube_ready_ms", 0) for i in v3_only_seeds]
tr6 = [d6["full_v6"][i]["result"].get("tube_ready_ms", 0) for i in v3_only_seeds]
print(f"\ntube_ready on v3_only seeds: v3={np.mean(tr3):.0f}±{np.std(tr3):.0f}ms  v6={np.mean(tr6):.0f}±{np.std(tr6):.0f}ms")

# 看 v6 是否有更短的可用规划时间
plan_time3 = [d3["softmin_only"][i]["result"].get("ball_near_ms", 0) - d3["softmin_only"][i]["result"].get("tube_ready_ms", 0) for i in v3_only_seeds]
plan_time6 = [d6["full_v6"][i]["result"].get("ball_near_ms", 0) - d6["full_v6"][i]["result"].get("tube_ready_ms", 0) for i in v3_only_seeds]
print(f"near-ready gap: v3={np.mean(plan_time3):.0f}±{np.std(plan_time3):.0f}ms  v6={np.mean(plan_time6):.0f}±{np.std(plan_time6):.0f}ms")
