"""逐 seed 对比 v3 softmin_only vs v6 full_v6 命中率差异"""
import json, numpy as np

d6 = json.load(open("results/exp_ablation_v6_20260604/raw_data.json"))
d3 = json.load(open("results/exp_ablation_corridor_20260602/raw_data.json"))

both_hit = 0; both_miss = 0; v3_only = 0; v6_only = 0
print("seed  v3_hit  v3_pos    v6_hit  v6_pos    t_pert      s_pert")
for i in range(50):
    r3 = d3["softmin_only"][i]
    r6 = d6["full_v6"][i]
    h3 = r3["result"] is not None and r3["result"].get("hit_type", "") in ("active", "passive")
    h6 = r6["result"] is not None and r6["result"].get("hit_type", "") in ("active", "passive")
    p3 = r3["result"].get("pos_error", -1) if r3["result"] else -1
    p6 = r6["result"].get("pos_error", -1) if r6["result"] else -1
    t = r3.get("time_perturb_ms", 0)
    s = r3.get("space_perturb_m", 0) * 100

    if h3 and h6:
        both_hit += 1
    elif not h3 and not h6:
        both_miss += 1
    elif h3 and not h6:
        v3_only += 1
    else:
        v6_only += 1

    # 只打印差异 seed 或关键失败
    if h3 != h6 or (not h6 and h3):
        tag3 = "HIT" if h3 else "MISS"
        tag6 = "HIT" if h6 else "MISS"
        print(f"  {i:2d}   {tag3:4s}  {p3*100:6.1f}cm  {tag6:4s}  {p6*100:6.1f}cm  {t:+7.1f}ms  {s:+6.1f}cm")

print(f"\nBoth hit={both_hit}  Both miss={both_miss}  v3_only={v3_only}  v6_only={v6_only}")
total = both_hit + both_miss + v3_only + v6_only
print(f"v3 hit rate={(both_hit+v3_only)}/{total} = {(both_hit+v3_only)/total*100:.0f}%")
print(f"v6 hit rate={(both_hit+v6_only)}/{total} = {(both_hit+v6_only)/total*100:.0f}%")

# 分析 v3_only 的 seed 特征：大时间扰动？大空间扰动？
v3_seeds = []
for i in range(50):
    r3 = d3["softmin_only"][i]
    r6 = d6["full_v6"][i]
    h3 = r3["result"] is not None and r3["result"].get("hit_type", "") in ("active", "passive")
    h6 = r6["result"] is not None and r6["result"].get("hit_type", "") in ("active", "passive")
    if h3 and not h6:
        t = r3.get("time_perturb_ms", 0)
        s = r3.get("space_perturb_m", 0) * 100
        p6 = r6["result"].get("pos_error", -1) if r6["result"] else -1
        v3_seeds.append((i, t, s, p6*100))

if v3_seeds:
    ts = [x[1] for x in v3_seeds]
    ss = [x[2] for x in v3_seeds]
    ps = [x[3] for x in v3_seeds]
    print(f"\nv3_only seeds ({len(v3_seeds)}): |t_pert| avg={np.mean(np.abs(ts)):.0f}ms, |s_pert| avg={np.mean(np.abs(ss)):.1f}cm")
    print(f"  v6 pos_error on these seeds: avg={np.mean(ps):.1f}cm (just above 12cm threshold)")

# 对比 tube_ready_ms
tr3 = [r["result"].get("tube_ready_ms", 0) for r in d3["softmin_only"] if r["result"] is not None]
tr6 = [r["result"].get("tube_ready_ms", 0) for r in d6["full_v6"] if r["result"] is not None]
bn3 = [r["result"].get("ball_near_ms", 0) for r in d3["softmin_only"] if r["result"] is not None]
bn6 = [r["result"].get("ball_near_ms", 0) for r in d6["full_v6"] if r["result"] is not None]
print(f"\ntube_ready_ms:  v3={np.mean(tr3):.0f}±{np.std(tr3):.0f}  v6={np.mean(tr6):.0f}±{np.std(tr6):.0f}")
print(f"ball_near_ms:   v3={np.mean(bn3):.0f}±{np.std(bn3):.0f}  v6={np.mean(bn6):.0f}±{np.std(bn6):.0f}")

# hit_time_error
hte3 = [r["result"].get("hit_time_error_ms", 0) for r in d3["softmin_only"] if r["result"] is not None and r["result"].get("hit_type","") in ("active","passive")]
hte6 = [r["result"].get("hit_time_error_ms", 0) for r in d6["full_v6"] if r["result"] is not None and r["result"].get("hit_type","") in ("active","passive")]
print(f"hit_time_error_ms: v3={np.mean(hte3):.0f}±{np.std(hte3):.0f}  v6={np.mean(hte6):.0f}±{np.std(hte6):.0f}")
