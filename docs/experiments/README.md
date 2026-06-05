# 实验记录索引

## 记录规范

1. 每次实验完成后，从 `_template.md` 复制结构填写。
2. 文件命名：`YYYY-MM-DD_expN_<简短描述>.md`
3. 数据源：对应 `experiment_data/expX_*/results.csv`
4. 结论控制在 2-3 句话，标注关键数字。
5. 填写后更新本文件的索引表。

## 索引表

| 日期 | 实验 | 球速 | 约束 | 命中率区间 | 关键结论 |
|------|------|------|------|-----------|---------|
| 2026-06-02 | [exp2 实时严格](2026-06-02_exp2_strict_joint.md) | 9 | 严格 | 0% | 实时脚本在 9m/s 严格约束下全部 miss |
| 2026-06-03 | [exp2 低球速](2026-06-03_exp2_v2_low_speed.md) | 7-8 | 严格 | 80% | 降到 7-8m/s 可命中；serve_box 最低 8m/s |
| 2026-06-03 | [exp1 豁免 bounce](2026-06-03_exp1_algorithm_capability.md) | 8-18 | 豁免 | 40-90% | 算法上限 ~15m/s；全部被动接触 |
| 2026-06-03 | [exp1 v3 no-bounce](2026-06-03_exp1_v3_nobounce_sweep.md) | 8-30 | 豁免 | 0-90% | no-bounce 更难命中；30m/s 归零；15.6min |
| 2026-06-04 | [exp2 v3 严格约束](2026-06-04_exp2_v3_strict_joint.md) | 8-18 | 严格 | 20-80% | 14m/s 断崖；关速峰值 1.65x；0 生成失败；3.6min |
| 2026-06-04 | [exp7 噪声×Tube 消融](2026-06-04_exp7_noise_tube_ablation.md) | 8-20 | 豁免+噪声 | 0-74% | lo/v1/v2均~2%；全管道一致性修复无改善；噪声容忍度是深层框架问题 |
| 2026-06-05 | [V8 连续20次击打](2026-06-05_exp_v8_20hits_continuous.md) | 7 | TCP≤1.8 | 100% | V8 连续20次全部主动命中，误差均值53mm |
| 2026-06-05 | [V8 扰动消融](2026-06-05_exp_v8_perturb_ablation.md) | 7 | TCP≤1.8+扰动 | 4-100% | Tube+Softmin缺一不可；300ms/15cm扰动下default=92%，notube=4% |
| 2026-06-05 | [V9 扰动对比](2026-06-05_exp_v9_perturb_robustness.md) | 7 | TCP≤1.8+扰动 | 0-94% | V9 default=94%略优于V8的92%；nosoftmin退化至6% |
