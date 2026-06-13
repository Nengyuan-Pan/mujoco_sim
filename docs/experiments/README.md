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
| 2026-06-04 | [exp7 v1/v2 噪声×Tube 消融](2026-06-04_exp7_noise_tube_ablation.md) | 8-20 | 豁免+噪声 | 0-74% | lo~2%；v2一致性修复无改善 |
| 2026-06-05 | [exp7 v3 噪声×Tube 消融](2026-06-05_exp7_noise_tube_ablation.md) | 6-15 | 豁免+噪声 | 0-94% | 6-15m/s+50seeds+anis坐标修正；lo仍~1%；噪声容忍度是深层框架问题 |
| 2026-06-05 | [V8 连续20次击打](2026-06-05_exp_v8_20hits_continuous.md) | 7 | TCP≤1.8 | 100% | V8 连续20次全部主动命中，误差均值53mm |
| 2026-06-05 | [V8 扰动消融](2026-06-05_exp_v8_perturb_ablation.md) | 7 | TCP≤1.8+扰动 | 4-100% | Tube+Softmin缺一不可；300ms/15cm扰动下default=92%，notube=4% |
| 2026-06-05 | [V9 扰动对比](2026-06-05_exp_v9_perturb_robustness.md) | 7 | TCP≤1.8+扰动 | 0-94% | V9 default=94%略优于V8的92%；nosoftmin退化至6% |
| 2026-06-06 | [V9 消融分析](../v9_ablation_analysis.md) | 7 | TCP≤1.8+扰动 | 45-100% | Tube+Softmin协同：无扰动100%，扰动57%；随挥无独立贡献 |
| 2026-06-07 | [V9 消融（解耦）](../v9_ablation_analysis_decoupled.md) | 7 | TCP≤1.8+扰动 | 45-94% | Softmin 是主要贡献者（94%→86%），Tube 仅+2pp，随挥无效果 |
| 2026-06-07 | [exp8 KF恢复](2026-06-07_exp8_estimator_recovery.md) | 6-15 | 豁免+噪声+KF | 0-81% | KF lo噪声恢复+12pp（17%）；off性能税-20pp；anis恢复意外好 |
| 2026-06-09 | [exp7 v4 preprocessor重跑](2026-06-09_exp7_noise_tube_ablation_v4.md) | 6-15 | 豁免+噪声 | 0-94% | off与v1完全一致(72.2%/81.4%)；lo从1.2%升至4.9%；preprocessor架构验证通过 |
| 2026-06-10 | [exp8 v2 KF恢复](2026-06-10_exp8_estimator_recovery_v2.md) | 6-15 | 豁免+噪声+KF | 0-94% | off性能税0pp(已消除)；lo KF负恢复-5pp；mid/hi/anis无显著恢复 |
| 2026-06-10 | [exp9 观测频率鲁棒性](2026-06-10_exp9_obs_freq_robustness.md) | 6-15 | 豁免+观测门控+噪声+KF | 0-94% | off频率退化0pp(200→10Hz)；lo KF恢复+9.4pp→+1.0pp随频率衰减；低频KF负恢复 |
| 2026-06-13 | [exp10 PD精调全网格](2026-06-13_exp10_pd_finetune.md) | 7 | 位置模式 | 74.7-90.5% | uniform Kp最优(90.5%)；选定Kp=20/KdR=0.08→100%命中/52mm误差；3360 runs |
