# Skill: 实验记录（experiment_log）

## 目的
实验完成后，自动从 `results.csv` 提取统计数据，生成标准化实验记录。
Agent 负责数据聚合和关键数字提取，人类负责结论叙述。

## 调用时机
- 用户说"记录实验""保存实验结果""写实验记录""生成实验报告"
- 批量运行完成后用户要求整理数据
- 对已有 `experiment_data/expX_*/` 目录补写记录

## 工作流

### 1. 定位实验目录

**优先级**：
1. 从对话上下文提取最近使用的 `experiment_data/expX_*/` 路径
2. 无法确定时列出所有含 `results.csv` 的目录，让用户选择
3. 确认目录下有 `results.csv`（或 `config.yaml` 作为备选）

### 2. 读取数据

**编码处理**：
- 优先 UTF-8，失败则回退 UTF-16LE（PowerShell `Tee-Object` 遗留）
- CSV 为空或无有效列名 → 报错退出

**列名标准化**：不同提取脚本列名可能不同，用模糊匹配：

| 匹配关键词 | 标准化名 | 用途 |
|-----------|---------|------|
| `ball_speed` / `speed` | `ball_speed` | X 轴 |
| `use_tube` / `tube` | `tube` | 分组 |
| `hit` | `hit` | 计数 |
| `pos_error` | `pos_error` | 误差 |
| `max_qdot` | `max_qdot` | 关速峰值 |
| `hit_type` | `hit_type` | active/passive/miss |
| `status` | `status` | 过滤 generation_failed |

### 3. 聚合统计

按 `(ball_speed, tube)` 分组，排除 `status == "generation_failed"` 的行：

| 指标 | 计算 |
|------|------|
| 命中数 | `sum(hit == "True")` |
| 命中率 | `命中数 / 组内行数 × 100` |
| 平均位置误差 | `mean(pos_error[hit == True]) × 1000` mm |
| 主动击球 | `sum("主动" in str(hit_type))` |
| 生成失败数 | `sum(status == "generation_failed")` |

**边界处理**：
- 全部 miss → 误差显示 "—"，命中率 0%
- 全部生成失败 → 单独标注，不计入总命中率
- pos_error = 0（被提取脚本清零） → 排除 0 值后计算

### 4. 提取关键数字

| 关键数字 | 计算方式 |
|---------|---------|
| 最高命中率 | `max(命中率)` 及其对应球速 |
| 50% 断崖点 | 命中率从 >50% 首次降至 <50% 的球速 |
| 误差范围 | `min(误差) - max(误差)` mm |
| 主动击球总数 | 所有速度的 active 求和 |
| 关速峰值 | `max(max_qdot)` 及其对应球速 |
| 生成失败总数 | 所有速度的 generation_failed 求和 |

### 5. 填写记录文件

**文件位置**：`docs/experiments/YYYY-MM-DD_expN_<描述>.md`

**Agent 自动填写的段落**（从模板复制结构后填入）：
- 日期、编号、数据目录链接
- 参数表（从 `config.yaml` 或对话上下文提取）
- 结果表（聚合数据生成的 Markdown 表格）
- 关键数字摘要

**不填写的段落**（插入占位标记）：
- 目的：`<!-- 人工填写 -->`
- 结论：`<!-- 人工填写，格式：1. xxx  2. xxx  3. xxx -->`

**同名文件处理**：已存在时询问用户是否覆盖。

### 6. 更新索引表

在 `docs/experiments/README.md` 索引表追加一行：

```markdown
| YYYY-MM-DD | [实验名](filename.md) | 球速范围 | 约束 | 命中率区间 | (结论待填) |
```

**去重**：日期+实验名已存在则跳过。

## 易错点与处理

| # | 易错点 | 触发条件 | 处理方案 |
|---|--------|---------|---------|
| 1 | CSV UTF-16LE 乱码 | PowerShell Tee-Object 遗留 | 先 `decode("utf-8")`，失败回退 `decode("utf-16-le")` |
| 2 | pos_error 为 0 被纳入均值 | miss 行被提取脚本清零 | 计算误差均值时排除 `pos_error == 0` 的行 |
| 3 | generation_failed 计入分母 | 某些球速全部生成失败 | 排除 `status == "generation_failed"` 行；如全组失败则标注不计算 |
| 4 | 球速太多表格过长 | >10 行 | 折叠为前 5 行 + "..." + 后 2 行；完整数据链接到 CSV |
| 5 | pos_error 单位混乱 | 米 vs 毫米 | 统一 ×1000 转为 mm，表头标注 `(mm)` |
| 6 | 主动击球=0 被误认为是提取失败 | 当前算法全部被动接触 | **明确标注** `active=N`，加注"当前算法均为被动接触，非提取错误" |
| 7 | 重复记录 | 同一实验多次触发 | 检查索引表中是否已存在相同记录文件链接 |
| 8 | CSV 空文件 | 实验未运行或路径错误 | 报错："results.csv 为空，请确认实验已运行完成" |
| 9 | 列名不匹配 | 旧版 CSV 列名不同 | 用模糊匹配 (`"speed" in col.lower()`)，不强制精确匹配 |
| 10 | 索引表格式破坏 | 追加行时管道符号错位 | 用正则 `^\|.*\|$` 验证格式后再写入 |
