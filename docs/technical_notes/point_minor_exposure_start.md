# 技术文档（开始）：Point-0 Minority Exposure@K

日期：2026-02-28  
负责人：Codex

## 目标
- 在推荐评测链路中新增 `Minority Exposure@K` 指标。
- 指标要进入：
  - `rows`
  - `per_target_culture`
  - `summary`
  - 与现有 bootstrap CI 报告体系一致。

## 计划实现
1. 基于 `interactions.csv` 构建 track popularity。
2. 用分位数阈值（默认 0.25）定义 minority tracks。
3. 对每个 `(user, target_culture)` 推荐结果计算 `minority_exposure_at_k`。
4. 在 `evaluate_recommender.py` 增加 CLI 参数 `--minority_quantile`。
5. 将此指标接入后续比较脚本和 Phase3/Ablation 报表。

## 风险点
- 小数据或均匀流行度时，分位数可能退化为全量命中，需要退化修正。
- 历史报告可能缺失新指标，比较脚本需容错处理。

