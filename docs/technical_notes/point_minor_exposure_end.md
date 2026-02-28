# 技术文档（结束）：Point-0 Minority Exposure@K

日期：2026-02-28  
负责人：Codex

## 已完成实现
1. `evaluate_recommender.py` 已新增：
- popularity 统计与 minority 集构建
- 指标 `minority_exposure_at_k`
- summary/per-culture 的 mean/std/CI95
- 配置输出：
  - `minority_quantile`
  - `minority_popularity_threshold`
  - `minority_catalog_ratio`
- CLI 参数：`--minority_quantile`

2. 退化场景修复：
- 当 popularity 全相同或接近全相同时，按固定比例确定 minority 子集，避免指标退化为全 1。

3. 评测对比链路联动：
- `compare_recommender_runs.py` 已增强为“缺指标容错”。
- `run_phase3_pal.py`、`run_ablation.py` 报表已纳入 `minority_exposure_at_k`。
- `dcas/cli/eval.py` 对比指标列表已加入 `minority_exposure_at_k`。

## 验证
- 通过 `python -m compileall` 静态编译检查。
- 通过 toy 数据端到端运行验证（见阶段收口文档）。

