# 非数据集升级结束文档（校准指标去退化 + 统一评测入口）

时间：2026-02-27  
负责人：Codex

## 已完成内容
1. 校准指标重构（去退化）
- 文件：`dcas/recommender.py`
- 变更：
  - 新 `cultural_calibration_kl`：
    - 在 `zs` 空间构建文化质心
    - 计算推荐列表的软文化分布
    - 与“目标文化平滑先验”做 KL
  - 保留旧定义为 `cultural_calibration_kl_legacy`
  - 新增：
    - `target_culture_prob_mean`
    - `user_culture_alignment_kl`

2. 推荐评测扩展
- 文件：`dcas/scripts/evaluate_recommender.py`
- 变更：
  - 行级数据保留新增指标
  - summary/per-culture 增加新增指标汇总
  - 与已加入的 bootstrap CI 兼容

3. 统一评测入口
- 新增文件：`dcas/cli/eval.py`
- 能力：
  - recommender 评测
  - baseline 对比显著性评测
  - disentanglement 评测（可选）
  - 自动落盘统一产物 `eval_suite_summary.json`

4. 文档
- 新增：`docs/EVAL_SUITE_GUIDE.md`

## 验证结果
1. 编译与 help
- `python -m compileall dcas` 通过
- `python -m dcas.cli.eval --help` 可用

2. 重新评测（v2）
- `reports/routeA_phase2_cn_eval_v2.json`
- `reports/routeA_phase4_tc_hsic_eval_v2.json`

关键观测：
- `cultural_calibration_kl` 不再恒定（phase2 vs phase4 出现可测差异）
- `cultural_calibration_kl_legacy` 仍保持常量，可用于历史对照

3. 显著性报告（v2）
- `reports/routeA_compare_phase2_vs_phase4_significance_v2.{json,md}`

4. 统一入口实跑
- 产物目录：`reports/eval_suite_phase4_v2/`
- 核心产物：
  - `recommender_eval.json`
  - `recommender_compare.{json,md}`
  - `disentanglement_eval.{json,md}`
  - `eval_suite_summary.json`

## 结论
本轮已在不引入新数据集的条件下，完成：
1. 指标定义层去退化（校准指标可用）
2. 统计证据层保留（paired significance）
3. 工程入口层统一（single-command eval suite）
