# 非数据集升级结束文档（评测稳健性与显著性）

时间：2026-02-27  
负责人：Codex

## 已完成内容
1. 推荐评测 CI 增强
- 文件：`dcas/scripts/evaluate_recommender.py`
- 新增：
  - `serendipity_ci95_low/high`
  - `cultural_calibration_kl_ci95_low/high`
  - per-target-culture 的 mean/std/ci95
  - CLI 参数：`--bootstrap_samples`, `--bootstrap_seed`

2. 新增成对显著性检验脚本
- 文件：`dcas/scripts/compare_recommender_runs.py`
- 能力：
  - 按 `(user_id, target_culture)` 对齐配对样本
  - 计算 `delta_mean`、`95% CI`（paired bootstrap）
  - 计算 `p_value_two_sided`（sign-flip permutation）
  - 计算 `cohen_d_paired`
  - 输出 `json + markdown`

3. 自动化流程接入显著性输出
- 文件：`dcas/scripts/run_phase3_pal.py`
  - 自动生成 `compare_baseline_vs_round*.{json,md}`
  - summary 增加 `comparisons` 字段
- 文件：`dcas/scripts/run_ablation.py`
  - 自动生成 `compare_full_vs_*.{json,md}`
  - summary 增加 `comparisons` 字段

4. 运行文档更新
- 文件：`docs/ROUTE_A_PHASE3_RUNBOOK.md`
- 已补充显著性参数与输出路径。

## 验证
1. 编译检查通过
- `python -m compileall dcas`

2. 显著性脚本实跑（使用现有报告）
- `phase2_cn -> phase4_tc_hsic`：
  - `serendipity delta = +0.0616306397`
  - `95% CI = [0.052786, 0.070391]`
  - `p = 0.000500`
- `phase3 baseline -> round2`：
  - `serendipity delta = -0.0123940788`
  - `95% CI = [-0.023392, -0.001334]`
  - `p = 0.031484`

对应产物：
- `reports/routeA_compare_phase2_vs_phase4_significance.{json,md}`
- `reports/routeA_phase3_pal_cn/compare_baseline_vs_round2_manual.{json,md}`

## 结论
本轮升级不依赖新增数据集，已将“是否提升”从点估计扩展为“效应量 + 区间 + 显著性”，可直接用于论文结果段与附录统计证据。
