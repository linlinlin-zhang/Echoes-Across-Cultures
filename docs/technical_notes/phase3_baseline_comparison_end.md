# 技术文档（结束）：阶段3 基线对比（VAE / beta-VAE / FactorVAE）

日期：2026-02-28  
负责人：Codex

## 已完成实现
1. 新脚本：
- `dcas/scripts/run_baseline_comparison.py`

2. 对比设置：
- full：`three_factor_dcas`
- baselines：
  - `vae`（shared encoder + 基础 KL）
  - `beta_vae`（shared encoder + 高 beta）
  - `factorvae`（shared encoder + TC 正则）

3. 自动化产物：
- 每变体每 seed 的模型与评测结果
- 配对显著性（`compare_recommender_runs`）：
  - `delta_full_minus_baseline`
  - p-value 统计
  - full_better_rate
- 草稿表：
  - `baseline_comparison_table_draft.md`
- 汇总 JSON：
  - `baseline_comparison_summary.json`

4. 必要性检查：
- 脚本给出 `three_factor_necessity_checks`，用于快速判断“3-factor 是否系统性优于三类基线”。

## 最小运行验证（smoke）
- toy 数据运行命令已通过：
  - `python -m dcas.scripts.run_baseline_comparison ... --seeds 42 --epochs 1`
- 成功输出：
  - `tmp/reports_baseline_smoke/baseline_comparison_summary.json`
  - `tmp/reports_baseline_smoke/baseline_comparison_table_draft.md`

