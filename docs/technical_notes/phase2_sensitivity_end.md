# 技术文档（结束）：阶段2 超参数敏感性（TC/HSIC）

日期：2026-02-28  
负责人：Codex

## 已完成实现
1. 新脚本：
- `dcas/scripts/run_tc_hsic_sensitivity.py`

2. 核心能力：
- 多 seed 网格训练与评测
- 输出指标：
  - `serendipity`
  - `cultural_calibration_kl`
  - `minority_exposure_at_k`
- 每网格点统计：
  - mean/std/ci95
  - 综合目标 z-score（用于排序）
- 稳定性证据：
  - `best_config_by_seed`
  - `dominant_best_config_ratio`
  - `rank_spearman_mean/std`

3. 产物格式：
- JSON 全量记录（含每 run 路径）
- Markdown 表格（可直接用于论文草稿整理）

## 最小运行验证（smoke）
- toy 数据运行命令已通过：
  - `python -m dcas.scripts.run_tc_hsic_sensitivity ... --tc_values 0.0,0.05 --hsic_values 0.0,0.02 --seeds 42 --epochs 1`
- 成功输出：
  - `tmp/reports_tc_hsic_smoke/tc_hsic_sensitivity_summary.json`
  - `tmp/reports_tc_hsic_smoke/tc_hsic_sensitivity_summary.md`

