# 技术文档（开始）：阶段3 基线对比（VAE / beta-VAE / FactorVAE）

日期：2026-02-28  
负责人：Codex

## 目标
- 与以下基线做统一协议对比：
  - 标准 VAE
  - beta-VAE
  - FactorVAE
- 证明三因子架构（3-factor DCAS）的必要性。

## 计划实现
1. 新增脚本 `dcas/scripts/run_baseline_comparison.py`。
2. 同一数据、同一评测协议、多 seed。
3. 每个 seed 计算“full - baseline”配对显著性。
4. 输出论文草稿表和必要性检查结论。

## 指标
- `serendipity`（越高越好）
- `cultural_calibration_kl`（越低越好）
- `minority_exposure_at_k`（越高越好）

