# 技术文档（开始）：阶段2 超参数敏感性（TC/HSIC）

日期：2026-02-28  
负责人：Codex

## 目标
- 完成 `lambda_tc × lambda_hsic` 网格搜索。
- 提供“稳定性证据”，避免只给单次最优点。

## 计划实现
1. 新增脚本 `dcas/scripts/run_tc_hsic_sensitivity.py`。
2. 支持参数：
- `--tc_values`
- `--hsic_values`
- `--seeds`
- 训练/评测超参与输出目录
3. 自动产出：
- 每次训练模型与 eval json
- 网格汇总（mean/std/ci95）
- 稳定性指标（seed 最优配置频率、跨 seed 排名 spearman）
- Markdown 草稿表

## 输出物
- `tc_hsic_sensitivity_summary.json`
- `tc_hsic_sensitivity_summary.md`

