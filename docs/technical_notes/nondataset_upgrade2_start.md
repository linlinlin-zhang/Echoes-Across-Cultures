# 非数据集升级开始文档（校准指标去退化 + 统一评测入口）

时间：2026-02-27  
负责人：Codex

## 目标
在不新增数据集的前提下，继续提升实验可用性与论文证据链质量：

1. 修复 `cultural_calibration_kl` 在固定 `target_culture` 评测中的常量退化问题。
2. 增加统一评测入口，减少“多脚本手工拼接”的复现实验成本。

## 计划改动
1. `dcas/recommender.py`
- 将 `cultural_calibration_kl` 改为基于 `zs` 软文化分布对目标文化先验的 KL。
- 保留旧版常量定义为 `cultural_calibration_kl_legacy` 用于历史对照。

2. `dcas/scripts/evaluate_recommender.py`
- 行级/汇总级支持新增指标：
  - `cultural_calibration_kl_legacy`
  - `target_culture_prob_mean`
  - `user_culture_alignment_kl`

3. 新增统一入口 `dcas/cli/eval.py`
- 一条命令可选执行推荐评测、显著性对比、解纠缠评测。
