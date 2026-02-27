# 第1点结束文档：OT 排序退化问题已修复

时间：2026-02-27
负责人：Codex

## 变更范围
- 修改文件：`dcas/recommender.py`
- 变更点：候选打分从 `plan.sum(dim=0)` 改为基于 OT 的“列条件平均运输代价”

## 具体修复
旧逻辑：
- `score_j = sum_i plan_ij`
- 在平衡 OT + 固定 `b` 时，`score_j` 近似常数，排序失真

新逻辑：
- `col_mass_j = sum_i plan_ij`
- `avg_cost_j = sum_i plan_ij * cost_ij / col_mass_j`
- `score_j = softmax(-avg_cost_j)`

意义：
- 排序由传输代价差异驱动，而非固定边际约束
- 在不改 OT 主体流程前提下，恢复候选可区分性

## 验证结果
命令：
- `python -m dcas.cli.recommend --model ./toy/model.pt --tracks ./toy/tracks.npz --interactions ./toy/interactions.csv --user u0 --target_culture india --k 10`
- `k=20` 分数离散度检查脚本

结果：
- 推荐链路可正常运行
- `k=20` 时 `score` 唯一值（保留 8 位小数）为 20 个
- 示例区间：`min=0.0025426268`, `max=0.0025540180`

## 残余风险
- 当前仍是平衡 OT + 均匀 `b`，仅修复了排序退化，不代表已达最佳推荐策略。
- 下一步可考虑：非均匀目标先验、unbalanced OT、联合优化 relevance/serendipity。
