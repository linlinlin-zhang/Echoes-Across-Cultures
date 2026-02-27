# 数据集步骤-开始文档（第2阶段：切分与防泄漏）

时间：2026-02-27
负责人：Codex

## 目标
实现可复现的数据切分器，并显式验证“无泄漏”与“分布稳定性”，作为训练/评测的标准输入。

## 计划产出
1. `dcas/scripts/make_splits.py`
2. 切分产物：
   - `split_track_ids.json`（train/val/test track_id 列表）
   - `split_assignments.csv`（track_id,culture,split）
   - `split_report.json`（覆盖率、重叠、分布偏差、interaction 覆盖）
   - `split_report.md`（人读报告）

## 规则
- 默认按 culture 分层切分，支持随机种子复现
- 确保 train/val/test 无 track_id 重叠
- 校验各 split 中 culture 占比偏差
- 若提供 interactions.csv，评估各 split 的交互覆盖与未知 track 比例

## 验收标准
- 一条命令产出全部切分与报告文件
- 报告中明确“是否检测到泄漏”
- 在 toy 数据上跑通并给出有效统计
