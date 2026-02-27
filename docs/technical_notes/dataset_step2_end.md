# 数据集步骤-结束文档（第2阶段：切分与防泄漏）

时间：2026-02-27
负责人：Codex

## 已完成内容
1. 新增切分脚本：
   - `dcas/scripts/make_splits.py`

2. 输出产物：
   - `split_track_ids.json`
   - `split_assignments.csv`
   - `split_report.json`
   - `split_report.md`

3. 已实现能力：
- 按 culture 分层切分 train/val/test
- 固定 seed 可复现
- 自动泄漏检查（split 间 track_id 重叠）
- culture 分布漂移统计（相对全局分布）
- interactions 覆盖统计（每 split 行为占比、用户数、未知 track 比例）

## 本地验证结果
1. toy 数据（`./toy`）
- status: `pass`
- split 比例：960 / 120 / 120
- 泄漏：0
- 最大分布偏差：0.0025
- 交互覆盖：train 79.17%, val 9.79%, test 11.04%

2. toy_small 数据（`./toy_small`）
- status: `pass`
- strict 模式返回码 0

## 输出示例路径
- `reports/splits_toy/split_report.json`
- `reports/splits_toy/split_report.md`

## 价值
该阶段完成后，训练和评测可统一使用固定 split，避免“随机切分波动”和“数据泄漏导致虚高指标”。

## 建议下一步（第3阶段）
- 构建 interactions 清洗与权重规范工具（日志 -> 标准 interactions.csv）。
- 输出行为权重映射文档与清洗报告，为论文实验节提供可复现依据。
