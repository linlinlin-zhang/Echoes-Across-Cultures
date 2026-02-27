# 数据集步骤-开始文档（第1阶段：验证与画像）

时间：2026-02-27
负责人：Codex

## 目标
实现一个可重复执行的数据集验证与画像工具，用于回答：
- 这批数据是否满足训练输入规范？
- 数据是否存在高风险质量问题（重复、缺失、异常、分布失衡）？
- interactions 是否与 tracks 一致并覆盖核心用户行为？

## 计划产出
1. `dcas/scripts/validate_dataset.py`
2. 报告输出：
   - 机器可读：`dataset_profile.json`
   - 人类可读：`dataset_profile.md`
3. 在现有 toy 数据上生成示例报告

## 核心检查项
- tracks schema、N/D、dtype、非空、有限值
- track_id 唯一性、culture 分布、embedding 范数分布
- affect_label 覆盖率与类别分布（若存在）
- interactions schema、权重统计、重复 user-track
- interactions 与 tracks 的 referential integrity（孤儿 track）
- 用户覆盖与冷启动风险提示

## 验收标准
- 一条命令生成完整报告
- 严重问题可以通过 `--strict` 返回非零退出码
