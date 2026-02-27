# 数据集步骤-结束文档（第1阶段：验证与画像）

时间：2026-02-27
负责人：Codex

## 已完成内容
1. 新增数据验证与画像脚本：
   - `dcas/scripts/validate_dataset.py`

2. 支持输出：
   - 机器可读：`--out_json`
   - 人类可读：`--out_md`

3. 支持阈值化检查与严格模式：
   - `--strict`（当 status=fail 时非零退出）
   - 可配置阈值：culture 最小样本、分布失衡、未知 track 比例、重复 user-track 比例等

## 覆盖检查项
- tracks：
  - schema 可用性（通过 `load_tracks`）
  - embedding 有限值比例
  - zero-norm 比例
  - track_id 重复
  - culture 分布与失衡比
  - affect_label 有无与分布

- interactions：
  - required columns 检查
  - unknown track 引用
  - duplicate (user,track)
  - 权重有效性（非有限值、非正值）
  - 用户活跃度统计
  - track 覆盖率

## 本地验证结果
1. toy 数据（`./toy`）
- status: `pass`
- 无 issue
- 报告：
  - `reports/dataset_profile_toy.json`
  - `reports/dataset_profile_toy.md`

2. 小样本 toy（`./toy_small`）
- status: `warn`
- issue: `tracks.culture_low_count`（india 仅 28 < 30）
- 报告：
  - `reports/dataset_profile_toy_small.json`
  - `reports/dataset_profile_toy_small.md`

## 价值
该工具可作为数据入库门禁（data gate），在训练前自动判断“是否可训、是否有高风险偏差”。

## 建议下一步（第2阶段）
- 实现数据切分器（train/val/test）并加入泄漏防护规则（track/culture 分层）。
- 输出 split 报告，和本阶段 profile 一起作为论文数据附录基础。
