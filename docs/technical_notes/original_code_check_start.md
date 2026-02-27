# 原始代码检查-开始文档

时间：2026-02-27
负责人：Codex

## 检查范围
仅针对原始项目代码（非新增模块）进行可靠性排查与修复。

## 已确认高优先级问题
1. `dcas/scripts/make_toy_data.py`：固定 `size=80` 且 `replace=False`，当候选不足时抛异常。
2. `dcas/cli/train.py` 与 `dcas/pipelines.py::train_model`：`drop_last=True` 在小数据集可能导致 0 个 batch，仍继续并保存无效模型。

## 本轮目标
- 让 toy 数据生成对小样本稳健。
- 让训练入口在小样本场景也可训练，且阻止“空训练成功”假象。
