# 第2点-子项2开始文档：生成式风格迁移（embedding 级）

时间：2026-02-27
负责人：Codex

## 目标
在现有 DCAS 解纠缠模型基础上，新增可运行的反事实风格迁移能力，用于解释跨文化推荐：
- 保留源曲目内容与情感（`zc`, `za`）
- 注入目标曲目风格（`zs`）
- 生成新的 embedding 并检索最近邻曲目作为“可听替代”

## 范围
1. 新增核心模块：`dcas/style_transfer.py`
2. 新增 pipeline 封装与 CLI
3. 新增 API endpoint，输出生成 embedding 文件与最近邻结果

## 验收标准
- 可从已训练模型和 tracks.npz 生成 counterfactual embedding
- 可按 target_culture 过滤候选池
- 输出 nearest neighbors 列表（track_id, culture, distance）
