# 路线A-开始文档（Phase 1：多文化公开集基线）

时间：2026-02-27  
负责人：Codex

## 目标
按路线 A 启动第一轮可复现实验：
- 冻结基础模型（CultureMERT）仅提取 embedding
- 训练下游 DCAS
- 在公开数据上形成首版 baseline 报告

## 数据策略（Phase 1）
- `sanchit-gandhi/gtzan` -> `culture=west`
- `neerajaabhyankar/hindustani-raag-small` -> `culture=india`
- `bilal63/turkish_music_emotion_dataset` -> `culture=turkey`

说明：先用小规模样本完成链路打通，后续扩展样本量与评测深度。

## 本阶段交付
1. 导入与合并后的多源 `metadata.csv`
2. `tracks.npz`（CultureMERT embedding）
3. `interactions.csv`（弱监督联调用）
4. 训练模型与推荐评估报告
5. 路线 A 技术结束文档
