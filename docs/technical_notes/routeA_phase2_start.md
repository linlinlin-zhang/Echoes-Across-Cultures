# 路线A-开始文档（Phase 2：扩样重训）

时间：2026-02-27  
负责人：Codex

## 目标
在路线A基础上扩充多文化公开数据样本，并重跑完整链路：
- 冻结 CultureMERT 提取 embedding
- 训练 DCAS 下游模型
- 生成可对比的 Phase 2 评估结果

## Phase 2 数据计划
- west: `sanchit-gandhi/gtzan`
- india: `neerajaabhyankar/hindustani-raag-small`
- turkey: `bilal63/turkish_music_emotion_dataset`

目标规模（本轮）：每文化 160 首，总计约 480 首。

## 交付物
1. `storage/public/routeA_phase2/` 数据与模型产物
2. `reports/routeA_phase2_*` 数据质量、切分、评估报告
3. 路线A Phase 2 结束技术文档
