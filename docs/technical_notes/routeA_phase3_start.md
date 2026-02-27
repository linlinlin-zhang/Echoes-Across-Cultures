# 路线A-开始文档（Phase 3：PAL 回灌 + 消融 + 解纠缠评测）

时间：2026-02-27  
负责人：Codex

## 目标
在四文化基线（west/india/turkey/china）上完成论文级实验链路补齐：
1. 执行 PAL 约束回灌两轮训练与增益对比
2. 执行消融实验（关闭 domain / OT / constraints）并生成表格草稿
3. 补充 MIG/DCI/SAP 解纠缠评测

## 输入基线
- 数据：`storage/public/routeA_phase2_cn/tracks.npz` + `interactions.csv`
- 基线模型：`storage/public/routeA_phase2_cn/model.pt`
- 元数据：`storage/public/routeA_phase2_cn/metadata_merged.csv`

## 计划产物
- Phase 3 PAL 报告：`reports/routeA_phase3_pal_cn/`
- Ablation 报告：`reports/routeA_phase3_ablation_cn/`
- Disentanglement 报告：`reports/routeA_disentanglement_*.{json,md}`
- 新增脚本：PAL 回灌自动化、ablation 自动化、MIG/DCI/SAP 评测
