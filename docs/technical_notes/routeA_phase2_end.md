# 路线A-结束文档（Phase 2：扩样重训）

时间：2026-02-27  
负责人：Codex

## 本阶段完成内容
1. 三文化公开数据扩样：
- west（GTZAN）= 160
- india（Hindustani）= 160
- turkey（Turkish Emotion）= 160
- 合计 = 480 tracks

2. 数据合并与交互构建：
- 合并元数据：`metadata_merged.csv`（480 行）
- 弱监督交互：`interactions.csv`（2432 行，60 users）

3. 表征构建（冻结基座）：
- 模型：`ntua-slp/CultureMERT-95M`
- 输出：`tracks.npz`（`n_tracks=480`, `dim=768`）

4. 训练与评估：
- 训练参数：`epochs=10`, `batch_size=64`, `lr=2e-3`
- 训练 loss：`1.2673 -> 0.7671`
- 批量评估：`n_user_culture_evals=180`

## 指标结果
- `serendipity_mean = 0.8619925425`
- `serendipity_std = 0.0503729024`
- `cultural_calibration_kl_mean = 1.0986122887`

## 与 Phase 1 对比
- Phase 1 `serendipity_mean = 0.8272667835`
- Phase 2 `serendipity_mean = 0.8619925425`
- 提升：`+0.0347257591`

评估规模：
- Phase 1：72 user-culture 组合
- Phase 2：180 user-culture 组合

## 产物路径
- 数据与模型：`storage/public/routeA_phase2/`
- 数据画像：`reports/routeA_phase2_dataset_profile.json`
- 切分报告：`reports/routeA_phase2_splits/split_report.json`
- 评估报告：`reports/routeA_phase2_eval.json`
- 运行手册：`docs/ROUTE_A_PHASE2_RUNBOOK.md`

## 说明与局限
- 当前交互仍为弱监督合成，适合路线 A 工程验证，不等于真实用户行为结论。
- `cultural_calibration_kl` 在固定目标文化推荐设置下几乎常量，后续需补更有区分度的公平性评测定义。

## 下一步建议（Phase 3）
- 引入真实或半真实交互数据（优先 PAL 约束回灌）。
- 增加消融实验（domain/OT/constraints 开关）和统计检验。
- 补标准解纠缠评测（MIG/DCI/SAP）以支撑论文级主张。
