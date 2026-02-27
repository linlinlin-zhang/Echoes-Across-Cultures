# 路线A-结束文档（Phase 3：PAL 回灌 + 消融 + 解纠缠评测）

时间：2026-02-27  
负责人：Codex

## 本阶段完成内容
1. 完成 PAL 两轮回灌自动化
- 新增 `pal_tasks -> constraints -> train -> eval` 的可重复流水线
- 在四文化数据上完成 round1 / round2 全流程

2. 完成消融实验自动化
- 支持 `domain`、`constraints`、`OT` 三项开关
- 自动导出论文主表草稿（Markdown + JSON）

3. 完成 MIG/DCI/SAP 评测
- 新增 `zc/zs/za` 三空间的标准化评测脚本
- 输出 JSON 与 Markdown 报告，可直接进入实验附录

## PAL 两轮结果（routeA_phase3_pal_cn）
- baseline: `serendipity=0.8109801431`
- round1: `serendipity=0.7974228684`（vs baseline `-0.0135572747`）
- round2: `serendipity=0.7985860643`（vs baseline `-0.0123940788`）
- `cultural_calibration_kl` 本轮保持常量：`1.3862943611`

结论：
- 本轮“模拟专家反馈”条件下，PAL 回灌未带来 serendipity 提升。
- round2 相比 round1 有小幅回升，但未超过 baseline。

## 消融结果（routeA_phase3_ablation_cn）
- full: `0.7929157051`
- no_domain: `0.7839433542`（`-0.0089723509`）
- no_constraints: `0.8074084540`（`+0.0144927489`）
- no_ot: `0.8056271891`（`+0.0127114839`）

结论：
- domain 对抗关闭后性能下降，说明该项在当前设置下有正贡献。
- constraints 与 OT 在当前约束质量/目标定义下并未体现正增益，提示需要改进约束来源与推荐目标。

## MIG / DCI / SAP（Phase2 vs Phase3 round2）
关键变化（culture,label 因子）：
- `za` 空间：
  - `DCI_disentanglement: 0.066638 -> 0.289469`（提升）
  - `DCI_informativeness: 0.648438 -> 0.304688`（下降）
- `zc` 空间：`SAP` 提升（`0.003906 -> 0.039062`），`MIG` 下降
- `zs` 空间：变化较混合

解释：
- 现阶段解纠缠指标存在 trade-off，尚未形成“全面单调提升”证据。
- 该结果支持下一步做约束质量提升与统计检验，而非直接给出强结论。

## 新增脚本
- `dcas/scripts/build_pal_feedback_constraints.py`
- `dcas/scripts/run_phase3_pal.py`
- `dcas/scripts/run_ablation.py`
- `dcas/scripts/evaluate_disentanglement.py`

## 核心产物路径
- PAL 报告：`reports/routeA_phase3_pal_cn/`
- Ablation 报告：`reports/routeA_phase3_ablation_cn/`
- 解纠缠报告：
  - `reports/routeA_disentanglement_phase2_cn.json`
  - `reports/routeA_disentanglement_phase3_round2_cn.json`
  - `reports/routeA_disentanglement_compare_cn.md`

## 风险与下一步
- 当前 PAL 反馈为“同标签模拟专家”，不是人工真实标注。
- 下一步必须切换为真实专家约束 + 置信度分层，补统计显著性检验（bootstrap/paired test），再形成 ISMIR 级主结论。
