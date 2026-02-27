# DCAS 论文主张与实现对齐矩阵

更新日期：2026-02-27

## 目的
本文件用于控制论文中的技术主张范围，确保每个主张都有可复现实验或代码证据支持，避免 claim-evidence mismatch。

## 主张-证据矩阵

| 主张 | 状态 | 证据文件 | 备注（论文建议写法） |
|---|---|---|---|
| 三因子解纠缠建模（`zc/zs/za`） | 已实现 | `dcas/models/dcas_vae.py` | 可表述为“提出并实现了原型级解纠缠网络”。 |
| 领域对抗去文化化（`za` 上 GRL + 判别器） | 已实现 | `dcas/models/dcas_vae.py`, `dcas/models/layers.py` | 可表述为“实现了 DANN 风格对抗项并在原型中验证可运行性”。 |
| 内容一致性对比学习（InfoNCE） | 已实现 | `dcas/models/losses.py`, `dcas/models/dcas_vae.py` | 可表述为“对 `zc` 使用噪声增强的一致性约束”。 |
| OT/Sinkhorn 跨文化推荐 | 已实现 | `dcas/ot/sinkhorn.py`, `dcas/recommender.py` | 2026-02-27 已修复候选分数退化问题（基于平均传输代价排序）。 |
| PAL 不确定性选样 | 已实现 | `dcas/pal/uncertainty.py`, `dcas/cli/pal_loop.py` | 可表述为“实现了 entropy-based PAL 原型”。 |
| 专家成对约束回灌训练 | 已实现（原型） | `dcas/pal/constraints.py`, `dcas/cli/train.py`, `dcas/pipelines.py` | 可表述为“支持 pairwise constraints 的训练增强”。 |
| 全栈交互流程（数据/训练/推荐/PAL） | 已实现 | `dcas_server/app.py`, `web/src/App.tsx` | 可表述为“提供实验原型系统，不等同生产部署”。 |
| CultureMERT 持续预训练并集成 | 未实现（计划中） | `README.md`（路线图） | 论文不可写成“已接入”；应写“可兼容外部 embedding，CultureMERT 为计划路线”。 |
| 分层 VAE + TC（严格 FactorVAE 级别） | 部分实现 | `dcas/models/dcas_vae.py`, `dcas/models/losses.py` | 当前为 KL + 协方差去相关近似，不应宣称“完整 TC 实现”。 |
| 风格迁移生成模块（反事实音频） | 未实现（计划中） | `README.md`（路线图） | 论文应写“接口预留/未来工作”。 |
| 标准解纠缠指标（MIG/DCI/SAP）评测 | 未实现（计划中） | 无 | 当前不可做强主张；需补评测脚本后再写实验结论。 |
| 大规模真实跨文化基准全面实验 | 未实现（计划中） | 无 | 当前结果以 toy + 原型验证为主。 |

## 可主张与不可主张边界

可主张（当前可被代码支持）：
- 提出并实现了一个可运行的 DCAS 原型，包含解纠缠、领域对抗、OT 推荐、PAL 回路。
- 在 toy/原型设置下完成端到端闭环验证。
- 给出跨文化推荐与公平性相关的初步指标。

不可主张（当前证据不足）：
- “已达到 SOTA” 或 “在公开大基准上全面领先”。
- “已完整实现 CultureMERT 持续预训练并验证泛化优势”。
- “已完成严格解纠缠定量评测（MIG/DCI/SAP）并显著领先”。

## 投稿文本建议（降风险）

建议使用以下措辞风格：
- “we present a runnable prototype / proof-of-concept”
- “preliminary evidence on synthetic or small-scale settings”
- “full benchmark validation is left for future work”

避免使用以下措辞：
- “state-of-the-art across datasets”
- “fully validated at scale”
- “comprehensive disentanglement evaluation”

## 与路线图衔接
后续实现优先级请参考：
- `NEXT_STEPS_ROADMAP.md`
- `docs/technical_notes/point1_ot_ranking_end.md`
