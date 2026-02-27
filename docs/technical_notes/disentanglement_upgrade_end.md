# 深度解纠缠增强结束文档

时间：2026-02-27  
负责人：Codex

## 本轮完成
1. 因子层升级
- 新增脚本：`dcas/scripts/build_shared_acoustic_factors.py`
- 基于音频构建跨文化共享离散因子：
  - `factor_energy`
  - `factor_brightness`
  - `factor_texture`
  - `factor_dynamics`
  - `factor_percussiveness`
- 产物：`reports/routeA_shared_factors_cn.csv`

2. 评测层升级
- 重写：`dcas/scripts/evaluate_disentanglement.py`
- 新增能力：
  - 多 seed 评测（`--seeds`）
  - 统计汇总（mean/std/ci95）
  - 线性 probe 的 DCI informativeness
  - 向后兼容单 seed 字段

3. 结果重评估
- 基线模型：`reports/routeA_disentanglement_phase2_cn_sharedfactors.{json,md}`
- PAL round2：`reports/routeA_disentanglement_phase3_round2_cn_sharedfactors.{json,md}`
- 对比：`reports/routeA_disentanglement_sharedfactors_compare_cn.md`
- 泄漏检查：`reports/routeA_factor_leakage_check_cn.md`

## 关键结论
1. 因子泄漏显著降低
- 旧因子 `label`：`MI(culture,label)/H(culture) = 1.000000`
- 新共享因子：`0.03~0.09` 区间（显著低于 1）

2. 解纠缠结论更可信，但仍非“充分”
- 在共享因子协议下，不同 latent 空间存在 trade-off：
  - `zc/zs` 部分指标提升
  - `za` 在部分指标下降
- 说明当前模型仍未形成稳定单调的“强解纠缠”证据链

3. 现阶段应继续升级模型目标
- 当前损失仍是近似去相关框架（KL + cov + contrast + domain）
- 建议下一步进入模型层改造（TC/MI/因子监督/交换一致性）

## 下一步建议（按优先级）
1. 模型层：引入更强解纠缠正则（TC 或 MINE/CLUB 近似）
2. 训练层：加入分阶段/权重调度，避免 `za` 退化
3. 评测层：补 bootstrap CI 与显著性检验，形成论文表格
