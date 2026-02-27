# 非数据集升级开始文档（评测稳健性与显著性）

时间：2026-02-27  
负责人：Codex

## 目标
在不引入新数据集的前提下，优先补齐论文证据链中的“评测统计稳健性”短板：

1. 推荐评测增加置信区间（bootstrap CI）。
2. 新增成对显著性检验（paired bootstrap + permutation test）。
3. 将显著性检验自动接入 Phase 3 PAL 与 ablation 流程，直接产出可写论文的统计报告。

## 计划改动
1. 修改 `dcas/scripts/evaluate_recommender.py`
- 增加 summary/per-culture 的 CI 字段。
- 增加 CLI 参数：`--bootstrap_samples`, `--bootstrap_seed`。

2. 新增 `dcas/scripts/compare_recommender_runs.py`
- 对两份评测 json 做配对比较（按 `user_id,target_culture`）。
- 输出 delta、95% CI、双侧 p-value、paired Cohen's d。

3. 修改自动化脚本
- `dcas/scripts/run_phase3_pal.py`：自动输出 baseline vs roundX 显著性报告。
- `dcas/scripts/run_ablation.py`：自动输出 full vs ablation 显著性报告。
