# V3 BPR-Hybrid 校准感知调优记录
更新日期：2026-03-19

## 1. 背景

第一版 `bpr_two_stage_hybrid` 已经证明：

- 相对 `BPR-MF`，它能提升 `serendipity` 与 `minority exposure`
- 但会牺牲 `KL` 与 `target_prob`

这说明问题不一定在模型主干本身，更可能在最终多目标融合权重没有调对。

因此本轮不重训主干，而是直接对 `BPR-hybrid` 的最终推理打分做一轮校准感知调优。

## 2. 调优方式

代码位置：

- `E:/Desktop/Echo/dcas/embedding_recommenders.py`
- `E:/Desktop/Echo/dcas/scripts/run_recommender_benchmarks.py`

本轮把最终分数从：

- `rerank + recall`

扩展成：

- `rerank`
- `recall`
- `BPR score`
- `target affinity`
- `minority`
- `source inverse frequency`

然后做小范围权重搜索。

## 3. 小范围搜索结果

单方法评估中，几组代表性权重大致如下：

1. `current_cfg`
- `ser = 0.5115`
- `KL = 2.0200`
- `minority = 0.2078`
- `target = 0.1979`

2. `target_bpr_heavy`
- `ser = 0.5119`
- `KL = 2.0099`
- `minority = 0.2130`
- `target = 0.2004`

3. `balanced_plus`
- `ser = 0.5119`
- `KL = 2.0146`
- `minority = 0.2330`
- `target = 0.1992`

4. `target_high`
- `ser = 0.5144`
- `KL = 2.0015`
- `minority = 0.1996`
- `target = 0.2029`

5. `ser_minority`
- `ser = 0.5102`
- `KL = 2.0082`
- `minority = 0.2838`
- `target = 0.2005`

最终选用的是 `ser_minority`，原因是它在几乎不损失 `ser` 的前提下，把 `minority` 明显推高，同时 `KL` 和 `target` 也都保持在很强的位置。

## 4. 最终采用的权重

配置已更新到：

- `E:/Desktop/Echo/configs/benchmark/recommender_benchmark_v3_culturemert_stage3_bprhybrid.run.json`

最终权重为：

- `rerank_weight = 0.60`
- `recall_weight = 0.12`
- `bpr_weight = 0.08`
- `target_affinity_weight = 0.06`
- `minority_weight = 0.10`
- `source_weight = 0.04`

## 5. 完整 benchmark 结果

完整结果位于：

- `E:/Desktop/Echo/reports/benchmarks/v3_main_culturemert_stage3_bprhybrid/benchmark_summary.json`
- `E:/Desktop/Echo/reports/benchmarks/v3_main_culturemert_stage3_bprhybrid/benchmark_table.md`

调优后 `bpr_two_stage_hybrid` 最终指标：

- `serendipity = 0.5102`
- `KL = 2.0082`
- `minority = 0.2838`
- `target_prob = 0.2005`

## 6. 关键比较

相对 `BPR-MF`：

- `serendipity +3.78%`
- `KL +0.71%`
- `minority exposure +90.25%`
- `target_prob +2.41%`

这意味着调优后它已经对 `BPR` 四项全胜。

相对 `DCAS stage3 / dcas_full_ot`：

- `serendipity -39.64%`
- `KL +1.70%`
- `minority exposure +18.33%`
- `target_prob +4.37%`

也就是说，调优后的 `BPR-hybrid` 已经形成了一个真正强的对手：

- 它在 `KL / minority / target_prob` 三项上超过 `DCAS`
- 但在 `serendipity` 上仍然明显落后

## 7. 结论

这轮调优把研究结论推进了一步：

1. `DCAS` 现在不能再只拿 “整体更强” 来描述
- 更准确的是，它主要在 `serendipity` 上保持显著优势

2. 一个足够认真调过的 `BPR + learned reranker` 强混合基线
- 完全可以在 `calibration / minority / target affinity` 上超过 `DCAS`

3. 因此下一步如果继续强化 `DCAS`
- 最关键的不是再补更多弱基线
- 而是专门追 `serendipity` 优势能否在不牺牲其余三项的情况下被保住
