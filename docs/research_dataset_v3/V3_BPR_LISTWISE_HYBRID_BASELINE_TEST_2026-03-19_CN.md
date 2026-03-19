# V3 BPR Listwise Hybrid Baseline 测试记录（2026-03-19）

## 1. 目标

在补完 `BPR-MF`、`BPR two-stage hybrid` 之后，继续补一条更标准的学习排序强基线，用来回答：

- 在当前 `CultureMERT stage3` 设定下，是否存在比 `BPR-hybrid` 更强的非 DCAS 基线？
- 若有，它和 `DCAS calibrated` 的差距还剩多少？

## 2. 实现选择

原本优先考虑的是更典型的树模型/学习排序基线，例如：

- `XGBoost`
- `LightGBM`
- `sklearn` GBDT / ranking proxy

但当前环境里这些依赖都被 `NumPy` 兼容问题卡住，无法稳定复现。因此本轮采用纯 `PyTorch` 路线，实现一条更标准的：

`BPR retrieval + listwise reranker`

核心思路：

1. 第一阶段仍使用 `BPR-MF` 做召回打分。
2. 第二阶段在 `BPR-hybrid` 现有 feature table 上训练 `listwise` reranker。
3. 使用当前最强 `BPR two-stage hybrid` checkpoint 做 warm start，避免从零开始训练。

对应代码已补到：

- `dcas/embedding_recommenders.py`
- `dcas/scripts/run_recommender_benchmarks.py`

## 3. 新增内容

新增的关键接口：

- `_make_bpr_hybrid_listwise_queries`
- `train_bpr_listwise_hybrid_ranker`
- `load_bpr_listwise_hybrid_ranker`
- `recommend_embedding_bpr_listwise_hybrid`

新的 benchmark 配置：

- `configs/benchmark/recommender_benchmark_v3_culturemert_stage3_bprlistwise.run.json`
- `configs/benchmark/recommender_benchmark_v3_culturemert_stage3_bprlistwise_tuned.run.json`

## 4. 第一版结果

第一版全量 benchmark：

- `bpr_listwise_hybrid`: `ser=0.5062`, `KL=2.0103`, `minority=0.2251`, `target=0.1994`

它已经是一条合格的强基线，但尚未超过当前 tuned `bpr_two_stage_hybrid`：

- `bpr_two_stage_hybrid`: `ser=0.5102`, `KL=2.0082`, `minority=0.2838`, `target=0.2005`

因此继续做了一轮 inference 权重搜索。

## 5. 调优后结果

最终晋级全量复测的 tuned 配置是：

- `rerank_weight=0.52`
- `recall_weight=0.12`
- `bpr_weight=0.08`
- `novelty_weight=0.02`
- `target_affinity_weight=0.10`
- `minority_weight=0.12`
- `source_weight=0.04`

全量结果：

- `bpr_listwise_hybrid`: `0.5135 / 1.9986 / 0.2503 / 0.2029`
- `bpr_two_stage_hybrid`: `0.5102 / 2.0082 / 0.2838 / 0.2005`
- `dcas_full_ot_calibrated_target`: `0.8386 / 1.8793 / 0.3814 / 0.2349`
- `dcas_full_ot_calibrated_minor`: `0.8404 / 1.9148 / 0.5190 / 0.2250`

顺序为：

- `serendipity`
- `cultural_calibration_kl`
- `minority_exposure_at_k`
- `target_culture_prob_mean`

## 6. 与当前最强非 DCAS 基线的比较

`bpr_listwise_hybrid` 相对 `bpr_two_stage_hybrid`：

- `serendipity +0.65%`
- `KL +0.48%`，更低更好
- `minority exposure -11.78%`
- `target culture prob +1.24%`

这说明：

1. `listwise` 路线确实比当前 `pairwise BPR-hybrid` 更强一些。
2. 它的优势主要体现在 `ser / calibration / target affinity`。
3. 它在 `minority exposure` 上仍然不如 tuned `BPR-hybrid`。

## 7. 与当前 DCAS 主线的比较

`dcas_full_ot_calibrated_target` 相对 `bpr_listwise_hybrid`：

- `serendipity +63.30%`
- `KL +5.97%`
- `minority exposure +52.37%`
- `target culture prob +15.73%`

因此，即便把非 DCAS 基线继续推进到 `listwise learning-to-rank`，当前 `DCAS calibrated` 仍然保持明显优势。

## 8. 结论

这轮补充后，当前 strongest non-DCAS baseline 的排序可以更新为：

1. `bpr_listwise_hybrid`
2. `bpr_two_stage_hybrid`
3. `bpr_mf`
4. `lightfm_like_hybrid`
5. 第一版 `two_stage_hybrid_ranker`

但更关键的是：

`DCAS calibrated` 仍然在这些强基线之上，而且优势不是单项优势，而是多指标共同领先。

## 9. 研究意义

这条新 baseline 的价值主要有两点：

1. 它让“我们只是打赢了一个弱 hybrid”这个质疑更难成立。
2. 它说明当前 `DCAS` 的优势并不是建立在对手过弱之上，而是在补入更标准的 listwise 排序基线后仍然成立。

下一步如果还要继续加强对手，最值得补的是：

1. 真正树模型式的 ranking baseline
2. 更接近工业多路召回 + learned reranking 的代理系统
3. 若环境允许，重新尝试 `XGBoost / LightGBM` 路线
