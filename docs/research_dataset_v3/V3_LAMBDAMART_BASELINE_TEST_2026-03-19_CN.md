# V3 LambdaMART / Tree-based Baseline 测试记录（2026-03-19）

## 1. 背景

在补完 `BPR-MF`、`BPR two-stage hybrid`、`BPR listwise hybrid` 之后，还缺一条更标准的 `tree-based ranking` 强基线。

这条基线的目标不是继续做 `MLP` 代理，而是回答：

- 当前 `DCAS calibrated` 是否仍然能压过真正的 `LambdaMART / tree-based` reranker？
- 当前环境是否已经具备 `sklearn / xgboost / lightgbm` 的可复现实验条件？

## 2. 环境修复

本轮没有去修改 Anaconda base 环境，而是将实验环境切换到仓库已有的 `.venv-gpu`：

- `numpy 2.4.3`
- `scipy 1.17.1`
- `sklearn 1.8.0`
- `xgboost 3.2.0`
- `lightgbm 4.6.0`
- `torch 2.8.0+cu128`

因此，原先 base 环境中的 `NumPy / SciPy / sklearn` ABI 问题已经绕开；后续树模型实验统一建议使用 `.venv-gpu/Scripts/python.exe`。

## 3. 实现方式

### 3.1 训练思路

新基线命名为：

`bpr_lambdamart_hybrid`

核心结构：

1. 第一阶段继续使用 `BPR-MF` 做候选召回。
2. 第二阶段使用 `LightGBM LGBMRanker(objective='lambdarank')` 做真正的 `LambdaMART` 学习排序。
3. 训练样本复用现有 `BPR listwise` 查询构造逻辑，但为了避免内存爆炸，树模型只使用紧凑 `feature table`，不再直接拼接高维 `user/item/diff/product` embedding 块。

### 3.2 使用的 rerank 特征

树模型实际使用的是紧凑标量特征表，主要包含：

- `cosine`
- `knn`
- `max_hist`
- `mean_hist`
- `novelty`
- `popularity`
- `minority`
- `target_affinity`
- `source_pref`
- `source_inv`
- `bpr_score`
- `recall_score`

这更接近工业排序系统里常见的 feature-table 输入形态，也更符合 `LambdaMART` 的使用习惯。

## 4. 代码与配置

新增代码：

- `dcas/embedding_recommenders.py`
  - `train_bpr_tree_hybrid_ranker`
  - `load_bpr_tree_hybrid_ranker`
  - `recommend_embedding_bpr_tree_hybrid`

- `dcas/scripts/run_recommender_benchmarks.py`
  - 新增 `kind = "bpr_tree_hybrid"` 路径
  - 修复 `BPR` 系列方法在 CUDA 环境下的 device mismatch 问题

新增配置：

- `configs/benchmark/recommender_benchmark_v3_culturemert_stage3_lambdamart.run.json`

产物：

- `storage/models/bpr_lambdamart_hybrid_v3_main_culturemert_stage3.pkl`
- `reports/benchmarks/v3_main_culturemert_stage3_lambdamart/benchmark_summary.json`
- `reports/benchmarks/v3_main_culturemert_stage3_lambdamart/benchmark_table.md`

## 5. 结果

### 5.1 主要方法结果

| method | serendipity | calibration_kl | minority@k | target_prob |
|---|---:|---:|---:|---:|
| `bpr_mf` | 0.4916 | 2.0226 | 0.1491 | 0.1957 |
| `bpr_two_stage_hybrid` | 0.5102 | 2.0082 | 0.2838 | 0.2005 |
| `bpr_listwise_hybrid` | 0.5135 | 1.9986 | 0.2503 | 0.2029 |
| `bpr_lambdamart_hybrid` | 0.5107 | 1.9967 | 0.2662 | 0.2038 |
| `dcas_full_ot_calibrated_target` | 0.8386 | 1.8793 | 0.3814 | 0.2349 |
| `dcas_full_ot_calibrated_minor` | 0.8404 | 1.9148 | 0.5190 | 0.2250 |

### 5.2 与最强非 DCAS 基线的关系

`bpr_lambdamart_hybrid` 相对 `bpr_listwise_hybrid`：

- `serendipity -0.55%`
- `KL +0.09%`，更低更好
- `minority exposure +6.34%`
- `target culture prob +0.42%`

这说明：

- `LambdaMART` 没有在 `serendipity` 上反超 listwise MLP
- 但它在 `KL / minority / target` 上更均衡一些
- 因此它可以视为当前最强、最标准的非 DCAS baseline 候选

### 5.3 与当前主线 DCAS 的关系

`dcas_full_ot_calibrated_target` 相对 `bpr_lambdamart_hybrid`：

- `serendipity +64.21%`
- `KL +5.88%`，更低更好
- `minority exposure +43.29%`
- `target culture prob +15.25%`

`dcas_full_ot_calibrated_minor` 相对 `bpr_lambdamart_hybrid`：

- `serendipity +64.57%`
- `KL +4.10%`，更低更好
- `minority exposure +94.95%`
- `target culture prob +10.41%`

## 6. 结论

本轮结果有三个关键信息：

1. 现在已经真正补上了 `tree-based / LambdaMART` 强排序基线，不再只是 `MLP` 代理。
2. `bpr_lambdamart_hybrid` 比 `bpr_listwise_hybrid` 更像标准树模型 reranker，但仍未反超 `DCAS calibrated`。
3. 当前 `DCAS calibrated` 的优势并不是建立在“没有强排序基线”之上；即使补进 `LambdaMART`，它在当前 `CultureMERT` 小规模设定下仍保持明显领先。

## 7. 后续建议

1. 若继续补强非 DCAS 对手，可再试 `xgboost rank:ndcg` 同协议对照。
2. 若要进一步增强论文说服力，下一步比继续调树模型更值钱的是：
   - 一个公开 benchmark
   - 或一轮真实 PAL / 真实用户反馈证据
3. 需要在论文中明确说明：当前推荐实验依赖合成交互，因此这些结果应被写成 `small-scale / synthetic-interaction evidence`，而不是大规模真实用户结论。
