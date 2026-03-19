# V3 强混合基线测试记录
更新日期：2026-03-19

## 1. 目标

在完成 `DCAS` 三阶段升级后，继续推进“更像主流平台的强混合基线”，避免只拿 `cosine / knn / heuristic hybrid` 作为主要对手。

本轮目标不是直接复刻工业系统，而是在当前仓库与依赖范围内，先构建一个更接近两阶段推荐的可复现代理基线：

- 第一阶段：多特征召回
- 第二阶段：学习式 reranker
- 训练时使用 hard negatives

## 2. 实现方式

本轮新增了一个 `two_stage_hybrid_ranker`，代码位于：

- `E:/Desktop/Echo/dcas/embedding_recommenders.py`
- `E:/Desktop/Echo/dcas/scripts/run_recommender_benchmarks.py`

设计要点：

1. 候选召回
   - 基于 `cosine / knn / popularity / source preference / target-affinity` 等标量特征构建召回分数
   - 在目标文化候选池中取 `top recall_k`

2. 学习式重排
   - 使用 `10` 个手工特征
   - 再拼接 `user/item/abs-diff/product` 交互特征
   - 最终输入维度为 `3082`
   - 使用 `3` 层 `MLP`，隐藏维度 `256`

3. 训练样本构造
   - 使用 `mixed interactions`
   - 以正样本所属文化作为目标文化
   - 先做候选召回，再从高分候选中采 hard negatives

本轮 benchmark 配置：

- [recommender_benchmark_v3_culturemert_stage3_stronghybrid.run.json](/e:/Desktop/Echo/configs/benchmark/recommender_benchmark_v3_culturemert_stage3_stronghybrid.run.json)

产物：

- 模型：
  - `E:/Desktop/Echo/storage/models/two_stage_hybrid_v3_main_culturemert_stage3_v2.pt`
- benchmark：
  - `E:/Desktop/Echo/reports/benchmarks/v3_main_culturemert_stage3_stronghybrid/benchmark_summary.json`
  - `E:/Desktop/Echo/reports/benchmarks/v3_main_culturemert_stage3_stronghybrid/benchmark_table.md`

## 3. 结果

`two_stage_hybrid_ranker`：

- `serendipity = 0.5255`
- `KL = 2.1522`
- `minority = 0.0221`
- `target_prob = 0.1678`

对比当前 `DCAS stage3 / dcas_full_ot`：

- `DCAS` 相对该强混合基线：
  - `serendipity +60.78%`
  - `KL +5.06%`
  - `minority exposure +980.15%`
  - `target_prob +14.43%`

对比当前仓库原有 heuristic hybrid：

- `two_stage_hybrid_ranker` 相对 `hybrid_content_popularity_diversity`：
  - `serendipity -2.12%`
  - `KL +0.54%`
  - `minority exposure -68.37%`
  - `target_prob +0.89%`

也就是说，这条“更像平台”的第一版 learned hybrid，并没有成为更强对手，甚至在当前协议下还弱于已有 heuristic hybrid。

## 4. 解释

这个结果很重要，因为它说明问题不只是“基线还不够平台化”，还包括当前数据与协议本身对这类基线并不友好：

1. 当前 `interactions` 仍然是合成/弱监督的
   - 对 learned hybrid 来说，训练信号可能不够稳定
   - 模型容易学到过强的 popularity / source 偏置

2. 当前 raw baseline 的候选池仍是“目标文化闭集”
   - 这会削弱平台式 multi-stage 系统的一个核心优势
   - 也让强 hybrid 的 candidate generation 空间被压缩

3. 当前 learned hybrid 仍是仓库内代理版
   - 它比 heuristic hybrid 更接近平台思路
   - 但还不是 `LightFM / BPR / LambdaMART / GBDT reranker` 这类真正常见的强基线

## 5. 结论

本轮第二步的结论不是“强混合基线已经打败 DCAS”，而是：

1. 在当前 `CultureMERT stage3` 协议下，`DCAS` 仍然明显强于第一版两阶段 learned hybrid。
2. 这说明 `DCAS` 的提升不是完全建立在“只挑弱基线比较”上。
3. 但这还不能替代真正的强平台基线验证；下一步仍应补：
   - `BPR / LightFM` 一类协同过滤基线
   - 更标准的 `GBDT / LambdaMART` 风格 reranker
   - 或更开放的 candidate generation 协议
