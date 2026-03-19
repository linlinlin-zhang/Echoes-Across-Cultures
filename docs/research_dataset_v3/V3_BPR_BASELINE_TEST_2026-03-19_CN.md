# V3 BPR-MF 基线测试记录
更新日期：2026-03-19

## 1. 目标

在第一版两阶段 learned hybrid 未能形成强对手后，继续补一个 reviewer 更熟悉、也更标准的经典协同过滤基线：`BPR-MF`。

这一步的目标不是证明 `BPR` 一定比 `DCAS` 强，而是验证：

- 当比较对象从 `cosine / knn / heuristic hybrid` 升级到标准 `CF` 时，`DCAS` 是否仍站得住；
- `DCAS` 的优势到底主要落在什么指标上。

## 2. 实现方式

本轮在以下文件中新增了 `BPR-MF`：

- `E:/Desktop/Echo/dcas/embedding_recommenders.py`
- `E:/Desktop/Echo/dcas/scripts/run_recommender_benchmarks.py`

实现要点：

- 用户因子：`240`
- 物品因子：`1106`
- latent 维度：`64`
- 目标函数：标准 `BPR` pairwise loss
- 正则：`1e-4`
- 训练轮数：`12`

本轮 benchmark 配置：

- [recommender_benchmark_v3_culturemert_stage3_bpr.run.json](/e:/Desktop/Echo/configs/benchmark/recommender_benchmark_v3_culturemert_stage3_bpr.run.json)

产物：

- 模型：
  - `E:/Desktop/Echo/storage/models/bpr_mf_v3_main_culturemert_stage3.pt`
- benchmark：
  - `E:/Desktop/Echo/reports/benchmarks/v3_main_culturemert_stage3_bpr/benchmark_summary.json`
  - `E:/Desktop/Echo/reports/benchmarks/v3_main_culturemert_stage3_bpr/benchmark_table.md`

## 3. 结果

`bpr_mf`：

- `serendipity = 0.4916`
- `KL = 2.0226`
- `minority = 0.1491`
- `target_prob = 0.1957`

相对 heuristic hybrid：

- `serendipity -8.43%`
- `KL +6.53%`
- `minority exposure +113.00%`
- `target_prob +17.69%`

也就是说，`BPR` 并不是一个全弱基线。它虽然在 `serendipity` 上比 heuristic hybrid 差，但在：

- `cultural calibration`
- `minority exposure`
- `target culture probability`

这三项上都明显更强。

相对 `DCAS stage3 / dcas_full_ot`：

- `DCAS` 相对 `BPR`：
  - `serendipity +71.86%`
  - `minority exposure +60.39%`
- 但 `BPR` 相对 `DCAS`：
  - `KL +1.02%`（更低、更好）
  - `target_prob +1.91%`

也就是说，`BPR` 是目前为止最值得认真对待的经典协同过滤对手之一：

- 它没有打败 `DCAS`
- 但它在 `KL` 与 `target_prob` 上已经能与 `DCAS` 正面对打，甚至略优

## 4. 解释

这个结果说明 `DCAS` 的优势并不是“全指标绝对最优”，而更像：

- 在 `serendipity` 上优势非常大
- 在 `minority exposure` 上优势也明显
- 但在更传统的偏好对齐与概率聚焦上，经典 `CF` 仍然有竞争力

换句话说：

`DCAS` 的价值更像是在扩展 Pareto frontier，而不是单纯替代所有传统推荐算法。

## 5. 结论

本轮 `BPR-MF` 测试带来了一个更强、也更可信的结论：

1. `DCAS` 在面对标准 `CF` 基线时，仍然在 `serendipity` 和 `minority exposure` 上保持显著优势。
2. `BPR` 在 `KL` 与 `target_prob` 上略优，说明论文里不能把 `DCAS` 写成“全面碾压传统方法”。
3. 更准确的叙述应该是：
   - `DCAS` 擅长把推荐从“相关但保守”推向“更有惊喜且更兼顾长尾”
   - `BPR` 仍然是更保守但更贴近传统偏好建模的一条强基线

下一步如果继续补强基线，最值得做的是：

- `LightFM`
- `GBDT / LambdaMART` 风格 reranker
- 或更开放的 candidate generation 协议
