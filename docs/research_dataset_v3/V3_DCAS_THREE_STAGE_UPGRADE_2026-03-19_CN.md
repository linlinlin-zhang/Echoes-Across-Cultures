# V3 DCAS 三阶段提升分析与实验记录
更新日期：2026-03-19

## 1. 背景

在 `V3_PIPELINE_UPGRADE_BATCH3_2026-03-18_CN.md` 完成后，`CultureMERT + mw3 + pseudo-PAL + source-aware` 已经成为当前主线，但结果仍存在一个明显问题：

- `dcas_full_ot` 对当前基线仍强，但还不足以证明其相对更强混合推荐系统的必要性。
- 新版系统的主要收益集中在 `serendipity`，而不是四项指标一起提升。
- 训练链路并未直接使用 `interactions` 学习排序或偏好，导致 DCAS 更像“可解耦表示学习器”，而不是“强推荐排序器”。

以 `CultureMERT` 主线为例：

- 旧版 `dcas_full_ot`：
  - `serendipity = 0.8137`
  - `KL = 2.0584`
  - `minority = 0.4027`
  - `target_prob = 0.1856`
- Batch3 版 `dcas_full_ot`：
  - `serendipity = 0.8332`
  - `KL = 2.0699`
  - `minority = 0.2370`
  - `target_prob = 0.1850`

也就是说，Batch3 版主要把收益换到了 `serendipity` 上，而 `minority exposure` 出现了较大回落。

## 2. 当前瓶颈

### 2.1 训练目标偏“表示”，不偏“排序”

当前 `train_model()` 主要优化：

- reconstruction
- KL
- domain adversarial
- contrastive regularization
- covariance / TC / HSIC
- pairwise constraints
- source-aware regularization

但并没有显式使用 `interactions` 去学习用户偏好与排序信号。

### 2.2 约束训练仍然较“软”

当前 pseudo-PAL 约束在训练时是随机抽样，模型只看到一小批随机 pair，缺少对高违约样本的优先压制，因此约束对主排序空间的塑形效率不够高。

### 2.3 训练过程缺少阶段化策略

当前虽然已有 `regularizer_warmup_epochs`，但仍然更像“单阶段训练 + 若干正则项一起上”，缺少：

- 先稳定 latent space
- 再强化 pairwise constraints
- 最后再把用户偏好压进 latent ranking

这样的分段训练逻辑。

## 3. 三阶段提升方案

### 阶段 1：表示学习 warmup

目标：

- 先稳定 `zc / zs / za` 的基础结构
- 让 reconstruction、domain/source-aware、对比正则先把 latent space 站稳

做法：

- 保留当前 `regularizer_warmup_epochs`
- 在前几轮不急着把 constraints 和 ranking 全开

### 阶段 2：hard-mined constraint consolidation

目标：

- 让 pseudo-PAL 约束从“随机碰到一些 pair”变成“优先修 hardest violation”

做法：

- 从更大的约束候选池中抽样
- 根据当前 latent distance 计算违约程度
- 只对最难的那一批 pair 施加强约束

预期：

- 更快压缩正 pair 距离
- 更稳定地拉开负 pair 距离

### 阶段 3：interaction-aware ranking alignment

目标：

- 让 `za` 不再只是一个被动表示空间，而开始直接服务推荐排序

做法：

- 引入基于 `interactions` 的 ranking loss
- 用用户历史的 latent 平均表示作为用户向量
- 用正样本与负样本的距离差做 margin ranking
- 负样本优先采同文化 hard negative，再混入全局 negative

预期：

- 提升主线 `dcas_full_ot / dcas_full_knn` 的 relevance 与稳定性
- 缓解“训练目标与最终推荐目标不一致”的问题

## 4. 本轮实现范围

本轮先实现一版可直接跑的三阶段升级，不引入新 backbone，也不改 benchmark 协议：

1. 在训练器中新增阶段化控制：
   - `constraint_start_epoch`
   - `constraint_warmup_epochs`
   - `rank_start_epoch`
   - `rank_warmup_epochs`
2. 新增 hard-mined constraint loss：
   - `constraint_hard_mining`
   - `constraint_candidate_pool_size`
   - `constraint_batch_size`
3. 新增 interaction-aware ranking loss：
   - `interactions`
   - `lambda_rank`
   - `ranking_batch_size`
   - `ranking_negatives`
   - `ranking_margin`
   - `ranking_same_culture_ratio`

## 5. 本轮实验假设

如果三阶段训练有效，本轮最理想的变化是：

- `serendipity` 持平或继续上升
- `KL` 不再继续恶化
- `minority exposure` 至少部分回补
- `target_prob` 不低于 Batch3

其中最关键的观察点是：

- ranking loss 是否能提升 `dcas_full_ot` 主线，而不是只提升 rerank 分支
- hard-mined constraints 是否能让 `minority` 回升，而不是进一步塌陷

## 6. 运行记录

### 6.1 训练配置

- [train_v3_culturemert_stage3.run.json](/e:/Desktop/Echo/configs/train/train_v3_culturemert_stage3.run.json)

### 6.2 Benchmark 配置

- [recommender_benchmark_v3_culturemert_stage3.run.json](/e:/Desktop/Echo/configs/benchmark/recommender_benchmark_v3_culturemert_stage3.run.json)

### 6.3 结果

本轮训练与 benchmark 已完成。

核心产物：

- 模型：
  - `E:/Desktop/Echo/storage/models/dcas_full_v3_main_culturemert_stage3.pt`
- benchmark：
  - `E:/Desktop/Echo/reports/benchmarks/v3_main_culturemert_stage3/benchmark_summary.json`
  - `E:/Desktop/Echo/reports/benchmarks/v3_main_culturemert_stage3/benchmark_table.md`

主线结果（`dcas_full_ot`）：

- `serendipity = 0.8449`
- `KL = 2.0432`
- `minority = 0.2392`
- `target_prob = 0.1920`

相对 Batch3 主线 `v3_main_culturemert_open_prepal / dcas_full_ot`：

- `serendipity +1.39%`
- `KL +1.29%`（更低、更好）
- `minority exposure +0.93%`
- `target_prob +3.81%`

这说明三阶段升级的第一轮已经把结果从“主要提升 serendipity”推向了“四项一起小幅改善”。

`dcas_full_knn` 也同步改善：

- `serendipity = 0.8454`
- `KL = 2.0417`
- `minority = 0.2395`
- `target_prob = 0.1925`

相对当前强非 DCAS 基线，主线优势仍然清晰：

- 相对 `cosine`，`dcas_full_ot`：
  - `serendipity +52.43%`
  - `KL +6.10%`
  - `minority +5.59%`
  - `target_prob +16.73%`
- 相对 `knn`，`dcas_full_ot`：
  - `serendipity +48.10%`
  - `KL +6.47%`
  - `minority +5.68%`
  - `target_prob +18.24%`
- 相对 `hybrid_content_popularity_diversity`，`dcas_full_ot`：
  - `serendipity +57.37%`
  - `KL +5.57%`
  - `minority +241.62%`
  - `target_prob +15.44%`

`open` 分支仍不是主线：

- `dcas_full_ot_open` 的 `serendipity = 0.2756`
- `KL = 2.3010`
- `minority = 0.3725`
- `target_prob = 0.1328`

也就是说，开放式 rerank 依然更像“换取更多少数项曝光”的分支，而不是整体更优模型。

## 7. 本轮结论

本轮三阶段升级已经证明两件事：

1. `DCAS` 的问题不一定是方法本体错了，而是此前训练目标和推荐目标之间脱节过大。
2. 一旦把训练改为“warmup -> hard constraints -> ranking alignment”，主线结果就能从“单指标提升”变成“四项同步小幅提升”。

当前仍然成立的判断：

- `CultureMERT + DCAS` 依然是主线
- `open` 仍是支线，不应作为主结果
- 接下来最值得继续做的是：
  - 扩大 ranking loss 的设计空间
  - 继续调 hard negative 策略
  - 再补一个更像平台级的强混合基线来验证必要性
