# 面向 ISMIR 2026 的项目重构方案

更新日期：2026-03-12

## 1. 这份方案的目标

这不是原始愿景版项目书的继续扩写，而是一份面向 `ISMIR 2026` 投稿的“收缩版、可执行版、可投稿版”方案。

目标只有一个：

在当前仓库已经具备的 `DCAS 原型 + Route A 实验链 + research_dataset_v1` 基础上，把项目重构成一篇更容易被 ISMIR reviewer 理解、接受和复现的论文。

因此，这份方案会主动做三件事：

1. 收缩研究范围。
2. 重排贡献优先级。
3. 把实现目标从“做一个宏大的全功能系统”改成“做一篇证据闭环的 ISMIR 论文”。

## 2. 先给结论：新的项目方向应该怎么改

### 2.1 新的方向定位

建议把项目重新定位为：

`一个面向跨文化音乐推荐的解纠缠表示与最优传输对齐框架，并通过小规模真实参与式反馈验证其对 serendipity 与文化校准的影响`

英文上可以收敛成类似下面这种方向：

`Disentangled Cross-Cultural Music Recommendation with Optimal Transport Alignment and Pilot Participatory Feedback`

### 2.2 这意味着什么

新的项目不再试图同时证明以下所有事情：

- 你做出了一个新的跨文化基础模型；
- 你完成了 CultureMERT 持续预训练；
- 你做出了高保真的跨文化风格迁移生成器；
- 你定义并解决了完整的认知正义本体系统；
- 你在所有公开 benchmark 上全面领先。

新的项目只做三件主事：

1. 提出并验证一个 `解纠缠 + 领域对抗 + OT` 的跨文化推荐框架。
2. 用统一协议评估 `serendipity / cultural calibration / disentanglement`。
3. 用一轮小规模但真实的 `PAL 专家反馈回灌` 证明“人类反馈对跨文化推荐是有用的”。

如果能把这三件事讲清楚，就已经比当前“大而散”的方案更像一篇能投 ISMIR 的论文。

## 3. 为什么必须这样改

### 3.1 时间窗口不允许再做大一统系统

根据 ISMIR 2026 官方信息，截至 `2026-03-12`：

- Mentoring 申请截止：`2026-03-13 AoE`
- 摘要截止：`2026-04-20 AoE`
- 全文截止：`2026-04-27 AoE`
- 正文科学内容限制：`6` 页
- 参考文献、ethics statement、AI usage statement 可以放到额外页
- PDF 中不允许 appendix
- 双盲严格执行

这对项目方向的直接约束是：

- 你不能再做一个“每个模块都来一点”的论文。
- 你必须让 reviewer 在很短时间里看懂问题、方法、实验和结论。
- 一切没有直接提升说服力的内容，都要降级到次要位置。

### 3.2 当前项目的最大问题不是没东西，而是东西太多

当前仓库已经有：

- 三因子解纠缠；
- 领域对抗；
- OT 推荐；
- PAL 回路；
- pairwise constraints；
- 动态本体；
- CultureMERT embedding；
- 风格迁移接口；
- 统一评测；
- 前后端原型；
- `research_dataset_v1` 真实数据前处理；
- 多阶段 Route A 实验报告。

问题在于：

- 这些东西不能都成为论文主贡献；
- 其中不少部分已经“工程上存在”，但还没达到“论文级证据充分”；
- 如果全都写进去，reviewer 会觉得题目太大、证据太薄、主线不清。

### 3.3 ISMIR 更适合“聚焦问题 + 可复现实验 + 人本价值”

ISMIR 2026 的主题与 call 对你的题目是友好的：

- cross-cultural / computational ethnomusicology
- music recommendation and personalization
- evaluation, datasets, reproducibility
- responsibility and ethics
- cognitive and user-centered MIR

所以关键不是换题，而是把题讲小、讲实、讲清楚。

## 4. 新方案的核心研究问题

建议新的论文只围绕一个主问题展开：

`在跨文化音乐推荐中，如何在保持情感相关性的同时，提高推荐的新颖性，并避免结果退化为目标文化的主流样本？`

围绕这个主问题，再拆成三个更具体的研究问题：

### RQ1

`解纠缠表示 + 领域对抗 + OT 对齐` 是否比简单 embedding 检索更适合跨文化推荐？

关注指标：

- serendipity
- top-k relevance / ranking proxy
- cultural calibration

### RQ2

显式的三因子分离 `zc/zs/za` 是否提供了可复现的、至少初步可信的跨文化潜空间结构？

关注指标：

- MIG
- DCI
- SAP

### RQ3

小规模真实专家反馈是否可以通过 `PAL -> pairwise constraints -> retraining` 改善推荐结果或校准表现？

关注指标：

- 回灌前后 serendipity 变化
- calibration 变化
- case study 变化

## 5. 新方案的主贡献，限定为 3 条

论文中建议只保留以下三条主贡献。

### 贡献 1：方法

提出一个用于跨文化音乐推荐的 `DCAS-lite` 方法框架：

- `zc/zs/za` 三因子表示
- `za` 上领域对抗去文化化
- `za` 空间中做 OT/Sinkhorn 对齐推荐

这里建议把论文里的名字保留为 `DCAS`，但在写作时强调它是：

- `a runnable prototype`
- `a proof-of-concept framework`

不要写成“大一统最终系统”。

### 贡献 2：评测协议

给出一套针对跨文化推荐的统一评测协议：

- 推荐效果指标
- serendipity
- cultural calibration
- disentanglement proxy metrics
- paired significance testing

这会让论文不只是“我有一个模型”，而是“我有一个可复现的评测问题定义”。

### 贡献 3：真实 PAL 试点

做一轮小规模真实专家反馈回灌实验。

这个贡献的重要性很高，因为：

- 它是你与普通推荐论文最不同的地方；
- 它直接贴合 ISMIR 2026 的 human-centered / ethics / culture 方向；
- 它能够把当前“模拟 PAL”的短板补上。

## 6. 哪些方向保留，哪些方向要降级

### 6.1 保留为主线的内容

- 三因子解纠缠 `zc/zs/za`
- 领域对抗
- OT/Sinkhorn 推荐
- PAL 不确定性选样
- pairwise constraints 回灌训练
- serendipity / cultural calibration / disentanglement 评测
- 小规模真实专家标注

### 6.2 保留但降级为配角的内容

- 动态本体
- CultureMERT 作为 embedding backbone
- 前后端 demo
- waveform style transfer baseline

这些内容可以在论文中出现，但只能作为：

- supporting component
- system detail
- qualitative example
- supplementary material

### 6.3 暂时移出主论文的内容

- CultureMERT 持续预训练（CPT）
- Task Arithmetic
- 高保真生成式风格迁移
- GlobalDISCO 生成评测
- 大规模认知正义理论扩展

这些方向不是没价值，而是现在写进主论文会显著稀释主线，并暴露证据短板。

## 7. 新的实验方案

### 7.1 数据方案

#### 主数据

继续使用当前已经做好的：

- `storage/public/research_dataset_v1`

它已经具备：

- 4 个文化域
- 共 1600 条音频
- CultureMERT embedding
- 统一 metadata
- 合成 interactions
- 固定训练/验证/测试切分

这意味着它适合：

- 快速跑 baseline
- 快速做 ablation
- 快速做 PAL 选样与回灌

#### 新增数据需求

为了让论文更可信，建议额外补两类数据：

1. `真实专家标注数据`
2. `至少一个公开 benchmark`

优先级上，真实专家标注比公开 benchmark 更关键，因为它直接影响 PAL 主张。

### 7.2 专家标注方案

建议不要做“大而全”的标注，而做“小而精”的试点。

#### 标注规模建议

- 标注者数量：`2~4`
- 样本对数量：`100~300` 对
- 文化域：优先覆盖当前的 `west / india / turkey / china`

#### 标注内容优先级

优先做以下 3 类：

1. `pairwise similarity / dissimilarity`
2. `文化语义或情感概念标签`
3. `简短 rationale`

示例：

- “这两首相似，不是因为速度，而是因为都呈现礼仪/仪式性功能。”
- “这首更接近 Han / longing / lament，而不是普通 sad。”

#### 为什么这么设计

因为这三类反馈都可以直接进入现有链路：

- pairwise label -> constraints
- concept label -> ontology annotation
- rationale -> 定性分析与案例解释

### 7.3 Baseline 方案

新的论文至少要补以下 baseline。

#### 最低配置 baseline

1. `CultureMERT + cosine retrieval`
2. `CultureMERT + kNN retrieval`
3. `DCAS without OT`
4. `DCAS without domain adversarial`
5. `DCAS without constraints`

#### 建议再补的 baseline

6. `simple MLP scoring head on embeddings`
7. `popularity / frequency baseline`

这样做的好处是：

- reviewer 能看懂你到底比什么强；
- 能拆出 OT、domain、constraints 各自的贡献；
- 不会让论文看起来只是在做“自己跟自己比”。

### 7.4 主实验设计

建议只保留 3 组主实验。

#### 实验 A：主任务推荐实验

目的：

- 证明 DCAS 相比简单 embedding 检索更适合跨文化推荐。

指标建议：

- Recall@K / nDCG@K 或你当前使用的 ranking proxy
- serendipity
- cultural_calibration_kl
- minority exposure@K

#### 实验 B：方法消融实验

比较：

- full model
- no OT
- no domain
- no constraints
- maybe no contrast

目的：

- 明确每个模块的必要性。

#### 实验 C：PAL 回灌实验

比较：

- baseline model
- simulated PAL
- real PAL feedback

目的：

- 证明真实人类反馈不是装饰，而是能带来 measurable benefit。

### 7.5 辅助实验与定性分析

可以保留，但不要占太多正文版面：

- 风格迁移案例图
- ontology concept case study
- 推荐案例表
- 错误分析

这些更适合：

- 一张图
- 一张案例表
- 或 supplementary audio demo

## 8. 论文叙事应该怎么改

### 8.1 从“系统全景”改成“问题驱动”

旧叙事的问题是：

- 模块太多
- 每个模块都想证明
- 结果像系统说明书，不像科研论文

新的叙事顺序应该是：

1. 跨文化推荐为什么难。
2. 现有 embedding 检索为什么会塌缩到目标文化主流样本。
3. 为什么需要把“风格/文化”与“情感/功能”拆开。
4. 为什么在共享语义空间上做 OT 更合理。
5. 为什么 PAL 的真实反馈对这个问题重要。

### 8.2 从“宏大理论宣言”改成“可验证主张”

建议把论文学术语气控制在：

- `we present`
- `we study`
- `we provide preliminary evidence`
- `we evaluate in a pilot setting`

避免：

- `we solve`
- `state-of-the-art`
- `fully validated`
- `comprehensive`

### 8.3 从“后殖民 MIR 宣言”改成“可操作的跨文化研究设计”

伦理和文化立场很重要，但不能压过实验主体。

更好的做法是：

- 在引言中简洁交代价值立场；
- 在 ethics statement 中诚实讨论文化代表性、标注权力关系、数据局限和潜在风险；
- 在正文中用实验支撑，而不是只做强叙事。

## 9. 新方案对应的项目结构调整

### 9.1 仓库中的主线目录

建议把接下来所有与投稿直接相关的工作，围绕以下路径组织：

- `storage/public/research_dataset_v1`
- `reports/ismir2026_*`
- `configs/train/*`
- `docs/ISMIR2026_*`

### 9.2 建议新增的交付物

#### 数据与标注

- `storage/pal/ismir_round1/tasks.jsonl`
- `storage/pal/ismir_round1/annotations.jsonl`
- `storage/pal/ismir_round1/constraints.jsonl`
- `reports/ismir2026_annotation_protocol.md`

#### 实验

- `reports/ismir2026_baselines.md`
- `reports/ismir2026_ablation.md`
- `reports/ismir2026_pal_round1.md`
- `reports/ismir2026_main_results.md`

#### 写作

- `docs/ISMIR2026_ABSTRACT_CN.md`
- `docs/ISMIR2026_PAPER_OUTLINE_CN.md`
- `docs/ISMIR2026_ETHICS_NOTES_CN.md`

### 9.3 暂时不要投入太多时间的方向

- `web/` 的新页面设计
- 更复杂的 UI 交互
- 生成式模块的重写
- 本体系统的重构

原因很简单：

这些事情现在几乎都不会显著提高投稿通过率。

## 10. 具体到论文，应该怎么组织 6 页正文

### Section 1: Introduction

只讲三件事：

- 跨文化推荐中的 representation mismatch
- 推荐系统只追求相关性会牺牲 serendipity 与文化多样性
- 本文的解法：disentanglement + OT + PAL

### Section 2: Method

只保留最必要的方法：

- `zc/zs/za`
- domain adversarial objective
- OT recommendation objective
- PAL constraints feedback

把动态本体和风格迁移只写成 supporting interface。

### Section 3: Experimental Setup

写清楚：

- dataset version
- culture domains
- interactions 来源
- baseline
- metrics
- significance testing

### Section 4: Results

用三类结果撑住：

- main recommendation results
- ablation
- PAL pilot

### Section 5: Discussion

重点讨论：

- why serendipity improved
- why calibration may still lag
- what real human feedback changed

### Section 6: Conclusion

不要写太满，只写：

- a pilot but reproducible framework
- promising evidence
- future work: larger benchmark and stronger expert studies

## 11. 风险与应对

### 风险 1：真实专家标注来不及

应对：

- 立即把第一轮标注范围压缩到 `100~150` 对。
- 只做最有信息量的 pairwise constraints。
- 先保证“有真实反馈”，再追求标注规模。

### 风险 2：公开 benchmark 接不上

应对：

- 论文定位改为 `pilot / proof-of-concept`。
- 明确写自建数据与真实 PAL 是贡献重点。
- 但至少给出清晰的数据协议和复现链。

### 风险 3：公平性指标仍不理想

应对：

- 不强行包装成全面公平；
- 诚实写成 trade-off；
- 强调你解决的是“高 surprise + 保留相关性”的跨文化推荐问题，而不是所有 fairness 问题都解决。

### 风险 4：生成模块拖慢主线

应对：

- 直接降级为 supplementary demo。

## 12. 从现在到投稿前的执行路线

### 第 0 步：立即完成

- 确定新题目和新贡献边界。
- 不再继续扩主功能。
- 若符合条件，尽快申请 mentoring。

### 第 1 周

- 在 `research_dataset_v1` 上跑完整 baseline。
- 补 `ismir2026_baselines` 报告。
- 确定第一轮专家标注协议。

### 第 2 周

- 完成第一轮真实 PAL 标注。
- 生成 `constraints.jsonl`。
- 跑回灌训练。

### 第 3 周

- 补主实验和 ablation。
- 固化统计显著性结果。
- 选 2~3 个最强案例做定性分析。

### 第 4 周

- 如果可能，补一个公开 benchmark。
- 写摘要初稿与论文提纲。

### 第 5 周

- 完成论文主体写作。
- 压缩到 6 页正文。
- 增加 ethics statement。

### 第 6 周

- 做匿名化。
- 整理 supplementary 音频样例和复现命令。
- 全面检查 claim 与 evidence 是否一一对应。

## 13. 最终建议

新的面向 ISMIR 方案，不应该再是：

`一个跨文化音乐理解、生成、推荐、认知正义、本体、基础模型、风格迁移的大一统系统`

而应该是：

`一个聚焦跨文化推荐的研究原型，核心方法是解纠缠表示与 OT 对齐，并通过一轮真实 PAL 试点证明人类反馈能够改进推荐行为`

这是一个明显更窄、但也明显更有投稿可行性的方向。

一句话总结：

把项目从“做尽可能多的事”，改成“证明少数几件关键的事”。

## 14. 官方参考

以下信息基于 ISMIR 2026 官方页面，截至 `2026-03-12`：

- Call for Papers: `https://ismir2026.ismir.net/authors/call-for-papers`
- Author Guidelines: `https://ismir2026.ismir.net/authors/author-guidelines`
- Mentoring Program: `https://ismir2026.ismir.net/mentoring`
