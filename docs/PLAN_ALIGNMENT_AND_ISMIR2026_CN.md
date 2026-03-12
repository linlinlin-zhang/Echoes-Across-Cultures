# 原计划对照、当前完成度与 ISMIR 2026 投稿调整建议

更新日期：2026-03-12

## 1. 这份文档要回答什么

本文件回答四个问题：

1. 最初写在 `声音.txt` 里的研究计划，现在完成到了哪一步。
2. 哪些模块已经落地，哪些还只是概念、接口或部分实现。
3. 当前项目有没有偏离最初计划，如果有，偏离在哪里、偏离程度多大。
4. 如果目标是投稿 ISMIR 2026，现在应该如何收缩题目、补实验、调整论文叙事。

这里的判断不是按“仓库里有没有某个文件”来算，而是按三层标准区分：

- `概念提出`：原计划里写了，但仓库没有可靠证据支撑。
- `工程落地`：代码、脚本、报告已存在，链路能跑通。
- `论文级证据`：有可复现实验、合适基线、评测设计与足够强的论证。

## 2. 一句话结论

如果按“核心研究原型是否已经做出来”计算，当前完成度大约是 `70%`。

如果按“是否已经达到原计划中那篇完整论文的证据强度”计算，当前完成度更接近 `55%~60%`。

也就是说：

- 方向没有跑偏。
- 重点发生了收缩。
- 项目已经从“概念草图”推进到了“可运行研究原型”。
- 但距离“ISMIR 级别的完整论文证据”还差一截，尤其差在真实标注、公开 benchmark、强基线和更聚焦的论文叙事。

## 3. 总体完成度总表

| 模块 | 原计划目标 | 当前状态 | 完成度估计 | 结论 |
|---|---|---:|---:|---|
| 理论重构（DDRL 替代五维启发式） | 用 `zc/zs/za` 取代显式五维规则 | 已落地到模型与训练流程 | 85% | 核心方向已经成立 |
| DCAS 主架构 | 解纠缠 + 领域对抗 + 对比学习 + OT 推荐 | 已实现并跑到 Phase 4 | 80% | 是当前项目最完整的部分 |
| CultureMERT 路线 | 用 CultureMERT 作为 backbone，并考虑更深层 adaptation | 已完成 embedding 接入；未完成 CPT/更深 adaptation | 45% | 做到了“接入”，没做到“充分利用” |
| OT / Sinkhorn 跨文化对齐 | 用 OT 做流形对齐推荐 | 已实现并有结果 | 80% | 已从想法变成主链 |
| 风格迁移 / 反事实生成 | 用 `zc/zs` 做解释性生成 | 仅部分实现（embedding 级 + 波形基线） | 35% | 不能作为论文主贡献来讲 |
| PAL 主动学习 | 不确定性采样 + 专家回路 | 原型实现完成；真实专家闭环未完成 | 60% | 机制有了，证据不够 |
| 动态本体 | 支持概念扩展、关系、训练约束导出 | 已实现工程 v1 | 60% | 系统存在，但仍偏轻量 |
| 解纠缠评测 | MIG/DCI/SAP 等标准评测 | 已实现多 seed 评测 | 75% | 已经足够作为实验部分的一环 |
| 推荐评测 | Serendipity、公平性、显著性检验 | 已实现统一评测 | 80% | 工程上完成度较高 |
| 公共 benchmark 验证 | GlobalMood / CultureMERT benchmarks / GlobalDISCO | 基本未完成 | 20% | 这是当前最大缺口 |
| 真实数据建设 | 至少多个文化域的真实音频与前处理 | 已做出 `research_dataset_v1` | 65% | 数据建设已起步，但还没支撑论文主张 |
| 工程化闭环 | 数据、训练、推荐、PAL、API、前端 | 已有可运行闭环 | 85% | 原型系统层面超出最初草图 |

## 4. 按原计划逐项对照

### 4.1 理论框架：DDRL / 三因子潜变量

原计划在 `声音.txt` 中最核心的变化，是废弃“五维启发式”，改为三因子潜变量 `zc/zs/za` 的深度解纠缠框架。

当前状态：

- 这一部分已经不是文案，而是模型主干。
- `dcas/models/dcas_vae.py` 已经承载了三因子建模。
- 领域对抗、对比学习、重构训练、约束回灌都已经进入训练链路。
- `docs/PAPER_CLAIM_ALIGNMENT.md` 也明确把三因子解纠缠、领域对抗、InfoNCE 这些列为“已实现”。

判断：

- 这块是项目最扎实的主线之一。
- 如果论文主轴聚焦在“跨文化音乐推荐中的解纠缠表示与对齐”，这部分是可以撑住的。

完成度判断：`85%`

还差什么：

- 需要更清晰的 ablation，把 `domain / contrast / constraints / OT` 的作用拆开讲清楚。
- 需要更稳的解纠缠证据，而不是只展示“模型存在”。

### 4.2 DCAS 主架构：领域对抗 + OT 推荐 + 训练闭环

原计划中的 DCAS 架构强调：

- backbone 表征；
- `za` 上去文化化；
- OT/Sinkhorn 做跨文化对齐推荐；
- 将推荐问题从“规则匹配”改成“流形对齐”。

当前状态：

- `dcas/recommender.py`、`dcas/ot/sinkhorn.py` 已经把 OT 推荐主链落地。
- `docs/PAPER_CLAIM_ALIGNMENT.md` 明确把 OT/Sinkhorn 标记为“已实现”。
- Phase 4 推荐实验报告已存在，说明这不是一次性 demo，而是迭代过的实验链。
- `reports/routeA_recommender_compare_phase4_cn.md` 显示 `serendipity_mean` 相比 Phase 2 有明显提升。

判断：

- 从“有没有”角度看，这部分已经做成。
- 从“能不能发 ISMIR”角度看，这部分还需要更标准的 baseline 与更可信的数据设定。

完成度判断：`80%`

还差什么：

- 至少补上 2 到 3 个有说服力的 baseline。
- 需要把“为什么 OT 比简单最近邻/线性映射更合适”用实验说清楚。

### 4.3 CultureMERT 路线

原计划对 CultureMERT 的设想比较激进，不只是把它拿来抽 embedding，而是希望把它作为跨文化 backbone，并进一步考虑持续预训练、零样本泛化与任务算术。

当前状态：

- `dcas/embeddings/culturemert.py` 和 `dcas/scripts/build_tracks_from_audio.py` 已经把 CultureMERT 接入进来了。
- 真实音频到 `tracks.npz` 的链路已经跑通。
- 但 `docs/PAPER_CLAIM_ALIGNMENT.md` 和 `docs/technical_notes/point2_missing_parts_end.md` 都明确写了：当前仅为 `embedding` 级接入，不可宣称已完成 CPT。
- 仓库里没有足够证据表明 `Task Arithmetic` 已经真正落地。

判断：

- 如果标准是“项目是否已经在用 CultureMERT”，答案是是。
- 如果标准是“是否完成了原计划里那条更完整、更强的 CultureMERT 研究路线”，答案是否。

完成度判断：`45%`

还差什么：

- 不一定必须做 CPT，但必须决定是否保留这条主张。
- 如果投稿周期来不及，不要把论文卖点放在“CultureMERT adaptation”上。
- 更现实的写法是：`we build on CultureMERT embeddings as a cross-cultural audio foundation representation`。

### 4.4 风格迁移 / 反事实生成

原计划把这一块放得很高，希望通过生成模块解释推荐理由，并验证解纠缠是否真的把“内容”和“风格”分开了。

当前状态：

- `dcas/style_transfer.py`、`dcas/waveform_style_transfer.py` 等脚本已存在。
- README 也给出了调用接口。
- 但现有实现主要是 embedding 级反事实和波形级基线，不是高保真、可训练、可听感验证的生成系统。
- `docs/PAPER_CLAIM_ALIGNMENT.md` 明确提醒：不可写成“高保真生成器已完成”。

判断：

- 这部分只能算“配套实验接口已存在”。
- 不能把它当成论文核心贡献。

完成度判断：`35%`

还差什么：

- 如果坚持写进论文，最多作为 `qualitative illustration` 或补充材料。
- 若想成为主结果，需要真正的音频生成质量、听测、内容保留度与风格转换度评估。

### 4.5 PAL：从田野调查到参与式主动学习

原计划非常强调这一点，希望用 PAL 替代传统的静态“先标注再训练”方式，把人类学意义上的参与性嵌入训练回路。

当前状态：

- `dcas/pal/uncertainty.py`、`dcas/cli/pal_loop.py` 已实现。
- `dcas/pal/constraints.py` 和训练链路支持 pairwise constraints 回灌。
- Route A Phase 3 文档表明，PAL 两轮注入和评估流程已经搭建。
- 但 `docs/ROUTE_A_PHASE3_RUNBOOK.md` 明确写明：当前 PAL 反馈仍由 metadata label agreement 模拟，而不是真实专家标注。

判断：

- 机制做出来了，闭环也能跑。
- 但这还不是原计划真正想要的“专家在回路中”。

完成度判断：`60%`

还差什么：

- 至少要有一轮小规模真实专家标注。
- 最好有“随机抽样标注 vs PAL 选样标注”的收益比较。
- 这部分如果补上，会直接提高论文的独特性和 ISMIR 契合度。

### 4.6 动态本体

原计划里，本体工程不是可有可无的附属模块，而是“认知正义”叙事的重要承载点。

当前状态：

- `dcas/ontology.py`、CLI、API 都在。
- 支持 concept / relation / annotation / constraints export。
- 但目前仍属于轻量工程实现，尚未看到更强的多模态或 LLM 级语义扩展。

判断：

- 已经做出一个能工作的系统接口。
- 但离“本体工程成为主要科研结论”还很远。

完成度判断：`60%`

还差什么：

- 需要真实标注示例支撑：专家如何新增概念、这些概念如何改变训练或推荐结果。
- 否则论文里只能作为系统功能点，不宜作为主贡献。

### 4.7 解纠缠评测：MIG / DCI / SAP

原计划希望用标准指标来证明潜变量不是“命名好看”，而是真的分离。

当前状态：

- `dcas/scripts/evaluate_disentanglement.py` 和多 seed 协议已经存在。
- `reports/routeA_disentanglement_sharedfactors_compare_phase4_cn.md` 已经对比 Phase 2 / 3 / 4。
- `docs/PAPER_CLAIM_ALIGNMENT.md` 把这一项标成“已实现（工程 v2）”。

判断：

- 这部分已经有足够的工程基础，能进论文实验。
- 问题不在“能不能评”，而在“结果够不够强、够不够稳定”。

完成度判断：`75%`

还差什么：

- 当前并不是所有指标都单调变好。
- 要避免写成“全面证明了解纠缠更优”，更适合写成“提供了初步且可复现的证据”。

### 4.8 推荐评测：Serendipity / 公平性 / 显著性

原计划很强调：不能只看准确率，要看 serendipity 和 cultural fairness。

当前状态：

- 这部分已经是目前仓库里最成熟的评测链之一。
- `dcas/cli/eval.py` 和比较脚本已存在。
- `reports/eval_suite_phase4_v2/eval_suite_summary.json` 中，Phase 4 的 `serendipity_mean` 已经达到 `0.8726` 左右。
- 同时，`cultural_calibration_kl_mean` 仍然比较高，说明“文化校准”问题并没有和 serendipity 一起被解决。

判断：

- 这部分是可以写进论文的。
- 但必须诚实：当前提升主要集中在 surprise / serendipity，不是全面解决了公平性。

完成度判断：`80%`

还差什么：

- 需要更强的 baseline。
- 需要更好地解释指标之间的 trade-off。
- 需要把“公平性”从口号变成更可辩护的实验设计。

### 4.9 公共 benchmark 与大规模验证

这是原计划和当前状态之间最大的落差。

原计划里明确提到了：

- GlobalMood
- CultureMERT Benchmarks
- GlobalDISCO
- 大规模跨文化 benchmark 验证

当前状态：

- 仓库里没有看到这些 benchmark 被系统跑完的证据。
- `docs/PAPER_CLAIM_ALIGNMENT.md` 也明确写了：大规模真实跨文化基准全面实验未实现。
- `docs/technical_notes/point2_missing_parts_end.md` 也明确把“增加公开基准实验”列为后续项。

判断：

- 这是当前最关键的缺口。
- 也是最影响 ISMIR 投稿可信度的部分。

完成度判断：`20%`

还差什么：

- 至少跑 1 个可公开复现的 benchmark。
- 最好不是只跑自建数据。
- 如果完全来不及，则必须把论文重新定位成 `prototype / pilot / preliminary evidence`，而不是 benchmark paper。

### 4.10 真实数据建设

原计划强调从 toy 和启发式设定走向真实多文化数据。

当前状态：

- 已经构建了 `storage/public/research_dataset_v1/`。
- 当前数据集包含 4 个文化域、1600 条曲目、CultureMERT embedding、统一 metadata、弱监督 interactions 和固定切分。
- `reports/research_dataset_v1_profile.md` 与 `reports/research_dataset_v1_splits/split_report.md` 表明这条数据前处理链已经健康跑通。
- 但 profile 同时指出：`affect_label` 缺失，这会直接限制 `za` 相关的情感评估。

判断：

- 数据工作已经从零推进到了“可训练”。
- 但还没推进到“可作为论文核心证据”。

完成度判断：`65%`

还差什么：

- 需要补 affect / concept / pairwise expert signals 中至少一类真实监督。
- 当前的 interactions 也还是弱监督，不是真实用户日志。

## 5. 当前项目相对原计划的偏离情况

### 5.1 是否跑偏

没有在方向上跑偏。

项目仍然紧贴原计划的主轴：

- 跨文化音乐理解；
- 解纠缠表示；
- OT 对齐；
- PAL；
- 公平性与 serendipity；
- 本体与文化语义。

真正发生变化的，不是研究方向，而是实现优先级。

### 5.2 实际偏离发生在哪里

偏离主要集中在以下三点：

1. 从“完整科研蓝图”偏向“先做可运行原型”。
2. 从“公共 benchmark + 大规模验证”偏向“自建数据 + 小规模原型验证”。
3. 从“专家真实参与”偏向“先用模拟 PAL 把工程链路跑通”。

### 5.3 偏离幅度如何

如果一定要量化，我给这个偏离一个 `25%~35%` 的幅度。

更准确地说：

- `理论方向偏离`：小。
- `实现范围偏离`：中等。
- `论文证据偏离`：中等偏大。

一句话概括就是：

不是做错了，而是做窄了；不是方向反了，而是先把最能跑通的部分做深了。

## 6. 如果目标是 ISMIR 2026，现在该怎么调整

### 6.1 先看官方要求与时间窗口

根据 ISMIR 2026 官方网站，截至 `2026-03-12` 可确认的信息如下：

- 会议主题是 `Crossroads`，强调东西交汇、跨文化、人本维度、社会影响与伦理。
- 摘要截止是 `2026-04-20 AoE`。
- 全文截止是 `2026-04-27 AoE`。
- 对中国时区（UTC+8）来说，分别约等于：
  - `2026-04-21 19:59`（摘要）
  - `2026-04-28 19:59`（全文）
- 论文格式是 `6+N`：
  - 正文科学内容最多 6 页
  - 参考文献、可选 ethics statement、可选 AI usage statement 可放到额外页
  - PDF 里不允许 appendix
- 双盲严格执行。
- 推荐系统、用户中心评测、数据集与可复现性、责任与伦理，都是官方明确欢迎的话题。

如果你是第一次投 ISMIR，而且满足官方 mentoring program 条件，那么还有一个很重要的时间点：

- Mentoring program 申请截止是 `2026-03-13 AoE`
- 对中国时区约等于 `2026-03-14 19:59`

这意味着：从现在到正式全文截止，窗口只有大约六周多一点，不能再按“大而全”的路线推进。

### 6.2 这个项目适不适合 ISMIR 2026

适合，但前提是你要主动收缩题目。

这个项目天然契合 ISMIR 2026 的几个重点：

- cross-cultural MIR
- recommendation / personalization
- human-centric evaluation
- responsibility and ethics
- datasets / reproducibility

真正的问题不是“方向不对”，而是“当前论文叙事太大，证据还不够支撑全部叙事”。

### 6.3 当前版本最不适合直接投稿的地方

如果照原计划原封不动去写，风险很大，原因有六个：

1. 题目太大
   当前项目同时想讲解纠缠、CultureMERT、OT、PAL、本体、风格迁移、公平性、生成、认知正义。这在 6 页正文里很容易失焦。

2. 公开 benchmark 证据不足
   没有充分跑完原计划里的公开 benchmark，会让 reviewer 质疑外部有效性。

3. 真实标注证据不足
   当前 PAL 主要还是模拟反馈，难以支撑“参与式”这条强主张。

4. 推荐数据设定偏弱
   当前 interactions 很大程度上是弱监督/合成的，不是真实用户日志。这个需要诚实处理。

5. `za` 的监督支撑不够
   `research_dataset_v1` 当前缺 `affect_label`，这会削弱“跨文化情感对齐”的说服力。

6. 风格迁移会分散主线
   现有生成部分还不够强，如果写重了，会变成 reviewer 的攻击点。

### 6.4 最推荐的投稿定位

最推荐的不是“全景式宏大论文”，而是一个更收敛的 ISMIR paper：

`A cross-cultural music recommendation framework with disentangled representation, OT alignment, and a pilot participatory feedback loop`

更具体一点，可以压缩成三条贡献：

1. 提出一个用于跨文化推荐的解纠缠表示与 OT 对齐框架。
2. 给出一套 serendipity / cultural calibration / disentanglement 的统一评测协议。
3. 展示一个小规模但真实的 PAL 专家反馈回灌实验，证明人类反馈能够改进模型。

这种写法的好处是：

- 保住了你的原创思想主轴。
- 贴合 ISMIR 2026 的人本和跨文化主题。
- 避开了当前最弱的“高保真生成”和“大 benchmark 全覆盖”。

### 6.5 不推荐的投稿定位

以下几种写法当前都不建议：

1. 把论文写成“完整的跨文化音乐基础模型方案”
   因为你没有完成 CPT、任务算术、公开大 benchmark 和大规模泛化验证。

2. 把论文写成“强生成论文”
   因为风格迁移目前还不是强证据。

3. 把论文写成“全面公平性论文”
   因为目前公平性指标还没有被系统性做强，尤其 cultural calibration 仍偏弱。

### 6.6 论文里应该删弱哪些部分

建议在正文里弱化或降级以下内容：

- `Task Arithmetic`
- `CultureMERT CPT / 持续预训练`
- `GlobalDISCO` 生成评测
- 高保真波形级风格迁移
- 过强的“认知正义系统已完整落地”叙事

这些内容不是完全不能提，而是更适合：

- 放 future work
- 放补充材料
- 放系统 demo 附件
- 或仅作为方法设计动机出现

### 6.7 论文里应该强化哪些部分

最该强化的是以下四块：

1. 一个清晰、可复现的主问题
   例如：如何在跨文化音乐推荐中，同时保留情感相关性并提升新颖性，而不过度塌缩到目标文化主流样本。

2. 一个聚焦的方法主线
   `disentanglement + domain adversarial + OT + PAL constraints`

3. 一个可信的数据与评测协议
   至少包括：
   - 数据版本
   - 训练/验证/测试切分
   - baseline
   - serendipity / calibration / disentanglement 指标
   - 统计显著性

4. 一个真实的人在回路中的证据点
   哪怕规模不大，也比“全模拟 PAL”更有说服力。

## 7. 按 ISMIR 2026 口径，投稿前必须补的东西

下面这些不是“锦上添花”，而是最值得优先补的缺口。

### 7.1 必做：补真实专家标注

为什么必须做：

- 原计划的独特性很大一部分来自 PAL，而 PAL 如果没有真人反馈，就会沦为普通 active learning 原型。
- ISMIR 2026 明确强调 human-centric dimensions，这一点与你的项目高度契合，但前提是要有真实人的证据。

最小可行版本：

- 选 `100~300` 个 PAL 高不确定样本对。
- 邀请 `2~4` 位懂相关文化语境的标注者。
- 标注内容优先选：
  - 相似 / 不相似 pairwise judgment
  - cultural concept / affect concept
  - 简短 rationale

输出目标：

- 至少完成 1 轮真实 pairwise constraints 回灌。
- 报告回灌前后指标变化。

### 7.2 必做：补 baseline

为什么必须做：

- 目前仓库里有自己的阶段对比，但从 reviewer 视角看，还不够像标准论文对照。

建议至少补这几类：

1. `CultureMERT embedding + nearest neighbor / cosine retrieval`
2. `CultureMERT embedding + simple MLP / metric head`
3. `DCAS without OT`
4. `DCAS without domain adversarial`
5. `DCAS without constraints`

如果时间允许，再加：

- 一个非 CultureMERT 特征基线
- 一个简单 popularity / frequency baseline

### 7.3 必做：把数据短板讲清楚

当前 `research_dataset_v1` 很有用，但它还有两个不能回避的问题：

1. `affect_label` 缺失
2. `interactions` 仍偏弱监督

你有两种策略：

- 策略 A：补一小批 affect / concept 标注，把 `za` 的论证补强。
- 策略 B：弱化“情感识别准确率”叙事，把重点转到 pairwise alignment 与 recommendation behavior 上。

如果时间有限，我更推荐 `A + B` 的折中做法：

- 不追求大规模 affect 标签；
- 但至少补一小批真实 affect / concept / pairwise supervision；
- 论文里明确交代 interactions 的来源与局限。

### 7.4 必做：重新组织主实验

建议实验部分只保留 3 个主实验：

1. `主任务实验`
   比较推荐效果与 serendipity / calibration。

2. `消融实验`
   去掉 OT、去掉 domain、去掉 constraints、去掉 PAL 回灌。

3. `PAL 试点实验`
   展示真实专家反馈能否带来改善。

把风格迁移放到：

- 定性案例
- 或 supplementary audio demo

### 7.5 强烈建议：至少补一个公开 benchmark

为什么：

- 只用自建数据，reviewer 很容易问外部有效性。

最现实的做法：

- 从你原计划里的公开 benchmark 里选一个最容易接上的先跑。
- 不求全覆盖，但至少有一个公开可复现实验。

如果确实来不及：

- 论文必须主动把定位降成 `pilot / proof-of-concept / preliminary evidence`。

## 8. 建议的论文结构调整

### 8.1 最好保留的主贡献

建议正文只保留以下三条主贡献：

1. 一个跨文化音乐推荐的解纠缠对齐框架。
2. 一套面向 serendipity 与 cultural calibration 的评测协议。
3. 一个带真实专家约束回灌的 PAL 试点。

### 8.2 建议降级到次要贡献的内容

- 动态本体
- 风格迁移
- 波形级生成基线
- 前端与全栈系统

这些内容更适合在补充材料或 demo 里出现。

### 8.3 推荐的论文气质

更像这样：

- 问题清楚
- 方法聚焦
- 实验可复现
- 主张克制
- 强调跨文化与人本价值

而不是这样：

- 什么都想讲
- 每块都讲一点
- 但每块证据都不够厚

## 9. 从今天到投稿前的现实推进顺序

当前日期是 `2026-03-12`，距离 ISMIR 2026 全文截止 `2026-04-27 AoE` 不到七周。

因此最现实的推进顺序是：

### 第 1 周

- 确定论文最终题目方向。
- 确定只保留 3 个主贡献。
- 如果符合条件，立即申请 mentoring。
- 在 `research_dataset_v1` 上跑出正式 baseline。

### 第 2 周

- 组织第一轮真实专家标注。
- 固化 pairwise constraints 格式。
- 跑第一次真实 PAL 回灌训练。

### 第 3 周

- 补主 baseline 与 ablation。
- 对 `serendipity / calibration / MIG/DCI/SAP` 统一出表。

### 第 4 周

- 如果来得及，补 1 个公开 benchmark。
- 整理音频 demo / supplementary。

### 第 5 周

- 写完整论文初稿。
- 把所有 claim 对照证据逐条缩紧。

### 第 6 周

- 做匿名化检查。
- 写 ethics statement。
- 如果使用了 AI 工具辅助某些非核心工作，准备 AI usage statement。
- 整理 supplementary：匿名代码说明、音频样例、复现命令。

## 10. 最终判断

对照最初计划，这个项目现在最准确的定位是：

它已经完成了“一个可运行、可迭代、可继续做成论文的跨文化 MIR 原型系统”，但还没有完成“最初蓝图里那篇大而全的完整论文”。

如果目标改为 ISMIR 2026，我的明确建议是：

- 不要继续按“大一统系统论文”推进。
- 要主动收缩到一个更尖锐、更可证的故事。
- 把真实专家反馈和公开 benchmark 至少补一个起来。
- 把风格迁移、动态本体、全栈 demo 降到次要位置。
- 用克制主张换通过率，而不是用宏大叙事换风险。

一句话版本：

现在最该做的不是再扩功能，而是把已有主线压缩成一个 reviewer 能快速看懂、并且证据足够闭环的 ISMIR 论文。

## 11. 参考证据

仓库内部主要参考：

- `声音.txt`
- `README.md`
- `docs/PAPER_CLAIM_ALIGNMENT.md`
- `docs/ROUTE_A_PHASE3_RUNBOOK.md`
- `docs/technical_notes/point2_missing_parts_end.md`
- `NEXT_STEPS_ROADMAP.md`
- `reports/routeA_recommender_compare_phase4_cn.md`
- `reports/routeA_disentanglement_sharedfactors_compare_phase4_cn.md`
- `reports/eval_suite_phase4_v2/eval_suite_summary.json`
- `reports/research_dataset_v1_profile.md`
- `reports/research_dataset_v1_splits/split_report.md`

ISMIR 2026 官方参考：

- Call for Papers: `https://ismir2026.ismir.net/authors/call-for-papers`
- Author Guidelines: `https://ismir2026.ismir.net/authors/author-guidelines`
- Mentoring Program: `https://ismir2026.ismir.net/mentoring`
