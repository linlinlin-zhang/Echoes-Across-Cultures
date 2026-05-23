# 项目总脉络图与主线说明

日期：2026-04-18

## 1. 一句话先说清楚

这个项目当前最应该被理解为：

**一个面向跨文化音乐推荐的下游框架研究项目。**

它不是单纯在做“音乐相似度检索”，也不是单纯在做“全栈音乐产品”，而是在研究：

- 当用户想听到“文化上不同、但情绪或功能上仍然合适”的音乐时，
- 我们能不能在强大的冻结音频 embedding 之上，
- 通过解耦式下游表示、OT 检索、calibration-aware reranking，以及少量 PAL 人类反馈回灌，
- 让推荐结果同时兼顾 serendipity、cultural calibration 和 minority exposure。

因此，当前论文最稳的主题不是“一个复杂的大系统做了很多事”，而是：

**在冻结音乐 foundation embeddings 之上，构建一个 backbone-agnostic 的跨文化推荐下游设计。**

## 2. 为什么现在会显得很乱

项目之所以显得乱，不是因为没有主线，而是因为有多层历史和用途叠在一起：

1. `V2 / V3 / V4` 三代数据与实验资产同时存在。
2. 研究主线、PAL 标注线、原型系统线、旧前端残留线混在一个仓库里。
3. 很多旧文档没有退场，所以“曾经考虑过的方向”和“现在真正写论文的方向”同时可见。
4. 有些目录是研究证据链的一部分，有些只是演示、采集、工作痕迹，但它们目前都在顶层共存。

所以真正的问题不是“项目没有方向”，而是：

**证据等级没有被显式分层。**

## 3. 当前最重要的四层结构

### A. 研究主线

这一层是论文主证据链，最重要。

核心目录：

- `dcas/`
- `configs/`
- `storage/public/research_dataset_v4/`
- `storage/models/`
- `reports/`
- `docs/research_dataset_v4/`
- `docs/manuscript/`
- `paper/`

这一层负责：

- 数据构建与 harmonization
- embedding 载入与下游训练
- benchmark 与 reranking sweep
- PAL 回灌后的正式对比
- 论文叙事与结果落地

### B. PAL 采集与回灌线

这一层是研究主线的重要实验分支，但不是另一篇独立论文。

核心目录：

- `storage/pal/v4_main_annotation/`
- `configs/pal/`
- `dcas/scripts/run_pal_platform.py`
- `dcas/scripts/run_phase3_pal.py`
- `docs/research_dataset_v4/V4_REAL_PAL_WORKFLOW_2026-03-21_CN.md`
- `docs/research_dataset_v4/V4_PAL_MIGRATION_AND_ALIGNMENT_2026-04-18_CN.md`

这一层负责：

- 生成 PAL 标注包
- 收集真人 pairwise judgments
- 将真人反馈迁移回当前 V4 主线
- 做 warm-start 或轻量增量训练
- 检查 PAL 是否真的改善 benchmark operating point

### C. 原型系统与展示线

这一层增强项目的可交互性与实用性，但它不是论文的核心证据。

核心目录：

- `dcas_server/`
- `web_prototype/`

这一层负责：

- 把研究系统做成可运行 API
- 支持上传音频、分析、推荐、反馈
- 作为展示、试用、潜在数据采集入口

### D. 历史资产与残留线

这一层不是没用，但不应该再和论文主线混为一谈。

代表目录：

- `docs/research_dataset_v2/`
- `docs/research_dataset_v3/`
- `storage/public/research_dataset_v3/`
- `configs/train/train_v3_*`
- `configs/benchmark/recommender_benchmark_v3_*`
- 已移除的 legacy `web/` 前端（当前仅存在于 git 历史中）
- `workspace_assets/`、`archive/reference_sources/`、`tmp/` 一类工作材料或参考文件

这一层主要价值是：

- 历史对照
- 方法演化记录
- appendix 或补充材料
- 工作过程留痕

## 4. 当前真正的论文主线到底是哪一条

当前最应该默认引用的正式主线是：

**V4 main + frozen backbone + stage3 + calibrated benchmark + PAL-ready / PAL-aligned feedback loop**

更具体地说：

1. 数据主线是 `V4 main`
2. 小规模 sanity-check 是 `V4 routeA_small`
3. 主 backbone 是 `CultureMERT mw3`
4. 迁移验证 backbone 是 `Gemini Embedding 2 mw3`
5. 正式评测不是裸检索，而是带 calibrated reranking 的 benchmark line
6. PAL 不是主方法本体，而是主线上的反馈修正分支

这意味着：

- `V3` 现在更适合作为历史前身，不应再当当前主证据。
- `routeA` 是 sanity-check 或公开来源小线，不应和 `V4 main` 同级叙述。
- `Yambda` 更像外部边界测试，适合 appendix 或 limitation supporting evidence。
- `web_prototype` 和 `dcas_server` 是系统化表达，不是主要 scientific claim。

## 5. 研究主线的执行链条

如果按“数据 -> 训练 -> 评测 -> PAL -> 论文”的顺序看，现在最清楚的链条是下面这条。

### 5.1 数据构建

主入口：

- `dcas/scripts/build_research_dataset_v4.py`

主配置：

- `configs/dataset/research_dataset_v4_main_from_v3.json`
- `configs/dataset/research_dataset_v4_routeA_small.json`

主输出：

- `storage/public/research_dataset_v4/main/`
- `storage/public/research_dataset_v4/routeA_small/`

### 5.2 训练

主入口：

- `dcas/scripts/run_train_from_json.py`

核心配置：

- `configs/train/train_v4_main_culturemert_stage3.run.json`
- `configs/train/train_v4_main_gemini_stage3.run.json`

关键产物：

- `storage/models/dcas_full_v4_main_culturemert_stage3.pt`
- `storage/models/dcas_full_v4_main_gemini_stage3.pt`

### 5.3 Benchmark 与 reranking

主入口：

- `dcas/scripts/run_recommender_benchmarks.py`

核心配置：

- `configs/benchmark/recommender_benchmark_v4_main_culturemert_stage3_lambdamart.run.json`
- `configs/benchmark/recommender_benchmark_v4_main_gemini_stage3_lambdamart.run.json`
- `configs/benchmark/recommender_benchmark_v4_main_culturemert_real_pal_stage3_calibration_sweep.run.json`

关键结果目录：

- `reports/benchmarks/v4_main_culturemert_stage3_lambdamart/`
- `reports/benchmarks/v4_main_gemini_stage3_lambdamart/`
- `reports/hparam/v4_main_culturemert_real_pal_stage3_calibration_sweep/`

### 5.4 PAL 真实反馈回灌

主配置：

- `configs/pal/pal_v4_main_culturemert_prepare.run.json`
- `configs/pal/pal_v4_main_culturemert_real.run.json`
- `configs/pal/pal_v4_main_culturemert_real_from_v4_main_annotation_stage3.run.json`

关键说明：

- `storage/pal/v4_main_annotation/` 里的真人 PAL，是当前真实反馈的重要来源。
- 但它最初来自标注工作流，并不天然和当前 benchmark evaluator 完全同构。
- 所以需要一层“迁移与对齐”，也就是把 PAL 结果接回现在的 `V4 main + benchmark` 主线。

### 5.5 论文表达

最关键的论文文件：

- `paper/ismir2026_draft.tex`

它当前的标题已经非常明确地说明主线是：

**A Backbone-Agnostic Framework for Culturally Calibrated Music Recommendation with OT Reranking and Participatory Feedback**

这说明论文主张并不是“做了一个音乐网站”，而是：

**在多种 backbone 之上都可工作的跨文化推荐下游框架。**

## 6. 现在最该优先读的文件

如果只想快速把项目读明白，建议按下面顺序：

1. `docs/workspace_index/README_2026-03-22_CN.md`
2. `docs/workspace_index/LATEST_V4_FULL_WORKFLOW_2026-03-22_CN.md`
3. `docs/EXPERIMENT_INDEX.md`
4. `docs/research_dataset_v4/README.md`
5. `docs/manuscript/PAPER_STORYLINE_AND_CLAIM_MAP_2026-03-21_CN.md`
6. `docs/workspace_index/PAPER_RESULTS_AND_ISMIR_ASSESSMENT_2026-04-15_CN.md`
7. `paper/ismir2026_draft.tex`
8. `docs/research_dataset_v4/V4_PAL_MIGRATION_AND_ALIGNMENT_2026-04-18_CN.md`

这 8 份文件分别回答：

- 现在推荐怎么看工作区
- V4 主流程是什么
- 实验配置和结果怎么索引
- V4 数据的当前状态
- 论文到底想讲什么
- 当前结果与 ISMIR 竞争力怎么评估
- 论文标题与 abstract 到底在主张什么
- 真人 PAL 如何接回当前主线

## 7. 前端为什么会制造额外混乱

当前仓库里现在主要有两种现役前端/交互入口，以及一种历史遗留信号：

1. `web_prototype/`
2. `storage/pal/v4_main_annotation/pal_annotation.html`
3. 已移除的 legacy `web/`

但它们并不是同一件事。

### `web_prototype/`

这是当前最像正式原型前端的一套静态站点资源，直接对接：

- `/api/prototype/bootstrap`
- `/api/prototype/upload`
- `/api/prototype/analyze`
- `/api/prototype/register`
- `/api/prototype/feedback`

它是展示线与交互线。

### `storage/pal/v4_main_annotation/pal_annotation.html`

这是 PAL 标注界面，不是通用产品前端。

它服务的是：

- pairwise judgment
- 标注进度追踪
- 结果导出

它属于 PAL 采集线。

### 已移除的 legacy `web/`

当前文件系统里已经没有 `web/` 目录了；它只存在于 git 历史里，说明旧前端源码已经被移除。

这说明：

- 旧前端已经不再是现役代码路径
- 当前真正可见、可读、可维护的前端资产在 `web_prototype/`

这正是“项目看起来乱”的一个非常具体的原因：

**旧前端入口信号没有被清理，而新原型前端已经转移到另一个目录。**

## 8. 当前哪些内容是论文主结果，哪些不是

### 属于论文主结果的

- `V4 main` 数据集
- CultureMERT / Gemini 两条 backbone 的 V4 benchmark
- stage3 训练协议
- OT + calibrated reranking operating points
- calibration sweep
- PAL 迁移与对齐后的结果比较

### 不属于论文主结果本体，但可以增强说服力的

- `web_prototype/` 的可交互系统形态
- `dcas_server/` 的 API 化封装
- 真人 PAL 标注包与平台流程
- routeA 小线 sanity-check

### 更适合做历史背景、对照或 appendix 的

- V2 / V3 全套历史实验
- Yambda log benchmark
- 根目录临时导出文件
- 已移除 legacy `web/` 的历史痕迹

## 9. 如果以后只想沿主线推进，应该怎么做

后续所有讨论，建议默认遵守下面这套约定：

1. 默认主数据集是 `storage/public/research_dataset_v4/main/`
2. 默认主 benchmark 先看 `v4_main_*_stage3_lambdamart`
3. 默认论文主张以 `paper/ismir2026_draft.tex` 的标题与 abstract 为准
4. 默认 PAL 要先问“是否已经对齐到 V4 main benchmark evaluator”
5. 默认把 `web_prototype` 当展示线，而不是论文证据主线
6. 默认把已移除的 legacy `web/` 当历史背景，而不是当前工作区目录

## 10. 当前最准确的总体判断

这个项目现在不是“没有主题”，而是“主题已经收敛，但历史层还没有被彻底压扁”。

如果必须用一句更学术但也更准确的话来概括它，可以写成：

**本项目的核心，是在统一的 V4 数据契约上，研究一个对 backbone 不敏感、对跨文化校准可控、并可接入少量真实人类反馈的音乐推荐下游框架。**

这句话基本可以同时解释：

- 为什么有 V4 主线
- 为什么有 calibrated rerank
- 为什么有 PAL
- 为什么有 prototype
- 为什么旧材料仍然多，但不该再混作同一级主张
