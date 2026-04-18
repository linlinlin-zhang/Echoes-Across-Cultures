# 论文工作区索引与主题对齐（2026-04-15）

## 目的

这份索引用于解决两个实际问题：

1. 当前工作区文件很多，论文主线、辅助线、背景叙事、边界测试容易混在一起。
2. 你现在最需要的不是再扩写一个宏大故事，而是先把“这篇论文到底在研究什么、哪些结果能当主证据、哪些内容只能当边界或动机”整理清楚。

---

## 一句话主题

当前工作区最扎实、最一致、也最适合投稿 ISMIR 的论文主题不是“纯音乐相似性检索”，也不是“纯主动学习”，而是：

**一个面向跨文化音乐推荐的、backbone-agnostic 的下游框架：它希望在不同文化之间识别情感/功能/听感上相近的音乐，同时尽量避免把来源偏差、文化标签或数据集捷径误当成音乐相似性。**

换句话说：

- 终点任务是：**cross-cultural music recommendation**
- 关键科学问题是：**怎样在跨文化场景下识别“情感/风格/功能相近但文化来源不同”的曲目**
- 核心方法约束是：**不要让模型只学到 source/culture shortcut**
- PAL 的作用是：**给这个推荐系统补一个人类反馈修正回路**

---

## 任务分层

### 1. 主任务

**跨文化音乐推荐**

代表文件：

- `README.md`
- `paper/ismir2026_draft.tex`
- `docs/EXPERIMENT_INDEX.md`
- `dcas/cli/recommend.py`
- `dcas/cli/eval.py`

典型信号：

- 标题直接写的是“深度文化对齐音乐推荐”
- 推荐接口有 `target_culture`
- 评估指标不是单纯准确率，而是 `serendipity / cultural_calibration_kl / minority_exposure_at_k`

### 2. 核心方法层

**三因子表征 + 对抗去文化化 + OT 检索 + calibration-aware rerank**

代表文件：

- `dcas/models/dcas_vae.py`
- `dcas/ot/sinkhorn.py`
- `dcas/recommender.py`
- `configs/train/train_v4_main_culturemert_stage3.run.json`

这一层回答的是：

- 如何把 embedding 拆成更适合跨文化推荐的下游空间
- 如何让真正用于跨文化匹配的子空间少学到直接文化泄漏
- 如何在最终推荐时显式控制 target culture、minority exposure 和 novelty

### 3. PAL 子任务

**pairwise 相似性判断 / playlist-context 判断**

代表文件：

- `storage/pal/v4_main_annotation/README.md`
- `docs/research_dataset_v4/V4_REAL_PAL_WORKFLOW_2026-03-21_CN.md`
- `dcas/cli/pal_loop.py`
- `dcas/pal/uncertainty.py`

这一层不是论文的最终目标，而是：

- 为推荐系统提供高信息量的人类反馈
- 让专家判断“这两首歌是否适合放在同一个歌单或相近听歌场景里”
- 把这种判断转换成 pairwise constraints，回灌训练

### 4. 问题诊断层

**模型到底在学音乐相似，还是在学来源/文化捷径**

这很重要，但它更像主线研究中的“方法诊断问题”，而不是整篇论文唯一的题目。

它回答的是：

- 为什么跨文化推荐会失真
- 为什么我们需要 source confound audit
- 为什么不能只看 embedding 上的相似度

### 5. 背景叙事层

**去殖民化、认知流形、范式重构、文化对齐**

这类表述主要来自较宏观的研究愿景文本，例如：

- `archive/reference_sources/original_idea_notes/voice_plan_extracted.txt`
- `archive/reference_sources/original_idea_notes/声音.txt`

它们适合做动机和意义扩展，但不适合作为“论文主证据定义”。

---

## 最值得优先看的文件

### A. 论文定义层

1. `paper/ismir2026_draft.tex`
   - 最正式的投稿主叙事。
   - 已明确把问题定义成 culturally calibrated music recommendation。

2. `README.md`
   - 整个项目的工程与论文定位入口。
   - 已把系统拆成表征、跨文化对齐、PAL、评估四条主线。

3. `docs/PAPER_CLAIM_ALIGNMENT.md`
   - 控制什么能写、什么不能写。
   - 这是避免 claim-evidence mismatch 的最关键文件之一。

### B. 主证据层

1. `docs/EXPERIMENT_INDEX.md`
   - 当前实验矩阵总入口。
   - 能清楚区分主线 benchmark、sanity check 和 appendix 边界测试。

2. `reports/benchmarks/v4_main_culturemert_stage3_lambdamart/benchmark_summary.json`
   - 现在最强的 V4 主线证据之一。

3. `reports/benchmarks/v4_main_gemini_stage3_lambdamart/benchmark_summary.json`
   - backbone transfer 证据。

4. `reports/benchmarks/v4_routeA_small_culturemert_stage3_lambdamart/benchmark_summary.json`
   - public-source 风格的小型 sanity check。

5. `reports/benchmarks/public_routeA_phase2_cn_lambdamart/benchmark_summary.json`
   - public-source 辅助复核线。

### C. 风险与边界层

1. `reports/benchmarks/yambda_5b_subset_global_log_benchmark/benchmark_summary.json`
   - 对外部日志排序任务的边界测试。
   - 这不是主卖点，但一定要如实交代。

2. `reports/datasets/research_dataset_v4/main/dataset_profile.json`
   - 包含 `1122 tracks / 10 cultures / 8 sources`
   - 也明确记录了 `weighted_source_predictability_from_culture = 0.911765`

3. `docs/PROJECT_SELF_AUDIT_AND_V4_EXECUTION_PLAN_2026-03-20_CN.md`
   - 明确提醒“论文与真实实验状态不同步”是高优先级风险。

### D. PAL 层

1. `storage/pal/v4_main_annotation/README.md`
   - 很直接地定义了人类标注任务在判断什么。

2. `reports/routeA_phase3_pal_cn/phase3_pal_summary.json`
   - simulated PAL 中文线结果。

3. `reports/pal/v2_main_gemini_simulated/phase3_pal_summary.json`
   - simulated PAL 的较完整量化记录。

---

## 主证据、辅助证据、背景材料的建议分工

### 可作为论文主证据

- `paper/ismir2026_draft.tex`
- `docs/PAPER_CLAIM_ALIGNMENT.md`
- `docs/EXPERIMENT_INDEX.md`
- `reports/benchmarks/v4_main_culturemert_stage3_lambdamart/benchmark_summary.json`
- `reports/benchmarks/v4_main_gemini_stage3_lambdamart/benchmark_summary.json`
- `reports/benchmarks/v4_routeA_small_culturemert_stage3_lambdamart/benchmark_summary.json`
- `reports/datasets/research_dataset_v4/main/dataset_profile.json`

### 可作为辅助复核或补充材料

- `reports/benchmarks/public_routeA_phase2_cn_lambdamart/benchmark_summary.json`
- `reports/audits/ablation_v4_main_2026-04-05/...`
- `reports/pal/v2_main_gemini_simulated/phase3_pal_summary.json`
- `reports/routeA_phase3_pal_cn/phase3_pal_summary.json`

### 更适合做 appendix / limitation / discussion

- `reports/benchmarks/yambda_5b_subset_global_log_benchmark/benchmark_summary.json`
- `reports/datasets/research_dataset_v4/main/source_confound_report.json`

### 更适合做动机和愿景，不宜当主证据

- `archive/reference_sources/original_idea_notes/voice_plan_extracted.txt`
- `archive/reference_sources/original_idea_notes/声音.txt`
- 旧版中文大段草稿中带宏大哲学叙事但未对应到现有实验的段落

---

## 现在最容易写偏的地方

### 1. 把论文写成“纯音乐相似性论文”

这会丢掉你现在最强的一层：

- 推荐目标是明确的
- OT rerank 和 calibration-aware rerank 的贡献是可量化的
- `serendipity / calibration / minority exposure` 的三指标结构比“相似性故事”更完整

### 2. 把 PAL 写成已经完成的人类研究

当前更稳妥的写法应该是：

- PAL pipeline implemented
- PAL-ready
- 有 simulated PAL 和 real workflow
- 但 publication-strength 的 full human study 还没有完全闭环

### 3. 把 V4 写成已经消除了 confound 的完美数据集

不能这么写。

当前更合理的写法是：

- V4 已经是一个更清晰的数据契约与评测主线
- 但 source confound 仍然显著
- 这恰恰说明论文价值更像“一个可靠的下游设计模式 + 风险透明的实验框架”

### 4. 把外部日志 benchmark 写成主战场

`Yambda-5B subset` 更适合被定位成：

- 外部边界测试
- 说明 DCAS 的优势并不是在所有通用日志排序任务上都成立
- 这反而帮助你把论文问题定义收紧到“cross-cultural recommendation”

---

## 建议的论文口径

如果要一句话稳稳概括当前论文，可以优先用下面这个版本：

**我们研究的是：在跨文化音乐推荐场景中，如何基于冻结音乐 embedding 构建一个可控的下游推荐框架，使系统更容易捕捉跨文化共享的情感/功能相关性，并减少把文化来源与数据集捷径误当作音乐相似性的风险。**

如果要再短一点，可以用：

**这是一篇跨文化音乐推荐论文，pairwise 相似性判断与 PAL 只是为推荐系统服务的监督与修正机制。**

---

## 后续写作顺序建议

1. 先以 `paper/ismir2026_draft.tex` 为主稿骨架。
2. 用 `docs/PAPER_CLAIM_ALIGNMENT.md` 约束主张边界。
3. 用 `docs/EXPERIMENT_INDEX.md` 和 V4 benchmark summaries 决定主实验矩阵。
4. 用 `reports/datasets/research_dataset_v4/main/dataset_profile.json` 写 Data / Limitations。
5. 把 PAL 单列成“workflow / PAL-ready / pilot-ready”，不要写成 fully validated human study。
6. 把 `Yambda-5B subset` 放到 appendix 或 limitation-style supplementary narrative。

---

## 最终判断

基于当前整个工作区，而不是局部草稿，这篇论文真正的主题已经相当清楚：

**主线是跨文化音乐推荐；“模型是否把来源捷径误当音乐相似性”是你们在这个主线中主动识别并处理的核心方法问题；PAL 与 pairwise similarity judgment 是服务于这个主线的反馈机制。**
