# `build_research_dataset_v4.py` 代码说明

对应源码：
[E:/Desktop/Echo/dcas/scripts/build_research_dataset_v4.py](E:/Desktop/Echo/dcas/scripts/build_research_dataset_v4.py)

## 1. 这个文件在整个项目里负责什么

这个文件是 `V4` 数据集的总调度入口。

如果说：

- `build_research_dataset_v3.py` 更偏“把多来源音频真正收进来”
- 那么 `build_research_dataset_v4.py` 更偏“把已有数据资产标准化、审计化、可复用化”

它的核心任务不是重新写一大堆数据源采集逻辑，而是按照 manifest 配置，把已有来源串成一条稳定的数据流水线：

- `merge`
- `harmonize`
- `interactions`
- `audit`
- `embeddings`

也就是说，这个文件负责把“分散的数据资产”变成“标准化研究数据底座”。

## 2. 它和 V3 的关系是什么

当前主线 `V4 main` 的 manifest 指向的是多份 `V3` 域级 `metadata.csv`。因此它的工作重点不是从互联网下载新数据，而是：

- 读取多份来源 metadata
- 合并成一张大表
- 统一 schema
- 生成统一标签字段
- 合成弱交互
- 产出 embedding 版本的 `tracks.npz`
- 生成数据审计报告和数据卡

所以你可以把它理解成：

- `V3` 更像“原始语料构建”
- `V4` 更像“研究资产工程化封装”

## 3. 文件最上方的导入在说明什么

它导入的模块本身就揭示了整个流水线：

- `align_assets_to_tracks`
  - 当 embedding 构建失败一部分样本时，用来把 metadata 和 interactions 重新裁对齐。

- `audit_manifest`
  - 先检查 manifest 是否写得合理。

- `audit_dataset_v4`
  - 做 metadata 级别的数据审计。

- `build_tracks_from_audio`
  - 用 CultureMERT 从音频生成 `tracks.npz`。

- `build_tracks_with_gemini`
  - 用 Gemini 生成 `tracks.npz`。

- `harmonize_v4_metadata`
  - 做 schema 归一化。

- `merge_metadata_dedup`
  - 合并多份 metadata 并按 `track_id` 去重。

- `synthesize_interactions`
  - 从 metadata 合成弱交互。

- `validate_dataset`
  - 对最终 `tracks + interactions` 进行验证。

这说明它本身不是做某一个步骤，而是一个 orchestrator。

## 4. 顶层常量是什么意思

### `REPO_ROOT`

作用：

- 获得仓库根目录。

意义：

- manifest 里可能出现相对路径，因此需要统一相对于仓库根目录解析。

### `DEFAULT_STAGES`

内容：

- `["merge", "harmonize", "interactions", "audit"]`

意义：

- 默认情况下不会自动跑 embedding。
- 这是一个很合理的设计，因为 embedding 往往最耗时、最依赖环境，也最容易失败。

## 5. 辅助函数逐个说明

### `_load_json(path)`

作用：

- 读取 JSON 配置文件。

特点：

- 使用 `utf-8-sig` 读取，兼容 BOM。

### `_resolve_path(path_like)`

作用：

- 把 manifest 中的路径解析成绝对路径。

逻辑：

- 如果原本就是绝对路径，直接使用。
- 如果是相对路径，则相对于仓库根目录解析。

意义：

- 保证 manifest 能写相对路径，也能写绝对路径。

### `_write_json(path, payload)`

作用：

- 统一 JSON 写出逻辑。

特点：

- 自动建目录
- `ensure_ascii=False`
- `indent=2`

### `_count_csv_rows(path)`

作用：

- 统计一个 CSV 的数据行数，不算表头。

意义：

- embedding 阶段会用它判断：
  - 生成出来的 `tracks.npz` 是否比 metadata 行数更少
  - 如果更少，就说明有样本在 embedding 阶段丢失，需要重新对齐资产

### `_metadata_thresholds(manifest)`

作用：

- 从 manifest 的 `validation` 段里读取 metadata 审计阈值，并组装成 `MetadataAuditThresholds`。

包含的阈值有：

- `min_tracks_per_culture`
- `max_culture_imbalance_ratio`
- `min_interactions_per_user`
- `max_unknown_track_ratio`
- `max_duplicate_user_track_ratio`

意义：

- 把“数据质量标准”外置到配置中，而不是硬编码。

### `_track_validation_thresholds(manifest)`

作用：

- 读取最终 `tracks + interactions` 验证所需阈值。

和上一函数的区别：

- 它额外加入了 `max_zero_norm_ratio`，因为 embedding 验证需要检查零向量。

### `_interaction_outputs(out_root)`

作用：

- 约定交互文件名。

固定返回：

- `single -> interactions_synth_single.csv`
- `mixed -> interactions_synth_mixed.csv`

### `_enabled_interactions(manifest, out_root)`

作用：

- 查看 manifest 中哪些交互协议被启用，并且对应文件是否已经存在。

意义：

- `audit_dataset_v4` 需要知道应该把哪些交互文件一并纳入审计。

### `_embedding_track_name(name, cfg)`

作用：

- 根据 embedding 类型和配置生成输出文件名。

例子：

- `culturemert + window_count=3`
  - `tracks_culturemert_mw3.npz`

- `gemini-embedding-2-preview + window_count=3`
  - `tracks_gemini_embedding2_mw3.npz`

意义：

- 文件名里显式编码了 backbone 和多窗口设定，方便管理多种版本。

### `_build_culturemert(metadata_csv, out_root, cfg)`

作用：

- 调用 `build_tracks_from_audio(...)` 构建 CultureMERT 版本的 `tracks.npz`。

关键参数从 `cfg` 中读取：

- `model_id`
- `device`
- `pooling`
- `layer_indices`
- `layer_weights`
- `max_seconds`
- `window_count`
- `window_strategy`
- `window_aggregate`
- `limit`
- `skip_errors`

意义：

- 这是一个“薄封装函数”，作用是把 manifest 配置翻译成底层 embedding 构建器能接受的参数。

### `_build_gemini(metadata_csv, out_root, cfg)`

作用：

- 调用 `build_tracks_with_gemini(...)` 构建 Gemini 版本的 `tracks.npz`。

关键参数：

- `model_id`
- `api_key`
- `api_key_file`
- `vertexai`
- `vertex_project`
- `vertex_location`
- `output_dimensionality`
- `task_type`
- `max_seconds`
- `target_sample_rate`
- `window_count`
- `window_strategy`
- `window_aggregate`
- `limit`
- `skip_errors`
- `cache_dir`
- `dry_run`
- `max_workers`

意义：

- 和 `_build_culturemert` 一样，本质上是配置适配层。

## 6. 核心函数 `build_research_dataset_v4(...)` 详解

这是整个文件的主入口。

函数签名：

- `manifest_path`
- `stages=None`
- `embedding_targets=None`
- `allow_manifest_errors=False`

### 参数逐个解释

#### `manifest_path`

作用：

- 指定 manifest 配置文件位置。

意义：

- manifest 决定：
  - 数据集名称
  - 输出目录
  - 来源 metadata
  - 交互协议
  - embedding 方案
  - 验证阈值

#### `stages`

作用：

- 指定本次实际要跑哪些阶段。

可能值：

- `merge`
- `harmonize`
- `interactions`
- `audit`
- `embeddings`

意义：

- 允许你只跑部分流水线。
- 例如：
  - 只想重新生成 metadata 时，不必重跑 embedding
  - 只想跑某个 embedding backbone 时，可以跳过前面的阶段

#### `embedding_targets`

作用：

- 当包含 `embeddings` 阶段时，只跑指定的 embedding 目标。

例子：

- 只跑 `culturemert`
- 只跑 `gemini`

意义：

- 节省时间和资源。

#### `allow_manifest_errors`

作用：

- 是否允许 manifest 审计里存在错误也继续执行。

默认：

- `False`

意义：

- 正常情况下，manifest 出错应直接阻断。
- 这个开关更像调试或抢修模式。

## 7. `build_research_dataset_v4(...)` 内部流程

### 7.1 读取 manifest 与准备目录

它会先：

- 解析 manifest 路径
- 读取 JSON
- 解析输出根目录
- 构造 `reports_root`

其中：

- `out_root`
  - 存数据资产

- `reports_root`
  - 存审计、验证、构建报告

### 7.2 先做 manifest 审计

这里会调用 `audit_manifest(...)`：

- 写出 `reports/.../manifest_audit/summary.json`
- 写出 `summary.md`

如果存在 `severity = error` 且 `allow_manifest_errors=False`，就直接抛异常。

这个设计的意义是：

- 先保证“配置正确”
- 再开始跑昂贵的数据构建流程

### 7.3 保存 manifest 快照

它会把当前 manifest 写成：

- `manifest.snapshot.json`

意义：

- 即便后续你修改了配置文件，当前这次构建对应的配置快照仍然被保留下来，方便复现。

### 7.4 `merge` 阶段

如果 `merge` 被启用：

- 从 manifest 的 `sources` 段提取所有 `local_metadata`
- 调用 `merge_metadata_dedup(...)`
- 输出：
  - `metadata_raw.csv`

这一阶段的目标是：

- 先把来源表拼在一起
- 并按 `track_id` 去重

### 7.5 `harmonize` 阶段

如果 `harmonize` 被启用：

- 调用 `harmonize_v4_metadata(...)`
- 输出：
  - `metadata_clean.csv`
  - `metadata_harmonized.csv`
  - `metadata_release.csv`

这里的语义区别是：

- `metadata_raw`
  - 合并后的原始表

- `metadata_clean`
  - 字段清洗和补齐后的表

- `metadata_harmonized`
  - 加入统一标签体系后的表

- `metadata_release`
  - 对外作为“最终主表”使用的版本

### 7.6 `interactions` 阶段

如果 `interactions` 被启用：

- 读取 manifest 里的 `interaction_protocol`
- 对 `single`、`mixed` 两类协议分别判断是否启用
- 启用就调用 `synthesize_interactions(...)`

会产出：

- `interactions_synth_single.csv`
- `interactions_synth_mixed.csv`

关键参数来自 manifest：

- `users_per_culture`
- `tracks_per_user`
- `min_weight`
- `max_weight`
- `genre_column`
- `mode`
- `secondary_cultures`
- `home_share`
- `seed`

这一阶段的意义：

- 当前项目没有原生用户日志，因此用合成交互为推荐任务提供监督信号。

### 7.7 `audit` 阶段

如果 `audit` 被启用：

- 调用 `audit_dataset_v4(...)`
- 读取：
  - `metadata_release.csv`
  - 当前启用且存在的交互文件

输出：

- `dataset_profile.json`
- `dataset_profile.md`
- `validation_report.json`
- `data_card.json`

这里每个文件的作用不同：

- `dataset_profile`
  - 更偏细粒度审计

- `validation_report`
  - 更偏摘要

- `data_card`
  - 更偏论文/共享/归档层面的数据说明

### 7.8 `embeddings` 阶段

如果 `embeddings` 被启用：

它会遍历 manifest 中定义的 embedding 目标，比如：

- `culturemert`
- `gemini`

对每个启用目标做：

1. 构建 `tracks.npz`
2. 检查生成的 tracks 数量是否少于 metadata 行数
3. 若数量变少，则调用 `align_assets_to_tracks(...)`
4. 调用 `validate_dataset(...)`
5. 写出每个 backbone 的验证报告

这一段是全文件最重要的“工程闭环”部分。

## 8. 辅助函数流程说明
这个文件的辅助函数数量也不少，但它们的职责分层非常清晰，可以按下面这条链来看：

1. 配置与路径层
   - `_load_json(...)`、`_resolve_path(...)`、`_write_json(...)`
   - 负责读 manifest、解析路径、保存阶段报告
2. 轻量统计与阈值层
   - `_count_csv_rows(...)`、`_metadata_thresholds(...)`、`_track_validation_thresholds(...)`
   - 负责给后续审计和验证准备参数与规模信息
3. 交互与输出命名层
   - `_interaction_outputs(...)`、`_enabled_interactions(...)`、`_embedding_track_name(...)`
   - 负责推断本次构建应产出哪些 interactions 和 tracks 文件
4. 表征构建适配层
   - `_build_culturemert(...)`、`_build_gemini(...)`
   - 负责把不同 backbone 的 embedding 构建入口封装成统一调用方式

所以 `build_research_dataset_v4.py` 的辅助函数，本质上是在帮主流程解决三件事：

- manifest 怎么变成可执行配置
- 每个阶段的输入输出怎么命名与组织
- 不同 embedding backbone 怎么以统一接口接入流水线

## 9. 为什么 embedding 后还要重新对齐 metadata 和 interactions

这是这个文件最专业的一个设计点。

在真实环境里，embedding 构建很可能失败一部分样本，原因包括：

- 音频损坏
- 解码失败
- API 失败
- 模型报错

这时如果你直接保留原始 metadata 和 interactions，下游会出现严重错位：

- metadata 里有 1122 条
- tracks.npz 里只有 1100 条
- interactions 还在引用那 22 条失败样本

所以脚本通过：

- `_count_csv_rows(...)`
- `align_assets_to_tracks(...)`

自动把：

- metadata
- interactions

裁到和 `tracks.npz` 一致的样本集合上。

这意味着：

- `tracks` 是最终真值集合
- 不是 `metadata_release.csv`

这是一个非常值得在面试中强调的工程意识。

## 10. `main()` 在做什么

`main()` 提供命令行入口，暴露的参数有：

- `--manifest`
  - 必填，指定 manifest 文件。

- `--stages`
  - 可选，指定要执行的阶段。

- `--embedding_targets`
  - 可选，限制 embedding 目标。

- `--allow_manifest_errors`
  - 是否允许 manifest 存在错误继续执行。

最终它会打印一个简要结果：

- `reports_root`
- 执行过的步骤列表

## 11. 这个文件最核心的设计思想

### 10.1 manifest 驱动，而不是把细节写死

数据集版本、来源、交互协议、embedding 方案、阈值标准都放在 manifest 里。

好处：

- 可复现
- 可切换
- 可比较不同版本

### 10.2 阶段化执行

不是每次都从头跑到底，而是按 stage 组织流程。

好处：

- 调试方便
- 成本可控
- 长流程更稳

### 10.3 审计和构建同等重要

这个文件不是“把数据做出来就结束”，而是明确把：

- manifest 审计
- metadata 审计
- 最终 dataset 验证

都写进主流程。

### 10.4 资产之间必须对齐

它把：

- metadata
- interactions
- tracks

看成三套不同资产，并且显式处理它们的对齐问题。

### 10.5 支持多 backbone 并存

同一份 `metadata_release.csv` 可以衍生出：

- CultureMERT 版 tracks
- Gemini 版 tracks

这说明项目的数据底座和表征底座是解耦的。

## 12. 读这个文件时最推荐的顺序

1. 先看 `build_research_dataset_v4(...)`
2. 再看 `_build_culturemert` 和 `_build_gemini`
3. 然后看 `_metadata_thresholds`、`_track_validation_thresholds`
4. 最后补看 `_enabled_interactions`、`_embedding_track_name` 这些辅助函数

这样你会先抓住主线，再理解辅助细节。

## 13. 面试或讲解时最值得强调的点

- 这是一个“数据流水线 orchestrator”，不是单功能脚本。
- 它把数据构建拆成多个可复用阶段。
- 它用 manifest 把数据版本、来源、审计标准、embedding 方案都配置化了。
- 它很重视数据资产的一致性与可复现性。
- 它不是只输出一个 CSV，而是输出一整套：
  - metadata
  - interactions
  - tracks
  - audit reports
  - validation report
  - data card

## 14. 你可以怎样用一句话概括这个文件

可以这样说：

“`build_research_dataset_v4.py` 负责把多来源音乐元数据按 manifest 配置统一合并、标准化、合成交互、构建多种 embedding 版本，并生成完整的数据审计与验证产物，是整个 V4 数据工程主线的总调度器。”
