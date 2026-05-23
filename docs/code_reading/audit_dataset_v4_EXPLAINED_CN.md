# `audit_dataset_v4.py` 代码说明

对应源码：[E:/Desktop/Echo/dcas/scripts/audit_dataset_v4.py](E:/Desktop/Echo/dcas/scripts/audit_dataset_v4.py)

## 1. 这个文件在数据链路里负责什么
这个脚本负责做 V4 风格数据集的“元数据审计”。

它不是训练前验证 `tracks.npz` 的脚本，那是 `validate_dataset.py` 更偏重的事情。它更像是数据工程和数据治理层的体检工具，关注的问题包括：

- metadata 字段齐不齐
- 缺失率高不高
- 各文化域是否失衡
- 音频路径是否存在
- 来源与文化域是否高度绑定
- 交互文件是否有 unknown track 或重复 user-track

所以它的角色可以概括为：

“站在 metadata 和数据治理视角，对整套数据资产做一次系统体检。”

## 2. 顶部常量与阈值类的意义
### `REQUIRED_FIELDS`
V4 元数据中最重要的必需字段集合。

### `RECOMMENDED_FIELDS`
建议尽量覆盖的分析型字段。

### `GOVERNANCE_FIELDS`
用于版本、QC、embedding 状态等治理信息的字段。

### `MetadataAuditThresholds`
这个 dataclass 封装了审计阈值，包括：

- `min_tracks_per_culture`
- `max_culture_imbalance_ratio`
- `min_interactions_per_user`
- `max_unknown_track_ratio`
- `max_duplicate_user_track_ratio`

它的意义在于把“什么叫有问题”显式配置化，而不是硬编码散在逻辑里。

## 3. 辅助函数流程说明
这个文件的辅助函数很多，建议你按“从基础统计到高级审计”的顺序来理解：

1. `_read_csv(...)`
   - 先把 metadata 或 interactions 读进来
2. `_coverage(...)`、`_safe_float(...)`、`_quantiles(...)`
   - 提供最基础的覆盖率、数值安全转换、分位数统计能力
3. `_safe_rel(...)`、`_normalized_entropy(...)`
   - 分别服务于路径输出和来源分布复杂度分析
4. `_field_coverage_report(...)`
   - 在字段层面生成 required / recommended / governance 覆盖率报告
5. `_metadata_report(...)`
   - 基于上面的基础工具，对 metadata 做完整审计
6. `_interactions_report(...)`
   - 对 interactions 做补充审计
7. `_to_markdown(...)`
   - 把最终 JSON 审计结果转成可读 Markdown

也就是说，这个文件的辅助函数不是零散小工具，而是分层托起了整个审计过程：

- 最底层做统计
- 中间层做字段和分布分析
- 最上层输出结构化报告

## 4. 辅助函数逐个说明
### `_read_csv(path)`
- 读取 CSV 与字段名

### `_coverage(rows, field)`
作用：

- 计算某个字段的非空覆盖率

它会把以下值视为空：

- 空字符串
- `nan`
- `none`
- `null`

### `_safe_float(value)`
- 安全地把值转成有限浮点数

### `_quantiles(values)`
作用：

- 返回 `min / p25 / p50 / p75 / max / mean`

用于：

- 时长分布
- 每用户交互统计等

### `_safe_rel(path)`
作用：

- 尽量返回绝对路径字符串
- 如果解析失败则退回原路径字符串

### `_normalized_entropy(counter)`
作用：

- 计算归一化熵

在这里主要用于衡量：

- 某个文化域背后来源是否单一

如果一个文化域只来自一个 source，熵就会很低。

### `_field_coverage_report(rows, fieldnames)`
作用：

- 对 required / recommended / governance 三组字段分别做：
  - 是否存在
  - 覆盖率统计

同时返回缺失的必需字段列表。

## 5. `_metadata_report(...)` 在做什么
这是整个脚本的核心函数之一。

它负责对 metadata 本身做全方位检查。

### 4.1 重复检查
会统计：

- `duplicate_track_ids`
- `duplicate_audio_paths`

如果 `track_id` 重复，会直接记为 error。

### 4.2 基础分布统计
会统计：

- `culture_distribution`
- `source_distribution`
- `duration_sec_stats`
- `sample_rate_distribution`
- `channel_distribution`

这让你能从宏观上看到数据集长什么样。

### 4.3 音频文件存在性检查
对每一行：

- 解析 `audio_path`
- 如果是相对路径，按 metadata 所在目录转绝对路径
- 检查文件是否存在

不存在就计入 `missing_audio`

### 4.4 文化不平衡检查
它会算：

- `culture_imbalance_ratio = max_count / min_count`

如果超过阈值，就发出 warn。

此外每个文化域若低于 `min_tracks_per_culture`，也会单独告警。

### 4.5 `source_confound` 检查
这是这个脚本最值得你重点理解的部分。

它会统计：

- 每个文化域由哪些 `source_dataset` 支撑
- 每个文化域最主要来源占比
- 每个文化域的来源熵
- 每个来源最主要对应哪个文化域

然后给出两个非常关键的指标：

- `weighted_source_predictability_from_culture`
- `weighted_culture_predictability_from_source`

其中第一个可以理解成：

- 如果知道文化域，能多大程度上猜到它来自哪个 source

如果这个值很高，说明文化标签和来源高度绑定，存在明显的 `source confound` 风险。

### 4.6 字段覆盖率检查
通过 `_field_coverage_report(...)`，它会：

- 报告哪些必需字段缺失
- 报告必需字段覆盖率是否小于 1.0

这使得 schema 完整性检查非常透明。

## 6. `_interactions_report(...)` 在做什么
这是另一个核心函数，专门审计交互文件。

它会检查：

- 是否有 `user_id / track_id`
- 是否为空文件
- 有多少 unknown track
- 有多少重复 `(user_id, track_id)` 对
- 每个用户交互数分布
- 按文化域暴露分布

并根据阈值发出 warn 或 error。

这说明这个脚本虽然以 metadata 审计为主，但它并没有完全忽略任务监督层。

## 7. `_to_markdown(report)` 的作用
这个函数把整个 JSON 报告转成可读的 Markdown 摘要，包括：

- 数据集规模
- 文化分布
- 来源分布
- source confound 摘要
- interactions 摘要
- issues 表格

这很适合人直接阅读和汇报。

## 8. 核心函数 `audit_dataset_v4(...)`
### 参数解释
`metadata_csv`

- 要审计的主 metadata

`out_dir`

- 输出报告目录

`interactions`

- 可选的交互文件列表
- 传入后会一起审计

`dataset_name`

- 报告中显示的数据集名称

`thresholds`

- 审计阈值配置

### 它的主流程
1. 读取 metadata
2. 调用 `_metadata_report(...)`
3. 从 metadata 构造 `track_to_culture`
4. 如果传入 interactions，就逐个调用 `_interactions_report(...)`
5. 汇总成总报告
6. 写出多个报告文件

## 9. 它最终会写出哪些文件
在 `out_dir` 下会生成：

- `dataset_profile.json`
- `dataset_profile.md`
- `schema_report.json`
- `missingness_report.json`
- `duplicate_report.json`
- `source_confound_report.json`

这说明它并不是只给一份总报告，而是把不同维度的审计拆开保存。

## 10. 为什么这个脚本重要
很多项目的数据处理停留在“能跑通”层面，但这个脚本体现的是“能解释、能审计、能批判性分析”。

尤其是 `source_confound` 这一段，非常能体现研究视角，因为它不是只看样本数够不够，而是在问：

- 你以为学到的是文化差异，还是其实学到了来源差异？

这类问题在跨文化数据集里尤其重要。

## 11. 局限与注意点
### 局限 1：审计主要基于 metadata 和 interactions
它不直接检查 embedding 数值质量，那是 `validate_dataset.py` 的职责。

### 局限 2：阈值仍是人工设定
阈值合理性依赖任务背景。

### 局限 3：`source_confound` 是结构性指示，不等于完整因果证明
它能提示高风险，但不能单独证明模型一定受混淆影响。

## 12. 面试时怎么讲
可以这样概括：

“`audit_dataset_v4.py` 是 V4 数据集的治理与审计脚本。它系统检查 metadata 的字段覆盖率、重复情况、文化分布、音频可用性，并特别构造 `source_confound` 统计来评估文化标签与来源数据集之间是否高度绑定。同时它还能联动检查 interactions 的未知 track、重复 pair 和用户活跃度。它体现的是一种研究级的数据集审计视角，而不只是工程级跑通。”

## 13. 建议重点看哪里
优先看：

1. `MetadataAuditThresholds`
2. `_field_coverage_report(...)`
3. `_metadata_report(...)`
4. `source_confound_report` 的构造逻辑
5. `_interactions_report(...)`

把这些地方看懂，你几乎就掌握了整个数据审计层。
