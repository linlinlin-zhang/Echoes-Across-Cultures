# `harmonize_v4_metadata.py` 代码说明

对应源码：[E:/Desktop/Echo/dcas/scripts/harmonize_v4_metadata.py](E:/Desktop/Echo/dcas/scripts/harmonize_v4_metadata.py)

## 1. 这个文件在数据链路里负责什么
这个脚本是 V4 阶段的“schema 标准化主力”。

如果说：

- `merge_metadata_dedup.py` 负责把多份表拼成一张表
- `harmonize_v3_metadata.py` 负责统一粗标签

那么这个脚本负责的是更完整的一层：

- 统一字段名和字段语义
- 补齐缺失的基础音频信息
- 写入治理字段
- 再调用 V3 的标签统一逻辑
- 最终得到 V4 可发布的标准 metadata

它是从“合并后的原始主表”走向“规范发布主表”的关键步骤。

## 2. 顶部常量分别表达什么
### `REQUIRED_FIELDS`
这些字段是 V4 主表最核心、最不能缺的字段：

- `track_id`
- `culture`
- `audio_path`
- `source_dataset`
- `source_split`
- `source_index`
- `duration_sec`
- `sample_rate`
- `channels`
- `era`
- `region`

### `RECOMMENDED_FIELDS`
这些字段不是强制，但很有价值：

- `fine_label`
- `label`
- `substyle`
- `instrument`
- `instrument_family`
- `language`
- `title`
- `artist`
- `license`
- `license_note`
- `url`
- `recording_condition`
- `notes`

### `GOVERNANCE_FIELDS`
这些字段是数据治理和生命周期管理需要的：

- `schema_version`
- `dataset_version`
- `import_batch`
- `dedup_group_id`
- `dedup_keep`
- `qc_status`
- `qc_notes`
- `embedding_status_culturemert`
- `embedding_status_gemini`
- `drop_reason`

### `FINAL_FIELD_ORDER`
这是最终输出字段顺序。

它把：

- 必需字段
- `coarse_label / is_instrumental`
- 推荐字段
- 治理字段

按一个稳定顺序拼接起来。

## 3. 辅助函数流程说明
这个文件的辅助函数可以按“单行标准化流水线”来理解：

1. `_read_rows(...)`
   - 先把 merge 后的原始主表读进来
2. `_clean_text(...)`
   - 先对每个字段做缺失值语义清洗
3. `_normalize_row(...)`
   - 在单条记录层面统一字段名、补回退字段、写治理字段、必要时从音频里反补时长和采样率
4. `_write_rows(...)`
   - 先写出 clean 版本，再写出 harmonized 版本

然后在这个辅助链条之外，再由主流程调用 `harmonize_metadata(...)` 完成 `coarse_label / is_instrumental` 的统一。

所以这里最值得抓住的是：

- `_clean_text(...)` 解决“值是否干净”
- `_normalize_row(...)` 解决“字段语义是否统一”
- `harmonize_metadata(...)` 解决“标签语义是否统一”

## 4. 辅助函数说明
### `_read_rows(path)`
- 读取 CSV 和字段名

### `_write_rows(path, rows, fieldnames)`
- 按指定顺序写出 CSV

### `_clean_text(value)`
作用：

- 去空白
- 把 `nan / none / null` 统一视为空

意义：

- 上游来源中这类伪缺失值非常常见
- 不先规范掉，后面的覆盖率统计会失真

### `_normalize_row(...)`
这是整个文件最重要的内部函数。

它负责对单条 metadata 做 V4 级别的标准化。

## 5. `_normalize_row(...)` 逐层在做什么
### 4.1 先做一层全字段文本清洗
它会先把当前行已有字段和 `FINAL_FIELD_ORDER` 中涉及到的字段全部过一遍 `_clean_text(...)`。

这一步的目标是：

- 先把空值语义统一
- 再做后续字段映射

### 4.2 统一关键字段来源
例如：

- `sample_rate` 如果空，则尝试用 `sample_rate_hz`
- `channels` 如果空，则尝试用 `num_channels`
- `fine_label` 如果空，则回退到 `label`
- `region` 如果空，则回退到 `culture`

这一步体现的不是“简单复制”，而是“吸收异构来源中等价字段”。

### 4.3 语言字段小写化
`language` 会统一转小写。

意义：

- 避免 `EN`、`en`、`En` 这类格式差异影响后续判断

### 4.4 写入治理字段
例如：

- `schema_version`
- `dataset_version`
- `import_batch`
- `dedup_group_id`
- `dedup_keep`
- `qc_status`
- `embedding_status_culturemert`
- `embedding_status_gemini`

这些字段很重要，因为它们让数据集不只是“内容集合”，还是“可治理资产”。

### 4.5 用音频文件反补缺失信息
如果这些字段缺失：

- `duration_sec`
- `sample_rate`
- `channels`

脚本会用 `torchaudio.info(...)` 直接读音频文件补上。

这一步特别关键，因为它把“从元数据继承字段”升级成了“从真实音频资产校验并回填字段”。

## 6. 核心函数 `harmonize_v4_metadata(...)`
### 参数解释
`metadata_csv`

- 输入主表，通常是 merge 后的 `metadata_raw.csv`

`out_clean_csv`

- 标准化但还未做统一标签增强的输出

`out_harmonized_csv`

- 最终 harmonized 输出

`dataset_version`

- 当前数据集版本号
- 会写入治理字段

`schema_version`

- 当前 schema 版本号

`import_batch`

- 当前导入批次标识
- 若为空，默认用 `dataset_version`

`fma_metadata_zip`

- 供 `harmonize_v3_metadata.py` 使用的 FMA metadata zip 路径

### 它的主流程
1. 读取输入 metadata
2. 对每一行调用 `_normalize_row(...)`
3. 写出 `clean` 版本
4. 调用 `harmonize_metadata(...)` 生成 `coarse_label / is_instrumental`
5. 再次读取 harmonized 结果
6. 以 `FINAL_FIELD_ORDER` 统一字段顺序重新写出
7. 生成报告

这意味着它既做 schema 规范化，也串起标签统一。

## 7. 为什么它要分成 `clean` 和 `harmonized` 两步
这是个很好的工程设计点。

### `clean`
主要解决：

- 字段命名和缺失值
- 基础音频属性补齐
- 治理字段写入

### `harmonized`
主要解决：

- 统一粗标签
- 最终字段排序与发布形态

这样分层的好处是：

- 每一步职责更清晰
- 更容易定位错误
- 中间结果也可以单独审查

## 8. 输出报告说明
`.report.json` 会记录：

- 输入路径
- clean 输出路径
- harmonized 输出路径
- `dataset_version`
- `schema_version`
- 行数
- clean 字段数
- harmonized 字段数

## 9. 这个脚本的价值
它相当于把一张“能看懂但不规范”的 metadata 表，变成了一张“可发布、可审计、可继续下游消费”的正式主表。

很多项目做到 merge 就停了，但这个项目继续往前做了：

- 基础字段补齐
- 治理字段显式化
- 标签统一
- 字段顺序固化

这恰恰是一个成熟数据工程流程的体现。

## 10. 局限与注意点
### 局限 1：依赖音频文件可访问
如果 `audio_path` 指向的文件不存在，反补音频属性就会失败。

### 局限 2：治理字段默认值偏保守
例如 `qc_status = pending`、embedding 状态默认 `pending`，意味着这些状态需要后续流程继续更新。

### 局限 3：标签统一逻辑仍依赖 V3 规则
也就是它本身不重新发明标签体系，而是复用现有统一器。

## 11. 面试时怎么讲
可以这样表述：

“`harmonize_v4_metadata.py` 是 V4 阶段的 schema 标准化与治理入口。它先逐行清洗空值、统一等价字段、补齐音频基础属性，并写入版本、批次、QC、embedding 状态等治理字段；然后再复用 V3 的标签统一逻辑生成 `coarse_label` 和 `is_instrumental`。最终输出的是一张具备稳定字段顺序和治理语义的正式 metadata 主表。”

## 12. 建议重点看哪里
优先看：

1. `REQUIRED_FIELDS / RECOMMENDED_FIELDS / GOVERNANCE_FIELDS`
2. `_normalize_row(...)`
3. `torchaudio.info(...)` 反补音频属性的逻辑
4. `harmonize_v4_metadata(...)` 中 clean -> harmonized 的两阶段流程

这几个地方最能体现这个文件的工程价值。
