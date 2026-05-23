# `import_europeana_audio_search.py` 代码说明

对应源码：[E:/Desktop/Echo/dcas/scripts/import_europeana_audio_search.py](E:/Desktop/Echo/dcas/scripts/import_europeana_audio_search.py)

## 1. 这个文件在数据链路里负责什么
这个脚本是“检索式采集入口”。

与前两个 HF 导入器不同，它不是从固定数据集仓库里读样本，而是：

- 根据搜索词调用 Europeana API
- 在检索结果里筛选可接受的音频条目
- 下载满足条件的音频
- 生成统一 `metadata.csv`

所以它更像一个“面向开放文化遗产平台的搜索抓取器”。

## 2. 它解决的核心问题
开放平台上的数据往往不是现成的训练集，而是海量检索结果。项目要把它变成研究语料，就必须解决三个问题：

1. 搜出来的不一定都是音频
2. 即使是音频，也不一定有可接受的授权
3. 就算授权可用，也不一定有可直接下载的音频地址

这个脚本就是围绕这三件事设计的。

## 3. 辅助函数流程说明
这个文件的辅助函数很少，但正好卡在搜索式采集最关键的两步：

1. `_slug(...)`
   - 把 Europeana 条目 ID 清洗成稳定 `track_id`
   - 保证下载后的本地文件与 metadata 能一一对应
2. `_is_allowed_rights(...)`
   - 在每条搜索结果进入下载前，先判断 rights URL 是否通过白名单
   - 它是整条采集链路里的合规过滤闸门

所以虽然主流程看起来是“搜索 -> 下载”，但辅助函数实际上先帮它解决了：

- 样本身份怎么稳定命名
- 哪些结果从授权层面根本不该进入数据集

## 4. 辅助函数说明
### `_slug(v)`
作用：

- 把 Europeana 的条目 ID 清洗成安全的 `track_id`

### `_is_allowed_rights(rights_url)`
作用：

- 判断某条 Europeana 结果的 rights URL 是否在允许范围内

允许的 token 包括：

- 多种 Creative Commons 许可
- 公有领域或部分 rightsstatements 许可

意义：

- 这是这个脚本最重要的合规过滤器之一
- 它体现了项目不是“搜到了就抓”，而是先做授权筛查

## 5. 核心函数 `import_europeana_audio_search(...)`
### 4.1 函数职责
它负责：

- 发起分页搜索
- 对结果做类型、授权、下载地址过滤
- 下载实际音频文件
- 输出项目内部 metadata

### 4.2 参数解释
`query`

- Europeana 搜索查询词
- 决定搜什么主题或文化对象

`out_dir`

- 输出目录
- 会生成：
  - `audio/`
  - `metadata.csv`
  - `import_report.json`

`culture`

- 这批搜索结果在项目里的目标文化域标签

`limit`

- 最终最多导入多少条

`wskey`

- Europeana API key
- 默认是 `api2demo`

`rows_per_page`

- 每页检索多少条

`use_cursor`

- 是否使用 cursor 分页
- 如果关闭，则退化到基于 `start` 的翻页

### 4.3 核心筛选逻辑
每条 item 进入本地数据集之前，会经过这些筛选：

1. `type` 必须是 `SOUND`
2. `rights` 必须存在，而且每个 rights URL 都要通过 `_is_allowed_rights(...)`
3. 必须有 `edmIsShownBy`
4. 当前实现里只接收 `.mp3` 直链

只有通过以上条件，才会实际下载音频。

这四层过滤非常值得你记住，因为它们基本就构成了“搜索平台型采集脚本的质量门槛”。

### 4.4 下载与落盘
通过筛选后，脚本会：

1. 从 `item["id"]` 生成 `track_id`
2. 下载 `audio_url`
3. 保存到本地 `audio/<track_id>.mp3`
4. 提取标题、rights、guid 等字段
5. 写入 metadata 行

## 6. 输出字段说明
写出的字段有：

- `track_id`
- `culture`
- `audio_path`
- `source_dataset`
- `source_split`
- `source_index`
- `label`
- `title`
- `rights`
- `source_url`

其中：

- `source_dataset` 固定为 `europeana_search`
- `source_split` 固定为 `search`

这两个字段在后续审计时很有意义，因为能明确区分“平台搜索导入”与“仓库导入”。

## 7. 为什么这个脚本重要
它展示了项目在采集层并不只依赖现成数据集，而是也能从开放平台通过搜索策略构建候选样本池。

换句话说，它体现的是“研究数据集的主动构造能力”，而不是被动下载。

## 8. 设计优点
### 优点 1：先做授权过滤再下载
这体现了合规意识。

### 优点 2：支持分页和 cursor
说明它能处理不止一页的小规模试验，也能处理更大的检索集合。

### 优点 3：输出结构仍然兼容统一 metadata
即使来源是搜索 API，下游仍可复用相同流水线。

## 9. 局限与注意点
### 局限 1：目前只接受 `.mp3`
这会牺牲一部分潜在可用资源，但实现更稳定。

### 局限 2：授权过滤是白名单式的
安全，但也意味着某些边界许可会被保守排除。

### 局限 3：它只做浅层元数据抽取
不会做更深层的文化判别或标签统一。

## 10. 面试时可以怎么讲
可以这样表述：

“`import_europeana_audio_search.py` 是一个平台搜索式采集脚本。它通过 Europeana API 按查询词检索候选音频，再依次做类型过滤、rights 白名单过滤、下载链接过滤，最后把结果统一落成 `audio/ + metadata.csv`。它体现了项目不仅能消费现成数据集，也能从开放文化平台主动构建候选研究语料。”

## 11. 建议重点看哪里
优先看：

1. `_is_allowed_rights(...)`
2. `while imported < limit` 这个分页抓取循环
3. item 级别的 4 层过滤条件
4. metadata 行最终写出的字段

把这几处看懂，你就能讲清“开放平台搜索式采集”的方法论。
