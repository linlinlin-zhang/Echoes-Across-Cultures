# `harmonize_v3_metadata.py` 代码说明

对应源码：[E:/Desktop/Echo/dcas/scripts/harmonize_v3_metadata.py](E:/Desktop/Echo/dcas/scripts/harmonize_v3_metadata.py)

## 1. 这个文件在数据链路里负责什么
这个脚本负责把 V3 阶段来源各异的原始标签，统一映射成项目内部更稳定、更可比较的标签字段。

它的核心产物只有两个新增列：

- `coarse_label`
- `is_instrumental`

但这两个字段非常重要，因为它们承担的是“跨来源统一语义层”的作用。

换句话说，这个脚本不负责采集音频，也不负责 schema 治理，它主要负责回答：

- 这些来源不同、命名混乱的音乐样本，在项目内部应该如何粗粒度归类？
- 哪些样本可以视作器乐类？

## 2. 整体执行流程
主流程在 `harmonize_metadata(...)` 中：

1. 读取输入 `metadata.csv`
2. 尝试加载 FMA 的 genre id -> title 映射
3. 遍历每一行
4. 标准化 `language`
5. 通过 `_coarse_label(...)` 生成统一粗标签
6. 根据 `instrument_family` 生成 `is_instrumental`
7. 写出新的 CSV
8. 生成 `.report.json`

## 3. 辅助函数流程说明
这个文件的辅助函数其实构成了一条非常完整的标签统一流水线：

1. `_read_rows(...)`
   - 先把原始 metadata 读进来
2. `_load_fma_genre_titles(...)`
   - 如果需要，就从 FMA metadata 里准备 genre id -> title 映射表
3. `_parse_label_titles(...)`
   - 把某条样本原始 `label` 解析成更稳定的标题列表
4. `_norm(...)`
   - 对参与判断的文本做小写化和轻量标准化
5. `_coarse_label(...)`
   - 综合 `substyle / source_dataset / label / era / instrument_family` 等信息，产出统一粗标签
6. `_write_rows(...)`
   - 把新增字段写回新的 metadata

也就是说，这些辅助函数合起来刚好对应：

- 读入原始标签
- 把标签解释成可用文本
- 做统一语义判断
- 再稳定地写回表中

## 4. 顶部常量的意义
### `REPO_ROOT`
- 仓库根目录

### `DEFAULT_FMA_METADATA_ZIP`
- 默认的 FMA metadata zip 路径

意义：

- 某些来源中的 `label` 不是文本，而是 FMA genre id 列表
- 需要用 FMA 官方 metadata 把 id 恢复成可读标题

## 5. 辅助函数说明
### `_read_rows(path)`
作用：

- 读取输入 CSV，并返回行列表和字段名列表

### `_write_rows(path, rows, fieldnames)`
作用：

- 按指定字段顺序写出 CSV

### `_load_fma_genre_titles(zip_path)`
作用：

- 从 `fma_metadata.zip` 中读取 `genres.csv`
- 构造 `genre_id -> genre_title` 的映射字典

意义：

- 如果原始 `label` 里保存的是数字 id，这个映射是后续语义判断的基础

### `_parse_label_titles(raw_label, fma_genre_map)`
作用：

- 把原始 `label` 解析成文本标题列表

它兼容两种情况：

- 纯文本标签
- 列表形式的 genre id / genre title

如果是数字 id，会通过 `fma_genre_map` 恢复成 genre title。

### `_norm(x)`
作用：

- 做很轻量的标准化
- 转成字符串、去空白、再转小写

### `_coarse_label(row, fma_genre_map)`
作用：

- 这是整个文件最核心的函数
- 它根据多列信息，推断统一的 `coarse_label`

## 6. `_coarse_label(...)` 为什么是这个文件的核心
这个函数本质上是一个“规则型标签统一器”。

它会综合使用这些信息：

- `substyle`
- `source_dataset`
- `era`
- `instrument_family`
- `language`
- 由 `label` 解析出的 genre titles
- `instrument`
- `title`
- `artist`

然后按一系列规则进行判断。

### 5.1 第一层：强规则直映射
例如：

- `jingju_acappella -> traditional_vocal`
- `traditional_instrumental -> traditional_instrumental`
- `hindustani_art_music -> art_music`
- `gamelan_orchestra -> traditional_instrumental`
- `mandarin_pop_singing -> modern_pop_song`

这类规则的特点是：

- 语义最明确
- 优先级最高

### 5.2 第二层：按来源数据集直推
例如：

- `opencpop -> modern_pop_song`
- `saraga_hindustani -> art_music`
- `compmusic_jingju_acappella -> traditional_vocal`
- `ccmusic-database/ctis -> traditional_instrumental`

这体现了一个很现实的工程思想：

- 有些来源本身就高度纯净
- 那么直接按来源指定粗标签是合理且高效的

### 5.3 第三层：按文本关键词推断
例如：

- 命中 `opera / choral / choir / orchestra / classical / soundtrack`
  - -> `soundtrack_classical`
- 命中 `jazz / blues`
  - -> `jazz_blues`
- 命中 `folk / acoustic / singer-songwriter / chanson`
  - -> `folk_acoustic`
- 命中 `ambient / drone / instrumental`
  - -> `instrumental_ambient`
- 命中 `pop / easy listening / song`
  - -> `modern_song`

这层规则用来兜住那些没有强结构标签、但文本线索足够明显的样本。

### 5.4 第四层：按结构属性回退
例如：

- `instrument_family == voice` 且 `era == traditional`
  - -> `traditional_vocal`
- `instrument_family == traditional_instrument`
  - -> `traditional_instrumental`
- `era == modern`
  - -> `modern_song`

最后兜底到：

- `unknown`

### 5.5 为什么这种设计合理
它体现了一个很典型的标签统一策略：

1. 先用最可信的结构化强信号
2. 再用来源先验
3. 再用文本关键词
4. 最后用较粗的属性回退

这比只看某一列要稳得多。

## 7. 核心函数 `harmonize_metadata(...)`
### 参数解释
`metadata_csv`

- 输入 metadata 表

`out_csv`

- 输出路径

`fma_metadata_zip`

- FMA metadata zip 路径
- 用于恢复 FMA genre titles

### 它做的事情
这个函数会：

1. 读入原始行
2. 构造 `fma_genre_map`
3. 逐行计算：
   - 规范化后的 `language`
   - `coarse_label`
   - `is_instrumental`
4. 把新增列追加到字段列表
5. 写出新表
6. 统计各 `coarse_label` 的分布，写到报告里

### `is_instrumental` 是怎么定义的
当前实现很保守：

- 如果 `instrument_family == traditional_instrument`
  - 写成 `"1"`
- 否则写成 `"0"`

这说明它不是一个通用“所有器乐 = 1”的判别器，而是更偏项目语境下的传统器乐标志。

## 8. 输出报告说明
`.report.json` 中会记录：

- 输入路径
- 输出路径
- 行数
- 每个 `coarse_label` 的数量分布

这能帮助你快速检查标签统一后是否出现异常偏斜。

## 9. 这个文件的价值
这是项目里非常关键的一层，因为如果没有统一粗标签：

- 不同来源之间几乎不可比
- 后续分析只能困在各自来源自己的 label 体系里
- 审计和实验汇报会非常混乱

所以它本质上是在把“来源自己的标签系统”翻译成“项目自己的标签系统”。

## 10. 局限与注意点
### 局限 1：规则驱动，非学习式
它依赖人工规则，因此覆盖率和准确率取决于规则设计质量。

### 局限 2：粗粒度而非细粒度
它不追求精确到所有流派，只追求得到稳定、可比较的粗语义层。

### 局限 3：`unknown` 不可避免
对信息过少或规则未覆盖的样本，只能回退到 `unknown`。

## 11. 面试时怎么讲
可以这样概括：

“`harmonize_v3_metadata.py` 是项目的标签统一层。它把各来源异构的 `label / substyle / source_dataset / era / instrument_family` 等信息通过一套分层规则，映射成统一的 `coarse_label`，并额外生成 `is_instrumental`。这样做的目的不是追求细粒度分类，而是构建一个跨来源可比较的语义层，为后续分析、审计和实验打基础。”

## 12. 建议重点看哪里
优先看：

1. `_load_fma_genre_titles(...)`
2. `_parse_label_titles(...)`
3. `_coarse_label(...)`
4. `harmonize_metadata(...)`

这四处加起来，基本就是整个文件的全部价值所在。
