# `build_research_dataset_v3.py` 代码说明

对应源码：
[E:/Desktop/Echo/dcas/scripts/build_research_dataset_v3.py](E:/Desktop/Echo/dcas/scripts/build_research_dataset_v3.py)

## 1. 这个文件在整个项目里负责什么

这个文件是 `V3` 数据集的“主构建脚本”。

它做的不是单一来源的数据下载，而是把多个不同风格、不同组织形式、不同授权条件、不同元数据质量的音频来源，按文化域分别处理，再统一整理成项目内部可以继续使用的标准形态：

- 每个文化域一个目录
- 每个目录下面有 `audio/`
- 每个目录下面有一个 `metadata.csv`
- 最后把多个文化域的 `metadata.csv` 合并成一个总表 `metadata_v3_main.csv`
- 再生成一个 `summary_v3_main.json`，用于汇总每个文化域最终保留了多少样本、时长范围如何、来源有哪些

简单说，这个文件负责回答四个问题：

- 数据从哪里来
- 每个文化域怎么选
- 样本为什么被保留或丢弃
- 多个文化域最后如何合成一套研究数据

## 2. 整体执行流程

这份脚本的主流程在 `build_research_dataset_v3(...)` 中，整体顺序是：

1. 准备输出目录、原始缓存目录、FMA 元数据缓存目录。
2. 构建 India 域。
3. 构建 Turkey 域。
4. 构建 China 域。
5. 构建 Indonesia 主域。
6. 构建现代英语流行对比域。
7. 构建 Indonesia probe 域。
8. 构建 FMA 西方国家域。
9. 把主域合并成 `metadata_v3_main.csv`。
10. 生成总览汇总 `summary_v3_main.json`。

这里最重要的设计思想是：

- 不假设所有来源都能走同一条通用导入流程。
- 每个文化域都允许有定制化处理逻辑。
- 先把每个域做扎实，再在最后一步统一合并。

## 3. 辅助函数流程说明
这个文件的辅助函数很多，但如果按主流程依赖关系去看，会更容易抓住重点。大体可以分成 5 层：

1. 基础文本与数值清洗层
   - `_slug(...)`、`_norm_text(...)`、`_to_text(...)`、`_clean_optional_text(...)`、`_safe_float(...)`
   - 作用是把不同来源里的脏文本、脏数值先规整好
2. 音频与文件 I/O 层
   - `_duration_from_bytes(...)`、`_duration_from_file(...)`、`_copy_bytes(...)`、`_copy_file(...)`
   - 作用是让脚本既能处理本地文件，也能处理下载后的临时音频
3. 通用表处理层
   - `_read_csv(...)`、`_write_csv(...)`、`_summarize_rows(...)`
   - 作用是把每个文化域最终都落成统一 metadata，并能汇总统计
4. 样本筛选与多样性控制层
   - `_contains_normalized_term(...)`、`_round_robin_diverse(...)`、`_uniform_subsample_rows(...)`、`_parse_list_label(...)`
   - 作用是做关键词判断、子采样和艺术家多样性控制
5. 文化域构建层
   - 各种 `_build_*` 函数
   - 作用是把前面几层能力真正组合成某个文化域的数据构建逻辑

所以读这个文件时，最推荐的顺序不是按源码行号硬啃，而是按这条链来理解：

- 先看基础工具
- 再看筛选与 I/O
- 最后看各个 `_build_*` 如何复用这些工具拼成具体文化域

## 4. 文件最上方的常量在表达什么

### 3.1 路径常量

- `DEFAULT_OUT_ROOT`
  - 默认输出目录。
  - 也就是最终 `research_dataset_v3` 的根目录。

- `DEFAULT_RAW_ROOT`
  - 默认原始数据目录。
  - 存放 zip、缓存网页、临时下载内容等。

- `DEFAULT_CACHE_ROOT`
  - 默认缓存目录。
  - 主要用于保存 FMA 筛选缓存、中间结果缓存，避免每次都重新爬取或重新筛选。

### 3.2 数据源常量

- `FMA_REPO_ID`
  - FMA 元数据来源对应的仓库 ID。

- `FMA_METADATA_ZIP`
  - FMA 本地元数据 zip 的位置。
  - 后续国家筛选、流派排除都依赖它。

- `OPENCPOP_SONGLIST_URL`
  - OpenCpop 歌曲列表网页。
  - 用来抓歌名、BPM、节拍等信息。

### 3.3 目标规模常量

- `CHINA_JINGJU_TARGET`
  - 中国域中京剧片段目标数。

- `CHINA_OPENCPOP_TARGET`
  - 中国域中 OpenCpop 目标数。

### 3.4 FMA 筛选规则常量

- `COUNTRY_PATTERNS`
  - 用于根据 artist location 文本判断属于哪个国家。

- `FMA_COUNTRY_CODES`
  - 用于把 reverse geocoder 的国家代码映射到文化域名。

- `FMA_SUPPLEMENT_COUNTRY_PATTERNS`
  - 用于补充 Indonesia 相关候选的国家文本匹配。

- `FMA_SUPPLEMENT_COUNTRY_CODES`
  - Indonesia 补充域的国家代码映射。

- `FMA_BANNED_GENRES`
  - FMA 国家域中不希望进入主数据集的流派集合。
  - 目的不是“审美过滤”，而是减少与项目目标不相关或噪声较大的流派。

- `ANGLO_POP_BANNED_TERMS`
  - 现代英语流行域中要排除的词。
  - 典型用于把 `pop` 基准域尽量收窄到更可控的对照集合。

## 5. 辅助工具函数在做什么

这一组函数不是数据集构建主逻辑，而是给主逻辑提供稳定、可复用的小能力。

### `_slug(value)`

作用：

- 把任意字符串清洗成适合做文件名、track_id 的安全字符串。

意义：

- 不同数据源的原始标题、ID、路径可能带空格、斜杠、特殊字符。
- 统一 slug 后，后续落盘和合并更稳定。

### `_norm_text(value)`

作用：

- 把文本转成小写、做基础规范化、去掉噪声符号。

意义：

- 后续做地理位置匹配、关键词匹配、去重 key 构造时，要依赖规范化文本。

### `_strip_audio_exts(filename)`

作用：

- 反复剥离音频扩展名。

意义：

- 某些文件名可能带多层扩展或打包过程中出现双后缀，这个函数保证得到干净 stem。

### `_to_text(value)`

作用：

- 把各种值都安全转成字符串。
- 对字典和列表会转成 JSON 字符串。

意义：

- `metadata.csv` 最终是字符串表格，这个函数用于统一写出格式。

### `_safe_console_text(value)`

作用：

- 把文本转成适合终端输出的 ASCII 安全形式。

意义：

- 爬取和下载阶段的日志里可能出现无法直接打印的字符。

### `_clean_optional_text(value)`

作用：

- 处理可选字段。
- 把 `nan`、`none`、空值清成空字符串。

意义：

- 避免 metadata 里混入很多伪缺失值字符串。

### `_safe_float(value)`

作用：

- 尝试把值转成有限浮点数。
- 失败就返回 `None`。

意义：

- 元数据里经常有时长、播放量、收藏量等数值字段，但来源格式并不总干净。

### `_contains_normalized_term(text, term)`

作用：

- 在规范化文本中判断是否包含某个词。

意义：

- 做地理位置、国家、流派判断时使用。

### `_duration_from_bytes(data)` / `_duration_from_file(path)`

作用：

- 读取音频时长。

意义：

- 脚本大量使用 `duration >= 30s` 作为样本保留门槛。
- 这是这套数据线最核心的过滤条件之一。

### `_write_csv(path, rows)`

作用：

- 把若干行 metadata 写成 CSV。

特点：

- 它会优先保证一些关键字段排在前面，如：
  - `track_id`
  - `culture`
  - `audio_path`
  - `source_dataset`
  - `source_split`
  - `source_index`
  - `label`
  - `title`
  - `artist`
  - `duration_sec`

意义：

- 这不是简单导出，而是在有意识地把主键、来源、标签、时长这些最常用字段放在前面，方便人工查看和后续脚本消费。

### `_read_csv(path)`

作用：

- 读取 CSV 为行字典列表。

### `_copy_bytes(path, payload)` / `_copy_file(src, dst)`

作用：

- 把音频字节或本地文件复制到目标目录。

意义：

- 这个脚本最终要求每个文化域都在本地拥有实际音频资产，而不是只保留一个外链。

### `_round_robin_diverse(rows, target_n, artist_key, max_per_artist=3)`

作用：

- 做一种“按艺术家轮转”的多样性采样。

意义：

- 避免某个大艺术家或某个高产来源在最终样本中占比过高。
- 这一步是构建“研究可用数据”时非常典型的多样性控制逻辑。

关键参数：

- `rows`
  - 候选样本列表。

- `target_n`
  - 最终想要保留的样本数。

- `artist_key`
  - 从哪一列读取艺术家信息。

- `max_per_artist`
  - 单个艺术家在轮转阶段最多拿多少条。

### `_summarize_rows(rows)`

作用：

- 汇总一个文化域的样本数、艺术家数、时长统计、来源集合。

意义：

- 给 `summary_v3_main.json` 服务。

### `_domain_out(out_root, culture)`

作用：

- 生成某个文化域的输出目录，并确保 `audio/` 子目录存在。

意义：

- 统一每个域的目录结构。

## 6. 每个文化域是怎么构建的

这部分是整个文件最重要的内容。

### `_build_india(out_root, raw_root)`

作用：

- 从 `saraga_hindustani.zip` 中抽取 India 域音频与 metadata。

核心逻辑：

- 遍历 zip 里的音频文件。
- 读取音频时长。
- 只保留 `>= 30s` 的片段。
- 从目录结构和文件名中抽取标题、作品信息、艺术家信息。
- 复制音频到 `india/audio/`。
- 写出 `india/metadata.csv`。

输出字段中比较重要的有：

- `culture = india`
- `source_dataset = saraga_hindustani`
- `substyle = hindustani_art_music`
- `era = traditional`

这个函数的意义：

- 它把一个相对原始的压缩包语料，重组为项目统一格式。
- 同时保留传统音乐研究中有价值的作品和来源说明。

### `_build_turkey(out_root, target_n)`

作用：

- 从较早版本已经准备好的土耳其音频表中挑选现代土耳其歌曲样本。

核心逻辑：

- 读取旧版 `metadata.csv`。
- 对每条音频重新计算时长。
- 只保留 `>= 30s`。
- 取前 `target_n` 条。
- 重新命名为 `turkey_modern_XXXX`。

关键参数：

- `target_n`
  - 最终保留的样本数。

这个函数的意义：

- 说明这个项目并不是“每次从零开始”，而是会复用前一版已经落地的数据资产。

### `_build_jingju_rows(out_dir, raw_root)`

作用：

- 从京剧无伴奏 wav 压缩包中构建中国传统声乐子集。

核心逻辑：

- 遍历 zip 中的 wav。
- 只保留 `>= 30s`。
- 解析文件名，提取角色、唱段、作品等信息。
- 标注：
  - `substyle = jingju_acappella`
  - `language = zh`
  - `instrument_family = voice`
  - `era = traditional`

意义：

- 这个函数是中国域“传统声乐部分”的主要来源。

### `_uniform_subsample_rows(rows, target_n)`

作用：

- 如果候选太多，均匀抽样到目标数量。

意义：

- 它不是简单取前 N 条，而是尽量在已有顺序上均匀覆盖整个集合。
- 中国京剧子集使用它来控制规模。

### `_build_ctis_rows(limit=None)`

作用：

- 从旧版 CTIS 元数据中构造中国传统器乐子集。

核心逻辑：

- 读取旧版 `china/metadata.csv`。
- 重新计算音频时长。
- 只保留 `>= 30s`。
- 设定：
  - `substyle = traditional_instrumental`
  - `instrument_family = traditional_instrument`
  - `era = traditional`

关键参数：

- `limit`
  - 可选上限。

意义：

- 中国域不仅需要声乐，还需要器乐，CTIS 负责补足这一部分。

### `_load_opencpop_songlist(raw_root)`

作用：

- 获取 OpenCpop 的歌曲列表。

核心逻辑：

- 如果本地缓存存在就直接读。
- 否则用 `pandas.read_html` 抓网页表格。
- 统一列名后缓存为 CSV。

意义：

- 这一步把网页结构化信息变成了可复用本地缓存。

### `_select_opencpop_songlist(df, target_n)`

作用：

- 从 OpenCpop 歌表里选一批样本。

核心逻辑：

- 如果歌曲总数本来就少于目标数，就全部保留。
- 否则按 `bpm + song_id` 排序后等距取样。

意义：

- 目标不是随机，而是尽量从节奏分布上均匀覆盖。

### `_build_opencpop_rows(out_dir, raw_root, password, target_n)`

作用：

- 从 OpenCpop 压缩包中提取现代中文流行样本。

关键参数：

- `password`
  - 压缩包密码。

- `target_n`
  - 目标样本数。

核心逻辑：

- 找到 `wavs_raw.zip` 或备选 zip。
- 没有密码就直接报错。
- 先选歌，再按 song_id 解压对应音频。
- 标注：
  - `label = mandarin_pop`
  - `substyle = mandarin_pop_singing`
  - `language = zh`
  - `instrument_family = voice`
  - `era = modern`

意义：

- 它为中国域补充现代流行部分，避免中国域只剩传统音乐。

### `_build_china(out_root, raw_root, jingju_target, opencpop_target, opencpop_password=None)`

作用：

- 把中国域的多个来源拼成一个文化域目录。

核心逻辑：

- 京剧部分先构建并按 `jingju_target` 控制规模。
- CTIS 器乐部分全部接入。
- OpenCpop 现代流行部分按 `opencpop_target` 接入。
- 三部分统一写入一个 `china/metadata.csv`。

关键参数：

- `jingju_target`
  - 京剧目标数。

- `opencpop_target`
  - OpenCpop 目标数。

- `opencpop_password`
  - OpenCpop 压缩包密码。

这个函数的意义：

- 它体现了“一个文化域可以由多个子来源拼成”的核心思想。

### `_parse_list_label(value)`

作用：

- 解析存成字符串的列表标签。

意义：

- Anglo-pop 基准域中需要解析多标签 `label` 列。

### `_build_anglo_pop(out_root, target_n)`

作用：

- 从旧版英文流行候选集里，挑出一个现代英语流行对照域。

核心逻辑：

- 读取旧版 `anglo_pop/metadata.csv`。
- `label` 中必须含有 `pop`。
- 不能包含禁词，如 `experimental`、`electronic`、`rock` 等。
- 时长必须 `>= 30s`。
- 最终取 `target_n` 条。

意义：

- 它不是一个国家域，而是一个对照基准域。

### `_build_indonesia_probe(out_root, raw_root)`

作用：

- 构建 Indonesia 的 probe 域。

核心逻辑：

- 从 `gamelan_music_dataset.zip` 中挑 `orchestra/` 类别。
- 保留 `>= 30s`。
- 标注为：
  - `substyle = gamelan_orchestra`
  - `era = traditional`

意义：

- 这个域一开始是探针域，不一定并入主表，但可以帮助探索数据可行性。

## 7. Indonesia 主域与 FMA 补充是怎么做的

### `_build_fma_indonesia_targets(cache_root)`

作用：

- 从 FMA 元数据中筛出可能属于 Indonesia 的候选。

核心逻辑：

- 读本地 FMA tracks/genres。
- 解析流派树。
- 排除被禁用的流派。
- 根据 location、国家代码、关键词等多路规则寻找候选。
- 做去重。
- 把候选缓存下来。

意义：

- 这是“从大而杂的公开音乐库中提取非西方补充样本”的典型实现。

### `_download_fma_supplement_rows(items, out_dir, culture, track_prefix, substyle, era, workers)`

作用：

- 把前一步筛出来的 FMA 候选真正下载到本地。

执行方式分两段：

1. `prepare`
   - 访问 track page。
   - 解析真实可下载的 `fileUrl`。

2. `download_one`
   - 下载音频文件。
   - 读取时长。
   - 写出统一 metadata 行。

关键参数：

- `items`
  - 候选记录列表。

- `out_dir`
  - 输出目录。

- `culture`
  - 该批样本归属的文化域名。

- `track_prefix`
  - 生成 track_id 时使用的前缀。

- `substyle`
  - 统一写入的子风格。

- `era`
  - 统一写入的时代字段。

- `workers`
  - 并发下载线程数。

### `_build_indonesia(out_root, raw_root, cache_root, workers)`

作用：

- 生成 Indonesia 主域。

核心逻辑：

- 先从 `gamelan_music_dataset` 中取传统音乐部分。
- 再用 FMA 候选补一些现代印尼补充样本。
- 最终两部分合并成一个 Indonesia 域。

意义：

- 这个函数说明文化域构建可以采用“传统本土源 + 大型公共库补充源”的混合策略。

## 8. FMA 西方国家域是怎么做的

### `_parse_fma_duration(value)`

作用：

- 把 FMA 时长字段解析成秒数。

### `_load_fma_tracks_and_genres()`

作用：

- 从本地 `fma_metadata.zip` 读取：
  - `tracks.csv`
  - `raw_tracks.csv`
  - `genres.csv`

意义：

- 这是所有 FMA 候选筛选的元数据基础。

### `_build_fma_selected_targets(cache_root, per_country, strict_min=True)`

作用：

- 从 FMA 中为各个西方国家先筛出一批候选。

核心逻辑：

- 排除禁用流派。
- 只保留 `>= 30s`。
- 根据地理位置文本和反向地理编码判断国家。
- 按收藏数、播放量排序。
- 使用 `_round_robin_diverse` 做艺术家轮转式采样。

关键参数：

- `per_country`
  - 每个国家至少希望得到多少条候选。

- `strict_min`
  - 是否严格要求达到最小数量。

意义：

- 这是“先做元数据级筛选，再进入真正下载”的典型两阶段设计。

### `_resolve_fma_page_urls(selected, cache_root)`

作用：

- 对候选记录补全或确认其 `page_url`。

意义：

- 后续下载真实文件时，需要先访问 track page 找到可下载链接。

### `_fetch_fma_file_url(session, page_url)`

作用：

- 打开 FMA 的 track page。
- 从页面 HTML 里解析 `fileUrl`。

意义：

- FMA 元数据中的 page URL 和最终音频文件 URL 不是同一层，需要二次解析。

### `_download_file(session, url, dst)`

作用：

- 流式下载文件到本地。

### `_download_fma_rows(rows, out_root, workers, target_per_culture)`

作用：

- 对每个西方国家真正执行下载。

核心逻辑：

- 先并发解析所有候选的真实 `fileUrl`。
- 再并发下载音频。
- 对失败样本打印日志并跳过。
- 下载够数量后，统一重命名为 `country_0000` 这种稳定 ID。
- 清理没有被保留的中间文件。

关键参数：

- `rows`
  - 候选记录。

- `out_root`
  - 输出根目录。

- `workers`
  - 并发线程数。

- `target_per_culture`
  - 每个国家最终目标数。

意义：

- 这是脚本里最“工程化”的一段：既要做网络抓取，又要做失败回退，又要保证最终 ID 稳定。

### `_build_fma_western(out_root, cache_root, per_country, workers)`

作用：

- 串起：
  - 候选筛选
  - page_url 解析
  - 真实下载

最终得到：

- `germany`
- `france`
- `italy`
- `great_britain`
- `russia`

五个国家域的 `metadata.csv`

## 9. 最后的合并与汇总

### `_write_summary(out_root, metadata_paths, main_merge_path)`

作用：

- 遍历各个文化域的 metadata。
- 统计：
  - 样本数
  - 艺术家数
  - 时长最小值、中位数、最大值
  - 来源集合
- 写出 `summary_v3_main.json`

### `build_research_dataset_v3(...)`

这是整个文件最重要的总入口。

参数说明：

- `out_root`
  - V3 数据集最终输出目录。

- `raw_root`
  - 原始压缩包、网页缓存等所在目录。

- `cache_root`
  - 中间缓存目录。

- `fma_per_country`
  - 西方国家域每个国家的目标样本数。

- `turkey_target`
  - Turkey 域目标数。

- `anglo_pop_target`
  - 现代英语流行域目标数。

- `workers`
  - 并发线程数，主要给 FMA 下载逻辑使用。

- `china_jingju_target`
  - 中国京剧部分目标数。

- `china_opencpop_target`
  - 中国 OpenCpop 部分目标数。

- `opencpop_password`
  - OpenCpop 压缩包密码。

函数内部做的事：

1. 设置随机种子。
2. 构建每个文化域。
3. 选出主域列表 `main_paths`。
4. 用 `merge_metadata_dedup` 合并得到 `metadata_v3_main.csv`。
5. 用 `_write_summary` 生成 `summary_v3_main.json`。
6. 返回构建结果字典。

它的返回值包含：

- 主 metadata 路径
- merge report
- summary 路径
- Indonesia 主域路径
- Indonesia probe 路径
- 各个域的 metadata 路径列表

### `main()`

作用：

- 提供命令行入口。

支持传入的参数基本就是 `build_research_dataset_v3(...)` 的对应版本。

## 10. 这个文件最核心的设计思想

### 9.1 按文化域定制，而不是一刀切导入

这个脚本没有试图用一个通用导入器解决所有来源，而是承认现实世界的数据源异构性很强，因此用“每个文化域一套策略”的方式做构建。

### 9.2 先保证每个域可用，再谈全局统一

每个域先各自落成：

- 本地音频
- 本地 metadata

之后才统一 merge。

### 9.3 时长门槛是全局硬规则

整个脚本反复使用 `duration >= 30s` 作为保留条件，说明项目明确排除了太短的样本，以减少表征和推荐任务中的不稳定性。

### 9.4 多样性控制是工程显式目标

不是下载越多越好，而是要：

- 控制每域规模
- 防止艺术家过度集中
- 防止某些来源支配整个文化域

### 9.5 构建结果不是一个 CSV，而是一套目录化资产

最终产物不是“一个表”，而是：

- 每个文化域的本地音频目录
- 每个文化域的 metadata
- 全局合并表
- 全局汇总报告

## 11. 面试或讲解时最值得强调的点

- 这不是简单的数据下载脚本，而是一个“多来源、多文化域研究数据集构建器”。
- 它把每个文化域拆成明确的数据工程子任务。
- 它显式做了时长过滤、多样性控制、来源筛选和本地资产落盘。
- 它允许一个文化域由多个子来源拼接而成，比如 China 和 Indonesia。
- 它最终把多源异构音乐数据整理成后续 V4 继续标准化、交互合成、embedding 构建的种子数据集。

## 12. 建议你读这个文件时的顺序

1. 先看 `build_research_dataset_v3(...)`
2. 再看 `_build_china`、`_build_indonesia`、`_build_fma_western`
3. 然后看 `_build_fma_selected_targets` 和 `_download_fma_rows`
4. 最后再补看辅助函数

如果你这样读，会比从第一行顺着读到底更容易抓住主线。
