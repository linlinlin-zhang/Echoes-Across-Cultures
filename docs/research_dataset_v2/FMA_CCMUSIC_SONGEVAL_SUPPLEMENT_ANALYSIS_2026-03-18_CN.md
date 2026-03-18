# FMA, CCMusic Music Genre Dataset, SongEval 补充分析

更新日期: 2026-03-18

## 1. 目标

本报告回答三个问题:

1. 是否可以在不下载 FMA 全量音频的前提下, 先基于 metadata 判断其国家分布可行性。
2. 在同样的思路下, `CCMusic Music Genre Dataset` 和 `SongEval` 是否适合作为非西方音乐补充来源。
3. 如果目标是构建“国家层次的文化域”, 三者分别更适合扮演什么角色。

这里的判断标准不是“数据集本身是否优秀”, 而是“是否适合进入我们当前这条 `metadata -> audio -> embedding -> culture/domain` 数据构建主线”。

## 2. 方法

### 2.1 FMA

- 使用本地已下载的 `fma_metadata.zip`, 不下载音频。
- 先看全量 `artist.location` / `artist.latitude` / `artist.longitude` 覆盖率。
- 再按严格规则去掉以下流派:
  - `Experimental`
  - `Electronic`
  - `Rock`
  - `Novelty`
  - `International`
  - `Spoken`
  - `Hip-Hop`
- 流派过滤使用 `raw_tracks.csv` 的 `track_genres` 与 `genres.csv` 的谱系关系:
  - 只要原始 genre 或其 top-level genre 命中上述任一流派族, 就剔除。
- 国家归属分两层估计:
  - 高置信: `lat/lon -> country`
  - 扩展覆盖: 对 `artist.location` 做规则化国家映射

### 2.2 CCMusic Music Genre Dataset / SongEval

- 只看公开 dataset card / README / viewer / metadata 结构。
- 核查以下字段:
  - 是否有真实音频
  - 是否有 `artist / country / region / language / instrument` 等文化线索
  - 许可是否允许研究用集成
  - 其任务目标是否与“文化域建库”一致

## 3. FMA 结论

### 3.1 不下载音频也足以先做国家可行性分析

结论是“可以, 而且应该先这样做”。

FMA 全量 metadata 审计结果:

| 指标 | 数值 |
| --- | ---: |
| 全量 tracks | 106,574 |
| 非空 `artist.location` 的 tracks | 70,210 |
| 有 `lat/lon` 的 tracks | 44,544 |
| unique artists | 16,341 |
| 非空 `artist.location` 的 unique artists | 6,033 |
| 有 `lat/lon` 的 unique artists | 3,836 |

这意味着:

- 还没下载任何音频, 就已经可以先做国家覆盖率审计。
- `location` 覆盖率远高于我之前的保守估计。
- 真正应当先做的是“metadata 清洗和国家 shortlist”, 而不是“先把几十 GB 音频都下下来再看”。

### 3.2 严格去流派后, 仍有可用规模

严格流派过滤后:

| 指标 | 数值 |
| --- | ---: |
| 保留 tracks | 17,125 |
| 其中非空 `artist.location` | 10,783 |

这说明:

- 如果我们的目标只是“每国几百条即可”, FMA 仍然有价值。
- 但它已经不是“什么国家都能轻松补”; 经过严格过滤后, 规模明显收缩。

### 3.3 国家层面的可用性

在严格过滤后的 FMA 上:

- 只看高置信 `lat/lon`, 仍有 `7` 个国家达到 `100+` tracks, `2` 个国家达到 `300+` tracks。
- 再加规则化 `location` 文本后, 约 `9,905 / 10,783` 条可映射到国家。

按 `tracks >= 100` 且 `unique artists >= 20` 的稳健门槛, 得到:

| 国家 | tracks | unique artists |
| --- | ---: | ---: |
| US | 5,438 | 764 |
| GB | 951 | 109 |
| AU | 824 | 121 |
| CA | 558 | 92 |
| FR | 371 | 45 |
| DE | 277 | 35 |
| RU | 242 | 26 |
| NL | 151 | 22 |
| IT | 150 | 20 |

如果把门槛提高到 `tracks >= 200` 且 `artists >= 30`, 还能稳定保留:

- US
- GB
- AU
- CA
- FR
- DE

### 3.4 FMA 对非西方补充的局限

在同样的严格过滤和规则化映射下, 若看几个我们关心的非西方或非主流西方国家:

| 国家 | tracks | unique artists |
| --- | ---: | ---: |
| CN | 10 | 8 |
| IN | 0 | 0 |
| TR | 0 | 0 |
| JP | 47 | 13 |
| RU | 242 | 26 |

因此:

- FMA 非常适合做西方国家地理基座。
- FMA 不适合单独承担中国 / 印度 / 土耳其这类非西方文化域补充。
- “用 FMA 做国家层次文化域”是可行的, 但它天然偏欧美。

### 3.5 对 FMA 的最终判断

FMA 适合:

- 做 metadata-only 国家 shortlist
- 做西方国家基座
- 做国家层次文化域中的“地理弱标签起点”

FMA 不适合:

- 单独解决非西方文化覆盖
- 直接把 `artist.location` 当作高质量文化真值

## 4. CCMusic Music Genre Dataset 分析

### 4.1 数据形态

根据 dataset card:

- 数据来源为 NetEase 音乐。
- 默认描述为约 `1,700` 首 `mp3`, 每首约 `270-300s`, `22,050 Hz`。
- genre 标签来自网站原始风格标签。
- 数据卡还写到 “Most are English songs”。

但更关键的是:

- README 明确写了: `Due to copyright issues with the original music, only spectrograms are provided in the dataset.`
- 当前公开可见的 `eval` 子集字段只有:
  - `mel`
  - `cqt`
  - `chroma`
  - `fst_level_label`
  - `sec_level_label`
  - `thr_level_label`

公开 `eval` 子集规模为:

| split | examples |
| --- | ---: |
| train | 29,100 |
| validation | 3,637 |
| test | 3,638 |

这些更像是切分后的频谱图样本数, 不是可直接接入我们音频 embedding 管线的“独立曲目数”。

### 4.2 对我们任务的意义

这个数据集的问题不在于“质量差”, 而在于“用途不对”:

1. 没有公开真实音频, 只有频谱图子集可稳定访问。
2. 没有公开 `artist / country / region / location` 元数据。
3. genre 体系本身偏主流现代流行分类, 而不是文化域标签。
4. README 明确提到样本大多是英文歌曲, 对“补非西方音乐”帮助有限。
5. README 中 “16 genres” 与 “17 genres” 的表述存在不一致, 进一步说明它更像任务导向数据卡, 不是为文化档案整理设计的结构化资源。

### 4.3 结论

`CCMusic Music Genre Dataset` 不适合作为我们“非西方文化域补充”的主数据源。

它更适合作为:

- 音乐 genre 分类任务 benchmark
- 多层级 genre label 实验数据
- 频谱图分类的下游测试集

它不适合作为:

- 国家层次文化域建库源
- 真实音频 embedding 主数据源
- 非西方音乐补充主力

### 4.4 更具体地说: 它不适合直接当“中国音乐补充主源”

如果更精确地回答“用于补充中国音乐如何”, 结论是:

- 可以做很弱的辅助。
- 不适合做 `china` 文化域的主数据源。

原因有四个:

1. 官方 CCM 页面虽然把它放在中国音乐数据库体系里, 但这个子集本身是从网易云收集的流派分类数据, 任务定义是 genre classification, 不是中国音乐建库。官方说明写到:
   - 至少 `1700` 首音频
   - 标注格式为 `file_name, duration, singer, fst_level_label, sec_level_label, thr_level_label`
   - 主标签体系是 `classical / non-classical`, 细分为 `pop / dance&house / indie / soul/R&B / rock` 等
2. 官方页面给出的示例歌手几乎直接说明它的语义中心不是“中国音乐”, 例如:
   - `A Fine Frenzy`
   - `Daniel Powter`
   - `R.E.M.`
   - `Black Strobe`
3. Hugging Face 公开版明确写了:
   - `Due to copyright issues with the original music, only spectrograms are provided in the dataset.`
   - `Most are English songs`
   所以它既不适合作为中国音乐真实音频补充, 也不适合作为我们现有 embedding 管线的主音频源。
4. 它缺少国家/地区/文化传统层面的结构化字段。即使原始标注里有 `singer`, 也仍然需要额外做:
   - 歌手国别识别
   - 语言识别
   - 华语 / 非华语过滤
   - 中国流行 / 泛西方流行区分
   这会把它变成一个需要重做清洗的大杂烩源。

因此, 如果非要使用它, 最合理的定位是:

- 只作为“现代华语流行层”的候选补充源
- 先做严格二次筛选, 只保留明确华语/中文歌手样本
- 不直接并入当前 `china` 主文化域

更合适的做法仍然是:

- 用 `CTIS` / `ErhuPT` / 民族器乐相关子集定义中国传统器乐域
- 如果要补“现代中国音乐”, 单独再建一个 `mandarin_pop` 或 `cn_pop` 子域
- 不要把这个 genre 数据集直接当成“中国音乐”本体

## 5. SongEval 分析

### 5.1 数据形态

根据 dataset card / README:

- `SongEval` 是“完整歌曲美学评价” benchmark。
- 公开宣称包含 `2,399` 首完整歌曲, 约 `140` 小时音频。
- 包含 `English and Chinese` songs。
- 覆盖 `9 mainstream genres`。
- 标注目标是五个审美维度:
  - Coherence
  - Musicality
  - Memorability
  - Clarity
  - Naturalness
- README 明确写到:
  - 数据中包含 `five generation models` 的输出
  - 再加上一部分 `real / bad-case samples`

公开 metadata 结构非常简单, viewer 可见字段只有:

- `audio`
- `gender`
- `annotation`

`metadata.jsonl` 预览中未见:

- `artist`
- `country`
- `region`
- `language`
- `genre`

至少在当前公开 schema 下, 这些文化域建库最关键的字段并没有随数据一并暴露出来。

### 5.2 对我们任务的意义

`SongEval` 的主任务是“评估歌曲审美质量”, 而不是“提供文化来源清晰、可分国别的音乐资源”。

它的主要问题是:

1. 数据目标偏向美学评价, 不是文化域组织。
2. 数据中显式包含生成模型输出, 会引入明显的生成分布偏差。
3. 当前公开 metadata 没有足够的国家/文化字段。
4. README 虽然说有英文和中文、9 个主流 genre, 但这些信息没有在公开逐条 metadata 中体现出来。
5. 即使可用, 它更接近“现代流行歌曲评价集”, 而不是“非西方文化音乐档案”。

### 5.3 结论

`SongEval` 不适合作为我们补充非西方文化域的主数据源。

它更适合作为:

- 后续做“生成歌曲审美评价”或“自动质量预测”的辅助 benchmark
- 现代中英歌曲的审美打分研究

它不适合作为:

- 国家层次文化域建库源
- 非西方传统音乐补充源
- 文化域真实分布建模的主训练数据

## 6. CCMusic 体系里真正更有价值的补充: CTIS

虽然本轮用户点名的是 `CCMusic Music Genre Dataset`, 但从“非西方音乐补充”角度看, `CCMusic` 体系里更值得继续投入的是 `CTIS`。

`CTIS` 的公开数据卡显示:

- `4,956` 条录音
- `32.63` 小时
- `209` 类中国传统乐器
- 由于变体拆分, 最终 `219` 个 labels
- 数据涵盖中国传统乐器、改良乐器、以及少数民族乐器
- 当前公开字段包括:
  - `audio`
  - `mel`
  - `label`
  - `cname`
  - `pinyin`

这比 `music_genre` 更贴近我们目标, 因为:

1. 它有真实音频。
2. 它的文化语义是清晰的“中文传统器乐”。
3. 它天然属于非西方音乐补充。
4. 它与我们当前项目的 `china` 域已经兼容。

仓库内已有本地链路证据:

- 当前项目已经把 `ccmusic-database/CTIS` 作为 `china` 域来源之一。
- 本地 `import_report.json` 显示我们已经成功导入 `250` 条样本做试运行。
- 本地 `china/metadata.csv` 当前有 `251` 行, 即 `250` 条样本加表头。

但也要看到它的边界:

- 它不是“按国家 location 分桶”的数据。
- 它没有 `artist / region / country` 级元数据来支持国家层次建模。
- 它更适合定义一个“instrument-centered Chinese traditional domain”, 而不是国家内多亚文化分层。

因此:

- 若目标是“补非西方音乐”, `CTIS` 是好的补充。
- 若目标是“按国家 location 统一做层次文化域”, `CTIS` 和 FMA 不是同一种资源, 需要在 ontology 层面做融合。

## 7. 综合结论

### 7.1 哪些能进入主线

适合进入当前文化域建库主线:

- `FMA`
  - 适合做 metadata-only 国家审计
  - 适合做西方国家基座
- `CTIS`
  - 适合补充中国传统器乐这一非西方文化域

不适合进入当前主线:

- `CCMusic Music Genre Dataset`
  - 更像频谱图分类任务集
- `SongEval`
  - 更像完整歌曲美学评价 benchmark

### 7.2 当前最现实的组合

最现实的数据策略是:

1. 用 `FMA` 做国家层面的西方基座。
2. 用 `CTIS` 补中国传统器乐域。
3. 后续再补印度、土耳其、哈萨克斯坦、韩国等具有明确文化语义的真实音频源。
4. 不把 `music_genre` 和 `SongEval` 当作非西方文化补充主源。

### 7.3 最直接的下一步

建议立刻做三件事:

1. 把 FMA 的国家清洗和 shortlist 产成正式 `metadata audit` 文件。
2. 继续沿 `CTIS -> china domain` 这条线扩充真实非西方样本。
3. 用同样标准继续审计:
   - 印度传统音乐源
   - 土耳其 makam / folk 源
   - 韩国传统 / 国乐源
   - 中亚音乐源

## 8. Sources

- FMA metadata archive: <https://os.unil.cloud.switch.ch/fma/fma_metadata.zip>
- FMA GitHub README: <https://github.com/mdeff/fma>
- CCMusic Music Genre Dataset: <https://huggingface.co/datasets/ccmusic-database/music_genre>
- CCMusic CTIS: <https://huggingface.co/datasets/ccmusic-database/CTIS>
- CTIS project page: <https://ccmusic-database.github.io/en/database/ctis.html>
- SongEval dataset card: <https://huggingface.co/datasets/ASLP-lab/SongEval>
- SongEval GitHub: <https://github.com/ASLP-lab/SongEval>
- SongEval paper: <https://arxiv.org/pdf/2505.10793>
