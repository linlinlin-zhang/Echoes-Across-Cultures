# MTG-Jamendo 非西方音乐补充可行性审计（2026-03-18）

## 1. 审计目标

本轮审计沿用此前处理 FMA 的思路，但重点改为：

- 不先下载大体量音频
- 先用官方 metadata 判断这个数据集能否补充更多非西方音乐
- 重点看：
  - 是否存在 `country / nationality / location / language` 这类可直接做国家或文化域分类的字段
  - 若没有，能否依靠 `tags + artist/title 文本信号` 做一轮弱筛选
  - 最终能否形成一个值得人工复核的非西方候选池

---

## 2. 官方数据结构结论

根据官方 README，MTG-Jamendo 是一个面向 music auto-tagging 的开放数据集，包含：

- `55,000+` 首完整音频
- `195` 个 tags
- tags 来自 `genre / instrument / mood-theme`

官方还明确说明：

- `raw.meta.tsv` 只提供额外的：
  - `artist`
  - `album name`
  - `track title`
  - `release date`
  - `track URL`
- 并没有 `country`
- 并没有 `nationality`
- 并没有显式 `language`

因此，和 FMA 不同，MTG-Jamendo **不能** 走“先按 location / nationality 分国家”的直线路线。

这意味着：

- 它不适合直接构建国家层次文化域
- 它最多适合作为一个“弱标签候选池”

---

## 3. 本地 metadata-only 审计结果

本地审计基于官方仓库：

- `data/raw_30s_cleantags.tsv`
- `data/raw_30s_cleantags_50artists.tsv`
- `data/raw.meta.tsv`

说明：

- `raw_30s_cleantags.tsv` 是过滤到 `>30s` 后、做过 tag merge 的主 metadata 文件
- `raw_30s_cleantags_50artists.tsv` 是再经过“每个 tag 至少 50 个 unique artists”过滤后的版本
- Windows 下 `data/autotagging.tsv` 是符号链接占位文本，实际应读取 `raw_30s_cleantags_50artists.tsv`

---

## 4. 直接字段层面：不能像 FMA 一样做国家筛选

`raw.meta.tsv` 的列是：

- `TRACK_ID`
- `ARTIST_ID`
- `ALBUM_ID`
- `TRACK_NAME`
- `ARTIST_NAME`
- `ALBUM_NAME`
- `RELEASEDATE`
- `URL`

没有：

- `location`
- `country`
- `language`
- `region`

因此：

- 不能像 FMA 一样按 `artist.location -> country` 建国家池
- 也不能像 OpenCpop 那样直接用 `language`
- 只能退而求其次，使用：
  - tags
  - 非拉丁字符脚本
  - 曲名/艺名中的地理或文化线索

---

## 5. 50-artist 过滤后：可形成一个小规模弱候选池

在 `raw_30s_cleantags_50artists.tsv`（`55,609` tracks）里，出现的直接相关 genre tags 只有：

- `genre---ethno`: `206`
- `genre---latin`: `233`
- `genre---world`: `119`
- `genre---african`: `111`
- `genre---ethnicrock`: `94`
- `genre---oriental`: `24`

这些 tag 的并集规模为：

- `787` tracks
- `221` artists

另外，如果只看 `artist/title` 中出现的更强非西方脚本信号：

- `han`: `18` tracks / `7` artists
- `kana`: `2` tracks / `2` artists
- `hangul`: `2` tracks / `1` artist
- `arabic`: `1` track / `1` artist
- `devanagari`: `1` track / `1` artist

把这些更强脚本信号合并后，只有：

- `22` tracks
- `9` artists

更值得注意的是：

- tag 候选池与强脚本候选池 **几乎不重合**
- 二者 union 只有：
  - `809` tracks
  - `230` artists

这说明：

- 这个数据集里的“非西方性”更多体现在粗粒度 world/ethno 标签
- 而不是稳定的语言、国家、民族、地区元数据

---

## 6. 50-artist 过滤前：非西方标签会稍微多一点，但仍然不深

在 `raw_30s_cleantags.tsv`（过滤到 `>30s`，但未做 50-artist 过滤）中，能找到更多弱标签：

- `genre---asian`: `88`
- `genre---middleeastern`: `14`
- `genre---oriental`: `21`
- `genre---african`: `111`
- `genre---world`: `109`
- `genre---ethno`: `175`
- `genre---ethnicrock`: `91`

如果把这些合成一个更接近“亚洲/中东/非洲/广义 world”的严格弱候选池，其并集约为：

- `611` tracks
- `187` artists

如果再把更宽泛的 `latin / flamenco / bossanova / reggaeton / samba / tango` 也算进去，则可扩到：

- `1026` tracks
- `286` artists

这说明：

- 在“广义非西方 / world-ish”意义上，MTG-Jamendo **确实有一定补充价值**
- 但在“明确中国 / 印度 / 中东 / 土耳其 / 韩国 / 日本”等可解释文化域意义上，信号仍然偏弱

---

## 7. 稀有但更有价值的标签非常少

过滤前数据里，我额外搜索了更直接的非西方语义标签或器乐标签，结果非常少：

- `instrument---oud`: `1`
- `mood/theme---indian`: `1`

没有在主 metadata 中形成有效规模的：

- `sitar`
- `tabla`
- `erhu`
- `guzheng`
- `pipa`
- `koto`
- `shamisen`

这说明：

- 这个数据集并不是一个“器乐文化域资源库”
- 它也不是一个“国家或地区音乐数据库”
- 它本质上仍然是一个面向 auto-tagging 的通用流派/风格/情绪标签集

---

## 8. 与 FMA 的关键差异

FMA 的优势在于：

- 有 `artist.location`
- 可以做 `metadata-only -> country normalization`
- 有机会构造国家层次文化域

MTG-Jamendo 的情况则是：

- 没有可直接地理化的字段
- 标签体系是 `genre / instrument / mood-theme`
- 可以得到一个“弱非西方风格候选池”
- 但很难得到国家层次文化域

因此：

- FMA 更适合做国家池和文化层次结构
- MTG-Jamendo 更适合做：
  - `anglo_pop / west` 锚点
  - 或少量 `world/ethno/african/asian/middleeastern` 手工补充候选池

---

## 9. 下载成本视角

官方 README 给出的 `raw_30s/audio` 大小约为：

- 全质量：`508 GB`
- 低质量：`156 GB`

所以如果只针对上面的弱候选池下载，理论上规模仍可控：

- 严格非西方弱标签池 `611 tracks`
  - 约占全体 `1.1%`
  - 粗略折算：
    - 全质量约 `5.6 GB`
    - 低质量约 `1.7 GB`
- 宽泛非西方池 `1026 tracks`
  - 粗略折算：
    - 全质量约 `9.4 GB`
    - 低质量约 `2.9 GB`

所以：

- metadata-only 先筛，再只下候选子集，这条路线是划算的

---

## 10. 最终判断

### 10.1 能不能从 MTG-Jamendo 得到更多非西方音乐？

能，但只能得到一个 **弱监督候选池**，不能像 FMA 那样得到稳定国家池。

### 10.2 值不值得继续？

值得继续做一层人工复核，但不值得把它当成非西方文化域的主来源。

更准确地说：

- 如果目标是补一些 `world / ethno / african / asian / middleeastern / latin` 候选样本：
  - 值得
- 如果目标是补 `china / india / turkey / arabic` 这种可解释文化域：
  - 不够稳

### 10.3 适合怎么用？

最合理的用法是：

- 不把 MTG-Jamendo 当作国家层次主源
- 把它当作一个 secondary source
- 先基于 tags 做 metadata-only 弱筛选
- 再对筛出的 `600-1000` 条候选做人工复核
- 最后只保留高置信非西方样本并补充进你的主数据集

---

## 11. 建议的下一步

如果继续推进，建议顺序如下：

1. 从 `raw_30s_cleantags.tsv` 提取严格弱非西方候选池  
   规则可先用：
   - `genre---asian`
   - `genre---middleeastern`
   - `genre---oriental`
   - `genre---african`
   - `genre---world`
   - `genre---ethno`
   - `genre---ethnicrock`

2. 单独保留更宽泛的 `latin/flamenco/bossanova` 池  
   不要和亚洲/中东池混成同一个结论

3. 用 `artist + title + tag` 做二次人工清洗  
   因为这里没有国家字段，必须有人审

4. 只下载人工通过的子集音频  
   没必要下全量 `508 GB`

---

## 12. 一句话结论

MTG-Jamendo **可以补一点非西方音乐，但补的是一个小而弱的 world/ethno 候选池，不是一个能直接拿来构建国家层次文化域的非西方主数据源。**

---

## Sources

- 官方站点：<https://mtg.github.io/mtg-jamendo-dataset/>
- 官方仓库 README：<https://github.com/MTG/mtg-jamendo-dataset>
- 官方 metadata：
  - <https://github.com/MTG/mtg-jamendo-dataset/blob/master/data/raw.meta.tsv>
  - <https://github.com/MTG/mtg-jamendo-dataset/blob/master/data/raw_30s_cleantags.tsv>
  - <https://github.com/MTG/mtg-jamendo-dataset/blob/master/data/raw_30s_cleantags_50artists.tsv>
