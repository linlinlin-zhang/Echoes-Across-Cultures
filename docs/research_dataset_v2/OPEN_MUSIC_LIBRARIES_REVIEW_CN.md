# 开源音乐库补充文化域的可行性整理

更新时间：2026-03-17

## 1. 我们现在到底在找什么

如果只是“找得到很多音乐”，那么公开音乐库并不少。

但对当前项目来说，我们真正要找的是：

- 原始音频可获取
- 许可清楚，最好可研究使用
- 可批量下载或至少可脚本化访问
- 元数据够稳定
- 最好还能清楚划成某个文化域

这五个条件同时满足时，候选源会少很多。

## 2. Free Music Archive (FMA) 值不值得用

### 2.1 FMA 的优点

FMA 是 MIR 历史上最经典的开放音频库之一。

官方与论文资料显示：

- 公开可得规模很大
- 有完整音频、元数据和预计算特征
- MIR 社区长期在使用

关键来源：

- FMA 官方数据仓库：
  <https://github.com/mdeff/fma>
- FMA 论文：
  <https://arxiv.org/abs/1612.01840>

根据 FMA 论文和官方仓库：

- 总量约 `106,574` 首
- 提供 `tracks.csv / genres.csv / features.csv`
- 有 `small / medium / large / full` 多个子集
- 可下载 30 秒切片版和 full-length 版

这是它最大的优点：

- **大**
- **开放**
- **MIR 社区认可**

### 2.2 FMA 的问题

FMA 很适合做：

- genre classification
- representation learning
- auto-tagging
- 通用音乐检索 baseline

但它不天然适合当前这种“国家/文化域”数据构建。

核心原因有三点：

#### A. 它主要是按 genre/artist/track 组织，不是按文化域组织

FMA 的主结构是：

- genre
- artist
- track
- tags

而不是：

- country
- tradition
- repertoire

所以如果我们想从 FMA 里构造：

- 德国域
- 日本域
- 韩国域

就得额外依赖：

- artist biography
- artist location
- free-form tags

这些字段的噪声会明显更大。

#### B. location 不是强约束的标准文化字段

FMA 站点文档里可以看到 artist profile 里有 `location` 这种信息，但它更像站点资料字段，而不是一个为研究准备的严格国家标签。

来源：

- FMA Help / Artist Guide 搜索结果显示 artist profile 可编辑 `location`：
  <https://freemusicarchive.org/index.php/Help>
  <https://freemusicarchive.org/artist_guide>

因此如果我们真的要按国家切分，往往需要再做：

- location 正规化
- 人工检查
- 曲风过滤

工作量并不低。

#### C. 文化辨识度通常不如传统音乐语料强

FMA 里的音乐很多是：

- indie
- electronic
- alternative
- rock
- experimental
- singer-songwriter

这些当然也有文化性，但不像传统民间音乐、古典传统音乐那样“文化痕迹特别尖锐”。

所以如果我们的目标是：

- 强跨文化差异
- 明确文化域
- 验证 DCAS 的跨文化结构能力

FMA 通常不如现在这类传统/民间/档案型语料“干净”。

### 2.3 对我们项目的判断

FMA **可以用**，但更适合作为：

- 现代开放音乐补充库
- Anglo-pop / open popular music 的替代或扩展来源
- exploratory domain 的来源

不太适合直接当：

- 高文化纯度的主实验国家域

一句话：

- **FMA 适合补现代开放音乐**
- **不适合直接替代“文化域明确”的传统音乐库**

## 3. MTG-Jamendo 值不值得用

### 3.1 优点

Jamendo/MTG-Jamendo 是我们现在已经在用的现代音乐开放源，优点非常明确：

- 规模大
- 标签化比较好
- 下载与复现链路成熟
- 研究社区使用广

官方说明：

- 数据集约 `55,000+` 首
- 有 `195` 个 tag
- 元数据和下载脚本都提供

来源：

- <https://mtg.github.io/mtg-jamendo-dataset/>

### 3.2 局限

和 FMA 类似，它也不是天然国家文化域库。

它更适合：

- 现代流行 / open commercial-like music
- tag-based subset construction

所以它很适合做：

- `anglo_pop`

但不适合直接拿来造：

- Germany folk
- Korea traditional
- Japan traditional

## 4. Internet Archive 值不值得用

### 4.1 优点

Internet Archive 的音频非常多，历史档案、现场录音、民间录音、78 转唱片等资源极其丰富。

如果从“世界上有没有大量音乐”来看，Internet Archive 的答案通常是有。

### 4.2 核心问题

但它对我们项目有一个很致命的问题：

- **rights 不是统一保证的**

官方帮助页明确写了：

- Internet Archive **不保证** 条目的版权状态
- 用户需要自行确保使用不侵权
- rights 信息往往由上传者填写

来源：

- <https://archivesupport.zendesk.com/hc/en-us/articles/360014759692-Rights>

这意味着：

- 用它做 exploratory search 可以
- 但如果要把它直接作为“最终高质量主域”的大规模主源，风险很高

一句话：

- **Internet Archive 是巨大资源池**
- **但不是统一合规、统一质量的研究主源**

## 5. Dunya / CompMusic 值不值得用

### 5.1 优点

这是目前最像“真正文化域研究库”的一类资源。

CompMusic 官方 corpora 页面和 Dunya 开发者文档都非常清楚：

- Arab-Andalusian
- Carnatic
- Hindustani
- Makam 等

都是强文化域。

来源：

- <https://compmusic.upf.edu/corpora>
- <https://dunya.compmusic.upf.edu/developers/>

对于 Arab-Andalusian，官方页面写了：

- `338 recordings`
- `112 hours`

来源：

- <https://dunya.compmusic.upf.edu/andalusian/info>

### 5.2 问题

它的问题不是文化不强，而是：

- 访问更像平台/API
- 实际下载需要 token
- 不是匿名直接开放整包

所以它非常适合：

- 做文化域研究扩展

但工程推进上通常比 FMA/Jamendo 慢。

## 6. 结论：开源音乐库应该怎么用

### 最适合拿来做主域补充的

- `Jamendo`：补现代开放流行锚点
- `FMA`：补现代开放音乐、做 exploratory 现代域

### 最适合拿来做强文化扩展域的

- `Dunya / CompMusic`

### 最适合拿来做资源检索池，但不建议直接做主源的

- `Internet Archive`

## 7. 对当前项目的建议

如果现在想“从开放音乐库里再补东西”，最现实的策略不是把 FMA 当某个国家文化域，而是：

### 路线 A

把 FMA 当作：

- `open_pop`
- `indie_open_music`
- `modern_open_music`

这类现代开放音乐域的补充池。

### 路线 B

用 FMA 只做：

- exploratory comparison
- supplementary modern-domain analysis

而不要动现在已经很清楚的主域结构。

### 路线 C

如果真的想补“文化域”而不是“开放音乐库”，仍然优先：

- 国家图书馆/档案馆
- 文化机构 OpenAPI
- Dunya / CompMusic

而不是先去 FMA 里硬切国家。

## 8. 一句话总结

`Free Music Archive` 很有价值，但它更像：

- **开放现代音乐库**

而不是：

- **现成的国家文化域数据集**

所以对我们当前项目，FMA 值得用，但更适合做：

- 现代开放音乐补充源
- 或 exploratory/对照语料

而不是直接取代现在这套文化域主版。
