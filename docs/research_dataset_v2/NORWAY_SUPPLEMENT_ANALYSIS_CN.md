# Norway 域补量分析

更新时间：2026-03-17

## 1. 当前状态

当前本地 Norway 域来自：

- `Bots4M/HF2-Hardanger-fiddle-dataset`
- <https://huggingface.co/datasets/Bots4M/HF2-Hardanger-fiddle-dataset>

本地导入位置：

- [norway](/E:/Desktop/Echo/storage/public/research_dataset_v2/norway)

当前规模：

- 总条目数：`119`
- 独立标题数：`39`
- `has_emotional_variations = True` 的条目：`100`
- `has_emotional_variations = False` 的条目：`19`

因此，Norway 现在的真实问题不只是“数量少”，而是：

- **119 条里只有 39 首真正独立曲目**
- 大部分条目是同一批 Hardanger fiddle 曲目的情绪变体

这意味着，如果把 Norway 直接和当前 `250+` 规模的主域并列，它会显得偏弱。

## 2. 为什么不建议直接靠 HF2 自身把 Norway 补到 250+

根据 HF2 数据集卡与配套论文：

- 数据集本身就是 `39 unique songs, 119 audio-MIDI pairs`
- 其设计目的更偏 Hardanger fiddle transcription / emotion-variation research

来源：

- <https://huggingface.co/datasets/Bots4M/HF2-Hardanger-fiddle-dataset>
- <https://transactions.ismir.net/articles/10.5334/tismir.139>
- <https://zenodo.org/record/5624587>

因此，HF2 不适合被简单“扩写”为一个大规模挪威文化域。继续从 HF2 内部取样，只会重复增加同类变体，而不会真正提升语料多样性。

## 3. Norway 还能不能补

可以，但要走 **第二来源**。

目前最值得看的不是另一个现成 Hugging Face 数据集，而是：

- **National Library of Norway / Norsk folkemusikksamling**

关键公开信息：

- 国家图书馆 folk music 页面说明其音频收藏约有 `47,000` 个单独录音，约 `2000` 小时
- 其中“大部分是最古老类型的挪威民间音乐”，并包含大量 field recordings
- 页面说明：
  - many recordings are now in the public domain and can be listened to online
  - 也可以预约馆内 listening stations
  - 还可以 order digital copies from the audio archive in Rana

来源：

- <https://www.nb.no/samlingen/musikk/folkemusikk/>
- <https://www.nb.no/en/the-collection/the-music/>

此外，国家图书馆公开的总览 PDF 也表明其收藏规模远大于 HF2：

- `Oversikt-folkemusikk.pdf`
- <https://www.nb.no/content/uploads/2019/06/Oversikt-folkemusikk.pdf>

这说明：

- Norway 并不是“没有数据”
- 真正的问题是：**现成 ML-ready 开放数据集少，但馆藏规模其实很大**

## 4. Norway 补量的两条路线

### 路线 A：低风险路线

保持当前 HF2 Norway 作为：

- exploratory domain
- supplemental domain

优点：

- 已经在本地
- 许可清楚
- 工程稳定

缺点：

- 规模小
- 独立曲目少
- 不适合当和主域等量的核心域

### 路线 B：高价值路线

把 Norway 扩量建立在 National Library of Norway 的 folk archive 上。

可执行方式：

1. 从 `nb.no/samlingen/musikk/folkemusikk/` 入口人工确认可在线收听条目
2. 优先筛：
   - Hardanger fiddle
   - vanlig fele
   - vocal folk
   - gammaldans
3. 优先选择：
   - clearly public-domain / openly listenable items
   - 或国家图书馆允许研究访问 / 数字副本订购的条目
4. 单独形成 `norway_archive` 子域，再和当前 HF2 Norway 合并

优点：

- 可以真正把 Norway 补到更像一个成熟文化域
- 文化代表性会明显优于只用 HF2

缺点：

- 需要人工馆藏检索
- 可能涉及访问申请或数字副本获取
- 比现在现成数据源慢很多

## 5. 当前建议

如果目标是 **尽快冻结最终高质量数据集**：

- 不建议现在把 Norway 当作正式第 6 主域直接并入主实验
- 更稳的做法是：
  - 保留 Norway 为 `candidate / exploratory domain`
  - 等找到第二来源后再升格

如果目标是 **继续把 Norway 做强**：

- 最值得推进的就是 National Library of Norway 这条线
- 它比继续在 Hugging Face 上找小碎片数据更有希望

## 6. 结论

Norway 现在的问题不是简单的“119 条比 250 少”，而是：

- **条目数少**
- **独立曲目更少**
- **当前开放来源过于集中在 Hardanger fiddle 情绪变体**

所以，如果要认真补 Norway，正确方向不是继续挤 HF2，而是转向：

- **National Library of Norway folk archive**

在那之前，Norway 更适合作为：

- `exploratory domain`
- 或未来升级候选域

而不是立刻并入最终主实验域。
