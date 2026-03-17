# Research Dataset V2 第三轮全域深度检索与主域可行性判断

更新时间：2026-03-14  
目的：在完成多轮来源与许可筛查后，结合工程可执行性与外国语学院协同需求，正式判断哪些文化域适合作为 `research_dataset_v2` 的主域。

---

## 1. 这轮检索关注的核心问题

这一轮不再只是问“某个国家有没有音乐数据”，而是问：

**哪些文化域现在真正具备进入 v2 主域的条件。**

判断标准只有四条：

1. 有真实音频  
2. 许可或研究使用边界足够清楚  
3. 样本规模至少有机会支撑 `100-200` 的 v2-lite 目标  
4. 能进入现有 `audio -> metadata -> embedding` 流水线

---

## 2. 当前主域决策总览

在兼顾数据可行性、论文跨文化叙事以及你后续可能挂靠外国语学院的现实需求之后，当前最合适的主域方案不是 France，而是：

- `china`
- `germany`
- `japan`
- `india`
- `turkey`
- `anglo_pop`

这 6 个域里：

- `china`：最稳，ready
- `india`：最稳的新主域，ready
- `anglo_pop`：可推进，但需过滤
- `germany`：有希望，通过 item-level 审计后可用
- `japan`：放宽到 `japanese_music_audio` 后有望可用
- `turkey`：主域保留，但需继续确认 license

---

## 3. 当前最强、最可直接推进的主域

## 3.1 China

主来源：
- `ccmusic-database/CTIS`

来源：
- <https://huggingface.co/datasets/ccmusic-database/CTIS>
- <https://ccmusic-database.github.io/en/database/ctis.html>

为什么强：
- 明确是中国传统器乐音频
- 样本规模足够
- 许可清楚：`CC-BY-NC-ND-4.0`
- 已经能进入现有工具链

结论：
- `ready`

---

## 3.2 India

主来源：
- `neerajaabhyankar/hindustani-raag-small`

来源：
- <https://huggingface.co/datasets/neerajaabhyankar/hindustani-raag-small>

官方关键信息：
- Audio modality
- Rows: `1,253`
- License: `CC-BY-4.0`

为什么强：
- 音频真实可用
- 规模足够做 `100-200` 的 v2-lite 主域
- 文化定义清楚：Hindustani / raag
- 许可清楚

结论：
- `ready`

说明：
- 这是当前最值得纳入主域的新文化域之一

---

## 3.3 Anglo-pop

主来源：
- `MTG-Jamendo`

来源：
- <https://mtg.github.io/mtg-jamendo-dataset/>
- <https://huggingface.co/datasets/rkstgr/mtg-jamendo>

为什么强：
- 音频规模很大
- metadata 完整
- 适合作为现代流行锚点

当前问题：
- 不是天然的“英语流行纯净集合”
- 必须加：
  - `pop` 标签过滤
  - 语言启发式过滤

结论：
- `provisional but highly usable`

---

## 4. 可补到可用状态的主域

## 4.1 Germany

当前最有希望的线索：
- Europeana `Westphalian Folk Song and Sound Archive`

来源：
- <https://www.europeana.eu/es/collections/organisation/1815-westphalian-folk-song-and-sound-archive>

为什么值得保留：
- 和 `german_folk` 方向较一致
- 集合规模看起来有希望支撑 `100-200`
- 比之前那些几十条级别的小来源更像主来源候选

当前卡点：
- Europeana 是聚合平台
- 需要 item-level rights 审计
- 需要 item-level 音频可下载性审计

结论：
- `provisional`
- Germany 继续保留为主域
- 下一步应该做 50 条 item-level 审计

---

## 4.2 Japan

Japan 域本轮最重要的调整不是找到一个完美主来源，而是：

**把定义从严格的 `japanese_traditional` 放宽为 `japanese_music_audio`。**

这样做的原因是：
- 如果坚持纯传统日本音乐，当前公开音频主来源规模不足
- 放宽后，日本域才有现实机会补到可用

当前主候选：

### 候选 A：Historical Recordings Collection（National Diet Library）

来源线索：
- <https://www.ndl.go.jp/en/news/fy2020/210127_01>
- <https://sydney.jpf.go.jp/japanese-studies/resources/historical-recordings-collection/>
- <https://dl.ndl.go.jp/view/download/digidepo_9551749_po_NDL-Newsletter192_928.pdf?alternativeNo=&contentNo=1>

当前判断：
- 规模看起来足够大
- 文化上合理
- 但不是现成标准化音频数据集
- 还需要更细的 access / rights 审计

### 候选 B：`tts-dataset/japanese-singing-voice`

来源：
- <https://huggingface.co/datasets/tts-dataset/japanese-singing-voice>

当前判断：
- 规模足够
- License 清楚：`CC-BY-NC-4.0`
- 工程上可接入
- 但它不是“纯传统日本音乐”

结论：
- 如果 Japan 域定义放宽到 `japanese_music_audio`，Japan 可以补到可用状态
- 如果坚持纯传统定义，则当前仍偏弱

---

## 4.3 Turkey

Turkey 作为主域保留，主要理由不是当前最强，而是：

- 与旧版本跨文化路线连续
- 在方法和论文叙事中有意义
- 后续 PAL 与对比实验会更自然

当前主要候选：

### 候选 A：`bilal63/turkish_music_emotion_dataset`

来源：
- <https://huggingface.co/datasets/bilal63/turkish_music_emotion_dataset>

当前判断：
- 页面可见音频数量约 `400`
- 规模足够做小型平衡域
- 但 license 信息目前没有稳定、明确地暴露出来

### 候选 B：其他 Turkey 学术数据集

- `MTG Ottoman-Turkish Makam Recognition Dataset`
  - <https://zenodo.org/records/4883680>
  - 许可清楚，但不提供原始音频

- `UCI Turkish Music Emotion`
  - <https://archive.ics.uci.edu/dataset/862/turkish%2Bmusic%2Bemotion>
  - 许可清楚，但只提供特征，不是原始音频

结论：
- Turkey 当前状态：`provisional`
- 主域保留
- 下一步重点是继续补强原始音频许可证据

---

## 5. 当前不建议优先作为主域的文化域

## 5.1 Spain

如果你坚持 `flamenco`，文化定义非常好。  
但当前主要问题没有变：

- 最好的学术候选 COFLA 不是“开箱即脚本可导入”的开放音频主源

结论：
- 当前不建议作为 v2-lite 主域

---

## 5.2 France

这轮检索后，我没有找到一个在以下三点上同时强的法国候选：

- 音频可直接拿
- 许可清楚
- 规模足够

所以 France 目前没有显示出比 Germany / Turkey / India 更强的可行性。

结论：
- 当前不建议优先纳入主域

---

## 5.3 Kazakhstan

Kazakhstan 其实是很强的文化域候选：

- <https://huggingface.co/datasets/rtrk/kazakh-traditional-audio>
- 规模足够
- 许可清楚
- 音频真实可用

但在这轮决策里，它没有进入主域，不是因为不可行，而是因为：

- 你当前更重视 Germany / Japan 这两个与外院协同更强的域
- 主域数量已经扩到 6 个

结论：
- Kazakhstan 是非常好的 `expansion domain`
- 如果后续某个主域卡住，它是最优先顶替者之一

---

## 6. 关于主域规模目标

当前建议按 `v2-lite` 执行：

- 每域目标：`100-200`
- 六域总量目标：`800-1200`

这意味着：

- Germany / Japan 不必再硬追 `400+`
- 只要许可清楚、音频可得、文化定义稳定，它们就有机会进入主域

但仍然不建议完全放任“收集到多少就用多少”。  
建议控制：

- 理想：各域在 `±20%`
- 可接受：最大域不超过最小域 `2:1`

---

## 7. 一句话总结

在结合现实协同需求后的正式主域方案中，当前最合理的 6 域是：

**China + Germany + Japan + India + Turkey + Anglo-pop**

其中：

- China / India 已接近 ready
- Germany / Japan / Turkey 继续补来源与许可
- Anglo-pop 继续做过滤后进入主实验

这比继续围绕 France 或 Spain 做主域，更适合你当前项目的工程推进和论文落地。
