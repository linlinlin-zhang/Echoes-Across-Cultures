# Research Dataset V2 来源与开源许可审计

更新时间：2026-03-13  
适用范围：`research_dataset_v2` 的来源筛选与许可审计。  
说明：本文件只覆盖“原始音频与元数据来源是否适合接入”的判断，不涉及后续 Gemini embedding 生成。

---

## 1. 这份审计文件要回答什么

这份文件回答 5 个问题：

1. 每个文化域目前有哪些候选公开来源  
2. 这些来源的音频是否真的可获取  
3. 这些来源的许可是否足够清楚  
4. 这些来源是否和目标文化域定义匹配  
5. 现在应该“采用 / 备选 / 暂缓 / 放弃”哪一个来源

本文件的使用原则是：

- 只记录可以追溯到公开页面的来源
- 尽量使用源站、模型卡、数据卡、Zenodo 页面等一手页面
- 如许可不清晰，则宁可标记为 `provisional` 或 `blocked`

---

## 2. 总体结论

截至当前阶段，5 个文化域的来源筛选结论如下：

- `china`：已经有一个可直接推进的主来源
- `anglo_pop`：已经有一个可推进的主来源，但还要做标签和语言过滤
- `germany`：目前找到的公开来源要么规模太小，要么不够形成 400+ 样本的稳定主域
- `spain`：目前最合适的 flamenco 候选在学术上很好，但音频不是即刻完全开放下载，因此暂缓
- `japan`：目前找到的“传统日本音乐”公开音频来源要么规模偏小，要么和域定义不完全一致，因此暂缓

这意味着：

- 现在最适合优先启动的域是：`china` 和 `anglo_pop`
- `germany / spain / japan` 在当前阶段还需要继续补强来源

所以，`v2` 这一步最现实的做法不是立刻批量导入全部 5 域，而是：

1. 先冻结域定义  
2. 先推进 `china` 和 `anglo_pop` 的导入测试  
3. 同时继续筛 `germany / spain / japan` 的最终主来源

---

## 3. 审计标准

### 3.1 文化匹配

来源是否与该文化域的音乐定义一致。

### 3.2 许可清晰度

是否有明确许可页面，能回答：

- 是否可用于研究
- 是否允许下载音频
- 是否允许后续发布 embedding 或 metadata

### 3.3 工程可接入性

是否能进入现有工具链：

- 单域 `audio/ + metadata.csv`
- 最终 `metadata_merged.csv`
- 后续统一生成 embedding

### 3.4 样本规模

是否能较稳定地支持：

- 每域 `400+` 样本的第一阶段目标

---

## 4. 逐域审计

## 4.1 China

### 主候选：`ccmusic-database/CTIS`

来源：
- Hugging Face 数据集卡：<https://huggingface.co/datasets/ccmusic-database/CTIS>

许可：
- 数据集卡显示 `License: cc-by-nc-nd-4.0`

一手证据：
- Hugging Face 页面显示 `License: cc-by-nc-nd-4.0`

适配性：
- 这是一个明确面向中国传统乐器音频的数据集
- 和 `chinese_traditional` 的目标定义是匹配的
- 规模足够，明显高于第一阶段 `400+` 的门槛

风险：
- 域内更偏“传统器乐/乐器识别”而非“曲目级文化语境”
- 需要在导入后检查是否存在某些乐器类别占比过高

结论：
- `ready`
- 可作为中国域主来源

---

## 4.2 Anglo Pop

### 主候选：MTG-Jamendo

来源：
- 官方数据页：<https://mtg.github.io/mtg-jamendo-dataset/>

许可：
- 代码：Apache 2.0
- metadata：CC BY-NC-SA 4.0
- audio：逐轨 Creative Commons，见 `audio_licenses.txt`
- 官方同时明确写了：仅供非商业研究与学术使用

一手证据：
- 官方页面 License 部分写明：
  - metadata 为 CC BY-NC-SA 4.0
  - audio 为逐轨 Creative Commons
  - 整体用于 non-commercial research and academic use

适配性：
- 这是一个规模大、工程上非常方便的音乐音频来源
- 适合用作现代流行/商业音乐锚点的候选池

风险：
- 它不是天然“英语流行音乐”数据集
- 必须后续做：
  - `pop` 标签过滤
  - 英语语言启发式筛选

结论：
- `provisional`
- 但可以作为 `anglo_pop` 当前最强主候选

说明：
- 当前可先将其作为“可推进主来源”
- 真正冻结前必须完成标签与语言筛选策略

### 备选：Free Music Archive (FMA)

来源：
- Terms of Use：<https://freemusicarchive.org/Terms_of_Use>

许可：
- 官方写明每首音频都对应：
  - Creative Commons
  - Public Domain
  - 或自定义许可

适配性：
- 公开音频来源清楚
- 可作为 anglophone/Western anchor 的备选池

风险：
- 许可是逐曲目级，不是单一统一 license
- 后续需要做逐轨 license 过滤
- 语言也需要额外筛选

结论：
- `backup_only`

---

## 4.3 Germany

### 候选 1：Schubert Winterreise Dataset

来源：
- Zenodo：<https://zenodo.org/records/5139893>

许可：
- 页面写明：`CC BY 3.0`

一手证据：
- Zenodo 页面写明数据集采用 CC BY 3.0
- 同时说明：9 个演出版本中只有 2 个音频版本被实际包含

适配性：
- 文化定义非常清楚
- 如果你把 Germany 定义成 `german_art_song`，它是高度匹配的

问题：
- 音频规模太小
- 远达不到第一阶段的 `400+` 样本目标

结论：
- `provisional`
- 不适合作为 Germany 主来源

### 候选 2：Open Music Academy German folk / choral pages

来源示例：
- `Die Gedanken sind frei`：<https://openmusic.academy/docs/ezYvknrxw7HVWX2yFHt3Eo/chorsingen-easy-peasy-die-gedanken-sind-frei>
- `Im Frühtau zu Berge`：<https://openmusic.academy/docs/q3AczjW9VvxJzRmHNoETxF/chorsingen-easy-peasy-im-fruehtau-zu-berge>

许可：
- 页面明确写有 `CC BY`

适配性：
- 与 German folk / choral 方向相符
- 许可清楚

问题：
- 这更像零散开放教学录音
- 不是大规模统一音频数据集
- 很难扩展到 `400+`

结论：
- `not_primary`
- 只能作为小规模补充来源

### Germany 当前结论

Germany 域目前**缺少一个“开放、规模足够、工程可直接接入”的主来源**。  
因此当前状态应定为：

- `blocked for primary-source freeze`

也就是说：
- Germany 域定义可以保留
- 但主来源还不能冻结

---

## 4.4 Spain

### 主候选：Corpus COFLA

来源：
- 数据集目录页：<https://computationalethnomusicology.wordpress.com/datasets/>

许可 / 访问条件：
- 页面说明：
  - COFLA 提供音频描述符与元信息
  - 音频文件 `shared on request for research purposes`

适配性：
- 从音乐内容上看，它几乎就是你要的 `flamenco` 候选
- 学术相关性很强

问题：
- 音频不是“开箱即下”的公开下载模式
- 这不满足当前 v2 第一阶段“可直接导入、可脚本复现”的要求

结论：
- `blocked`
- 除非后续获得明确研究访问许可，否则不建议作为 v2 主来源

### Spain 当前结论

Spain 域如果坚持走 `flamenco`，**学术上方向正确，但当前公开可复现性不足**。  
所以应保持：

- 域定义保留
- 主来源未冻结

---

## 4.5 Japan

### 候选 1：RWC Music Database（traditional Japanese subset）

来源：
- RWC Music Database 总页：<https://staff.aist.go.jp/m.goto/RWC-MDB/>
- Genre DB 说明页：<https://staff.aist.go.jp/m.goto/RWC-MDB/rwc-mdb-g.html>
- 2026 Zenodo 重发页：<https://zenodo.org/records/17177919>

许可：
- Zenodo 页面写明：`CC BY-NC 4.0`

适配性：
- 官方 genre page 明确包含：
  - Enka
  - Min'you
  - Gagaku
- 所以它确实包含传统日本相关子类

问题：
- 传统日本子类在整库中所占规模明显有限
- 大概率不够支撑 `400+` 主域样本

结论：
- `provisional`
- 可作为日本域的小规模合法候选
- 不适合作为当前主来源冻结

### 候选 2：tts-dataset/japanese-singing-voice

来源：
- Hugging Face：<https://huggingface.co/datasets/tts-dataset/japanese-singing-voice>

许可：
- 数据集卡显示 `CC-BY-NC-4.0`
- 但同时明确说明：
  - 音频版权仍归原始所有者
  - 数据集来自公开 YouTube 视频
  - 访问需接受条件

适配性：
- 工程上可接入
- 规模较大

问题：
- 不是 `japanese_traditional`
- 它更像“大规模日本歌声数据”
- 与当前预定文化域定义不一致

结论：
- `backup_only`
- 只有在你愿意把 Japan 域定义放宽为“Japanese music audio”时，才适合考虑

### Japan 当前结论

Japan 域目前的主要问题不是完全没有来源，而是：

- 真正“传统日本音乐 + 规模足够 + 许可清楚 + 可脚本导入”的来源还没找到

因此当前状态应为：

- `blocked for primary-source freeze`

---

## 5. 当前推荐的冻结结果

基于当前审计，我建议的结果是：

### 可以先冻结并推进

- `china`：`ccmusic-database/CTIS`
- `anglo_pop`：`MTG-Jamendo`（需加 pop + language 过滤）

### 先保留域定义，但不要冻结主来源

- `germany`
- `spain`
- `japan`

---

## 6. 现在最合理的下一步

### 先做的事

1. 保留 5 域目标不变  
2. 先对 `china` 和 `anglo_pop` 做实际导入 probe  
3. 同时继续补筛：
   - Germany 主来源
   - Spain 主来源
   - Japan 主来源

### 不建议现在做的事

- 在 Germany / Spain / Japan 主来源尚未冻结前，立刻开始大规模统一 embedding
- 把小规模或不够开放的来源硬塞成主来源
- 在 source audit 未完成时启动人类标注

---

## 7. 一句话总结

当前第二阶段来源筛选的结果是：

**China 已经 ready，Anglo-pop 基本可推进；Germany / Spain / Japan 的主来源仍未达到“规模足够 + 许可清晰 + 工程可直接接入”的冻结标准。**

这不是坏事，反而说明 source audit 起到了作用：

- 我们现在知道哪些域能立即开工
- 也知道哪些域必须继续补证据，避免后面走进无法复现或许可不清的死路
