# Research Dataset V2 第二轮多平台深度检索补充说明

更新时间：2026-03-14  
目的：补充 Germany / Spain / Japan 三个尚未冻结主来源的文化域，并回答“文化域样本量是否必须接近平衡”。

---

## 1. 这轮检索覆盖了哪些平台

这轮我重点查了以下平台与一手页面：

- Hugging Face 数据集卡
- Zenodo 数据集页面
- Europeana 集合与单条音频页面
- Wikimedia Commons 音频文件与分类页
- Open Music Academy 页面
- 官方项目/数据集主页

目标不是“搜到更多名字”，而是判断：

1. 有没有真实音频  
2. license 是否足够清楚  
3. 是否能进入现有导入工具链  
4. 是否有机会达到 `400+` 可用样本

---

## 2. 第二轮检索后的总体判断

结论比第一轮更清楚了：

- `china` 依然是最稳的 ready 域
- `anglo_pop` 依然是最可执行的现代流行锚点
- `germany` 出现了一个新的潜在方向：Europeana 的 Westphalian Folk Song and Sound Archive
- `spain` 依然是“学术方向正确，但大规模开放音频主来源不足”
- `japan` 依然缺少“传统日本音乐 + 规模足够 + 许可清楚 + 可脚本导入”的主来源

也就是说，深搜以后并没有推翻第一轮结论，但它提供了新的候选线索：

- Germany 现在不再是“完全无规模线索”，而是有了一个值得抽样审计的候选集合
- Spain / Japan 目前更适合作为“小规模合法 seed set”或后续补充域，而不是立刻冻结主来源

---

## 3. Germany：有新线索，但还不能直接冻结

### 新发现：Europeana Westphalian Folk Song and Sound Archive

来源：
- <https://www.europeana.eu/es/collections/organisation/1815-westphalian-folk-song-and-sound-archive>

这条线的优点：

- 主题上和 `german_folk` 很一致
- 从集合页看，条目规模看起来不小
- 比之前那些几十条级别的小来源更像“真正的主来源候选”

但当前仍然不能直接冻结，原因有两个：

1. Europeana 是聚合平台  
   集合页能看到条目，不代表每条都有稳定、可直接下载的音频

2. rights 往往是逐条声明  
   你不能把整个集合一概当作统一开放 license 数据集

### 当前建议

- Germany 域继续保留
- 把 Europeana Westphalian collection 作为第二阶段优先人工审计对象
- 先抽样审计 `50` 条，看：
  - 多少条真的有音频
  - 多少条 rights 足够开放
  - 多少条能脚本稳定下载

如果抽样通过，再考虑冻结为 Germany 主来源。

---

## 4. Spain：flamenco 方向仍对，但开放主源还是不够强

### 仍然最贴题的候选：COFLA

来源：
- <https://computationalethnomusicology.wordpress.com/datasets/>

问题没有变：

- 音频是 `shared on request for research purposes`
- 不是即取即用的开放音频主来源

### 新发现：Wikimedia Commons 上有合法小规模 flamenco 音频

示例：
- <https://commons.wikimedia.org/wiki/Category:Flamenco_songs_by_Rosario_Amador>

优点：

- 明确可见的 CC 信息
- 音乐上确实属于 flamenco
- 很适合做一个“合法 seed set”

问题：

- 规模太小
- 来源太碎
- 不适合作为主域数据集

### 当前建议

- Spain 域如果坚持 `flamenco`，现在仍不能冻结主来源
- 但可以把 Wikimedia Commons flamenco 条目保留为：
  - 小规模种子集
  - 质检样例
  - 后续定性分析样例

---

## 5. Japan：仍然缺少足够强的传统主来源

### 已知较清楚的主候选：RWC 的传统日本子类

来源：
- <https://staff.aist.go.jp/m.goto/RWC-MDB/rwc-mdb-g.html>
- <https://zenodo.org/records/17177919>

问题还是规模：

- 传统日本相关子类看起来存在
- 但很可能不足以支撑 `400+` 样本主域

### 新发现：Wikimedia Commons 上有合法传统日本单曲 / 民谣音频

示例：
- <https://commons.wikimedia.org/wiki/File:Sakura_Sakura.song.ogg>

优点：

- 许可可查
- 与 `japanese_traditional` 定义一致

问题：

- 完全不成规模
- 更像合法样例，不像主来源

### 当前建议

- Japan 域继续保留定义
- 但当前不冻结主来源
- 若短期一定要推进 Japan 域，有两个方向：
  - 方向 A：继续深搜传统日本开放音频源
  - 方向 B：放宽域定义，从 `japanese_traditional` 放宽为 `japanese_music_audio`

---

## 6. 文化域样本量必须差不多吗

### 短答案

不是必须完全一样。  
但**不能完全放任“收集到多少就用多少”**，否则会明显增加偏差风险。

### 为什么不建议完全随缘

你的系统和后续论文都会受到以下影响：

1. 大文化域会主导表示空间  
   无论是 backbone 还是 DCAS，下游几何都会更容易围着大域转

2. 推荐会更容易偏向大域  
   尤其跨文化推荐里，数据量大的域更容易成为“默认更可见”的文化

3. 域间对比会变得不公平  
   最终你分不清：
   - 是文化更难
   - 还是只是因为这个域样本更少

4. PAL 和后续人工标注成本会失衡  
   小域往往更难、却样本更少，如果前面不控量，后面很难补救

### 这件事在推荐研究里为什么重要

公开研究里一直反复出现两个相关问题：

- 交互和样本分布偏斜会带来 popularity bias
- 在跨域推荐中，data imbalance 是实际挑战

可参考：
- RecSys-DAN 指出 cross-domain recommendation 里的 data sparsity 和 data imbalance 是核心问题：
  <https://lihui.info/doc/TNNLS20.pdf>
- 音乐推荐中的 popularity bias 复现实验：
  <https://arxiv.org/abs/1912.04696>
- 推荐反馈循环与国家/地区代表性偏斜：
  <https://arxiv.org/abs/2408.11565>

### 那到底要平衡到什么程度

我建议用三档标准：

- 理想：各域样本数控制在 `±20%` 内
- 可接受：最大域和最小域不超过 `2:1`
- 高风险：超过 `3:1`

### 如果现实里做不到完全平衡怎么办

可以继续做，但一定要补这些机制：

- 训练时按文化域做分层采样或 balanced batches
- loss 加 culture-aware weighting
- 评测时报告 macro average，不只看 micro average
- 强制做 per-target-culture breakdown
- 测试集尽量保持平衡

所以答案不是“必须完全一样”，而是：

**不必一模一样，但绝不建议毫无约束地“收集到多少就用多少”。**

---

## 7. 现在最稳的决策建议

如果你坚持现在这 5 域目标，我建议：

### 继续保留 5 域目标不变

- `germany`
- `china`
- `spain`
- `japan`
- `anglo_pop`

### 但在实际执行层分成两组

#### 第一组：可立即推进

- `china`
- `anglo_pop`

#### 第二组：继续深搜/抽样审计后再冻结

- `germany`
- `spain`
- `japan`

### 关于样本量

当前建议不要放弃“近似平衡”的目标。  
可以放宽，不必一刀切 `400/400/400/400/400`，但建议至少控制在：

- `400-600` / 域 的窗口内
- 或至少让最大域不超过最小域的 `2x`

---

## 8. 一句话结论

第二轮多平台深度检索之后，当前最现实的判断是：

- `china` 和 `anglo_pop` 已经足够进入实际导入 probe
- `germany` 出现了值得深挖的新候选，但还必须做 rights 抽样审计
- `spain` 和 `japan` 仍缺少能立刻冻结的大规模开放主来源
- 文化域样本量不必完全一样，但如果严重失衡，会明显影响跨文化推荐结果与论文说服力
