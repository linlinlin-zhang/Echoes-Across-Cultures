# 非西方国家文化域音乐数据库广搜与短名单（2026-03-18）

## 1. 目标

本轮目标不是继续泛泛搜“非西方音乐数据集”，而是收敛到适合当前项目主线的数据源：

- 优先考虑国家或至少强国家文化域指向
- 优先考虑音频单位能稳定满足 `>30s`
- 优先考虑有音频而不只是标签/频谱/符号
- 优先考虑后续能进入你现在这条 `metadata -> audio -> embedding -> culture/domain` 管线

---

## 2. 先给结论

如果目标是尽快搭出 `3-4` 个非西方国家文化域，我建议的主短名单是：

1. **India**  
   主源：`Saraga / CompMusic Indian Art Music`

2. **Turkey**  
   主源：`CompMusic / Dunya Turkish Makam corpus`

3. **China**  
   主源：`Annotated Jingju Arias Dataset`

4. **Korea` 或 `Indonesia` 二选一**  
   - 想要现代流行、可直接拉大量歌曲：`AI-Hub 音乐相似性判别数据（优先取 Trot 子集）`
   - 想要更传统、更明确的本土音乐传统：`Indonesian Music Archive`

也就是说，最稳的三域是：

- `india`
- `turkey`
- `china`

第四域建议按你的偏好选：

- 想更“现代音乐生态”就选 `korea`
- 想更“传统音乐文化性”就选 `indonesia`

---

## 3. A-tier：建议直接纳入主线的 3 个国家文化域

### 3.1 India

#### 推荐主源

- `Saraga Carnatic and Hindustani collections`
- `Indian Art Music Raga Recognition datasets`

#### 关键证据

CompMusic 官方数据页明确写到：

- Saraga 提供两大印度艺术音乐开放语料
- 含时间对齐 melody / rhythm / structure annotations
- Raga recognition datasets 明确是 **full length audio recordings**
- Saraga 官方 access 页明确说可通过 Zenodo 直接下载，并支持 `mirdata` / `compiam`

#### 时长判断

这是当前最稳的一类：

- 官方直接写了 `full length audio recordings`
- 不需要做“是否大于 30 秒”的额外假设

#### 适合作为文化域的原因

- 文化语义很强，不是泛流派
- 有成熟 metadata 与社区工具链
- 后续可细分为 `hindustani` / `carnatic`

#### 风险

- 如果你把 India 当成单一文化域，内部异质性会比较大
- 更稳的做法是先选一个子传统作为主域，例如先做 `hindustani`

---

### 3.2 Turkey

#### 推荐主源

- `CompMusic Turkish makam corpus`
- 如需更小更可控的切入点，可先从：
  - `Turkish Makam Section Dataset`
  - `Turkish şarkı vocal dataset`
  - `Turkish Makam recognition dataset`
  入手

#### 关键证据

Dunya 官方统计页明确写到：

- Turkish makam corpus 含约 `6500` 条 audio recordings
- 总时长约 `412 hours`
- 另有 scores、lyrics、editorial metadata

CompMusic 官方 datasets 页还给出了多个任务子集：

- `257` audio recordings 的 section dataset
- `10` 条 vocal recordings 的 şarkı vocal dataset
- `50 x 20 makams` 的 recognition dataset

#### 时长判断

这个域也很稳：

- `412 hours / 6500 recordings`，平均每条明显远大于 `30s`
- 即使落到任务子集，多数也仍是曲段或录音级内容

#### 适合作为文化域的原因

- `makam` 是强文化语义，不是通用 western genre
- 土耳其方向在 CompMusic 体系里非常成熟
- 适合后续做 `culture=turkey`，并保留 `makam / form / performer` 作为 `substyle`

#### 风险

- Dunya 主 corpus 的访问方式比纯 Zenodo 包略复杂
- 如果你想快速起量，建议优先使用官方可直接下载的任务子集

---

### 3.3 China

#### 推荐主源

- `Annotated Jingju Arias Dataset`

#### 不再把它排在第一位的旧候选

- `OpenCpop`

#### 为什么这次更推荐 Jingju，而不是直接推荐 OpenCpop

`OpenCpop` 官方 GitHub README 明确写的是：

- `100 unique Mandarin songs`
- 但最终发布数据是 `3756 utterances`
- 总时长约 `5.2 hours`

这意味着：

- `OpenCpop` 的发布单位本质上是 **utterance-level wavs**
- 平均每条样本远小于 `30s`
- 如果你要求“用于文化域建模的音频单元最好都在 30 秒以上”，那它不能直接无条件满足

相比之下，`Annotated Jingju Arias Dataset` 更适合这次的筛选标准：

- 官方明确写是 `34 jingju arias`
- 研究音频可用于 research purposes
- 每个 aria 有完整分层标注：artist / school / role-type / shengqiang / banshi / lyric lines / syllables / percussion patterns

#### 时长判断

这里虽然页面没有逐条列 aria 时长，但 aria 级京剧唱段在实际语义上显然不是几秒片段。  
和 `OpenCpop` 的 utterance 发布粒度相比，`Jingju arias` 更接近你需要的 `>30s` 单位。

#### 适合作为文化域的原因

- 中国文化语义非常强
- 元数据解释性强
- 传统戏曲比通用中文流行更能体现文化域差异

#### 风险

- 音频是 `available for research purposes`，需要联系维护者
- 规模不大，适合作为高纯度中国传统域，而不是“大规模中国音乐总域”

#### China 域的现实建议

如果你坚持“现代中文流行”为中国主轴：

- `OpenCpop` 仍可作为 modern Chinese singing anchor
- 但要承认它是 utterance-level，不是天然 `>30s`

如果你坚持“音频单位最好 >30s”：

- 当前更应该优先用 `Jingju arias`

---

## 4. 第四域：两个可选方向

### 4.1 Korea 方案

#### 推荐主源

- `AI-Hub 音乐 유사성 판별 데이터`

#### 关键证据

AI-Hub 官方页明确写到：

- 数据类型：`오디오`
- 数据格式：`WAV`
- 构建量：`20,000곡`
- 44.1kHz / 16bit / stereo
- genre 包括：
  - `Ballade`
  - `Dance`
  - `Hiphop`
  - `RnB`
  - `Rock`
  - `Trot`

#### 为什么它值得做韩国域

这里最关键的不是“韩语”本身，而是：

- 它是韩国平台体系下构建的歌曲级音频数据
- 并且包含 `Trot` 这一较强韩国本土文化指向子域

#### 时长判断

页面没有直接逐条给出 duration 字段，但数据单位明确是 `곡`（song）级 WAV，且还围绕整首歌做 cover / arrangement / rhythm / tempo / timbre 变化。  
这里我判断“绝大多数样本显著大于 30 秒”是**高概率推断**，但这条仍建议你在真正下载前做一次 sample audit。

#### 风险

- 官方页明确写了：`※ 내국인만 데이터 신청이 가능합니다.`
- 也就是：**仅本国用户可申请**

#### 结论

- 如果你拿得到权限，它是很强的现代韩国域候选
- 如果你拿不到权限，就不要把它列成主线依赖

---

### 4.2 Indonesia 方案

#### 推荐主源

- `Indonesian Music Archive`（Cornell）

#### 关键证据

Cornell 官方页明确写到：

- 约 `193 hours` 音频
- 录音时间跨 `1952-1977`
- 大部分是 `Central Javanese gamelan`
- 还包括 Java / Lombok / Aceh 的传统音乐、仪式与说唱/叙事内容

#### 为什么它值得做印度尼西亚域

- 国家文化语义强
- 传统性很强，不是通用世界音乐标签池
- 还有 location / agents / materials/techniques 等可继续深挖的馆藏 metadata

#### 时长判断

Cornell 页给的是 collection-level `193 hours`，没有在集合首页逐条列出每条 recording duration。  
考虑到它是“录音档案馆”而不是短片段分类库，collection 语义上明显偏向长录音，但这里仍需要一轮 item-level metadata sweep 才能把 `>30s` 规则做成硬约束。

#### 风险

- 档案馆型资源的 metadata 清洗成本会比标准机器学习数据集更高
- 不一定像 Zenodo/AI-Hub 那样一键结构化下载

#### 结论

- 如果你想要更传统、更有文化厚度的第四域，优先选它
- 但它会比 Korea 方案更费清洗时间

---

## 5. 为什么这次不把 OpenCpop 继续列为“30 秒以上主源”

这是这轮搜索里最重要的修正点。

`OpenCpop` 很适合：

- 中文现代流行歌声
- 高质量录音
- 明确的 Mandarin singing corpus

但它官方 README 同时明确写：

- 数据最终是 `3756 utterances`
- 而不是 100 首完整长音频直接发布

所以：

- 如果你当前约束是“数据单元最好都 >30s”
- 那 `OpenCpop` 不能被无条件视为满足条件

它更适合作为：

- `china_modern_pop_singing` 补充源
- 或后续再做 song-level 聚合/重构的候选源

---

## 6. 为什么这次不把 MTG-Jamendo 列入 3-4 国家的主短名单

因为它前一轮审计已经很明确：

- 可以补一点 `world/ethno/african/asian` 弱候选
- 但没有 `country / location / language`
- 更像风格池，不像国家文化域主源

所以：

- 它适合 secondary source
- 不适合主短名单

---

## 7. 为什么当前没有把日本列进主短名单

这一轮广搜里，日本方向能找到的公开资源大多存在以下问题之一：

- 更偏 speech / singing synthesis，而不是 country-cultural music corpus
- 虽然有 song 数据，但规模偏小或文化代表性偏弱
- 更偏特定任务 benchmark，而不是文化域建库源

因此，在当前这轮筛选标准下，日本没有进入主短名单。

这不是说日本没有音乐数据，而是说：

- 当前没有找到一个同时满足
  - 国家文化域清晰
  - 音频足够长
  - 可获取性较好
  - 适合直接进你这条建库管线

的强候选。

---

## 8. 最终建议：一版可执行组合

如果你想尽快做出 `4` 个非西方国家文化域，我建议这样组合：

### 方案 A：优先可执行

- `india` -> Saraga / CompMusic Hindustani
- `turkey` -> Turkish makam corpus / task datasets
- `china` -> Annotated Jingju Arias
- `korea` -> AI-Hub 音乐相似性判别数据中的 `Trot` 子集

适合你如果：

- 更在意现代研究可操作性
- 能解决韩国 AI-Hub 的访问权限

### 方案 B：优先传统文化厚度

- `india` -> Saraga Hindustani
- `turkey` -> Turkish makam
- `china` -> Jingju arias
- `indonesia` -> Indonesian Music Archive

适合你如果：

- 更在意传统音乐文化性
- 可以接受第四域清洗成本更高

---

## 9. 我建议的下一步

最值得立刻继续的不是再“找更多库”，而是把短名单变成可下载、可清洗、可进你管线的执行单：

1. 先固定四域组合  
   我建议先在 `A 方案` 和 `B 方案` 里二选一。

2. 对每个域单独建立 `source audit sheet`  
   要记录：
   - access path
   - license/access restriction
   - duration evidence
   - metadata fields
   - expected usable tracks

3. 先只做 metadata harvest，不下全量音频  
   先确认每个域是否真能稳定抽出 `>30s` 样本。

4. 再按域做小样本 probe  
   每域先拿 `50-100` 条，验证 embedding 和文化区分度。

---

## 10. 一句话总结

这轮广搜之后，**最值得推进的非西方国家文化域短名单是 India、Turkey、China，再在 Korea 和 Indonesia 中选一个作为第四域；其中 China 这次要把 `OpenCpop` 从“30 秒以上主源”降级为补充源。**

---

## Sources

- CompMusic datasets: <https://compmusic.upf.edu/datasets>
- Saraga access: <https://mtg.github.io/saraga/access.html>
- Hindustani rhythm dataset: <https://compmusic.upf.edu/hindustani-rhythm-dataset>
- Dunya Turkish makam stats: <https://dunya.compmusic.upf.edu/makam/stats>
- Annotated Jingju Arias: <https://compmusic.upf.edu/node/349>
- OpenCpop official repo: <https://github.com/wenet-e2e/opencpop>
- OpenCpop official site: <https://wenet-e2e.github.io/opencpop/>
- AI-Hub music similarity dataset: <https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71544>
- Indonesian Music Archive (Cornell): <https://digital.library.cornell.edu/collections/indonesianmusic>
