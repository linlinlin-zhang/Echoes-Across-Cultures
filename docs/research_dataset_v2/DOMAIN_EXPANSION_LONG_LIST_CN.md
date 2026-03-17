# Domain Expansion Long List

本文档汇总一轮扩域导向的深度检索结果。筛选标准是：

- 真实音频，而不是只有符号、embedding、MIDI 或纯特征
- 尽量有明确许可，或至少有可记录的研究访问方式
- 对音乐推荐任务有实际意义，优先保留音乐文化域而不是纯语音域
- 目标规模优先考虑 `100-300+` 条可抽样音频，或更大的可访问语料

## 结论先看

如果现在要继续扩域，我最推荐优先考虑：

1. `norway`
2. `korea`
3. `arab_andalusian`
4. `spain`
5. `japan`
6. `persia`
7. `ireland`

其中：

- `norway` 是最容易直接落地的新增开放域
- `korea` 是最值得推进的官方文化机构域
- `arab_andalusian` 是非常有研究味道且音频开放的强候选
- `spain / japan` 很强，但更偏受限访问路线

## A. 强候选：适合优先接入

### 1. Norway: Hardanger fiddle

- 来源：`Bots4M/HF2-Hardanger-fiddle-dataset`
- 链接：<https://huggingface.co/datasets/Bots4M/HF2-Hardanger-fiddle-dataset>
- 关键事实：
  - `119` 个 audio-MIDI pairs
  - `39` 首独立曲目
  - `WAV`
  - `CC-BY-4.0`
- 判断：
  - 虽然规模不大，但已经达到 `100+`
  - 许可清楚
  - 工程接入最简单
  - 非常适合作为第 `6` 个域

### 2. Korea: National Gugak Center digital audio

- 来源：
  - <https://www.data.go.kr/data/15097515/openapi.do>
  - <https://www.data.go.kr/en/data/3062269/openapi.do>
- 辅助公开报道：
  - <https://www.koreatimes.co.kr/www/culture/2024/04/135_315698.html>
- 关键事实：
  - 官方机构：National Gugak Center
  - 数据类型明确写为 `국악디지털음원`（数字音源）
  - API 自动审批
  - 开发流量 `10,000`
  - 许可范围为公共著作物出处标示
  - 外部报道提到数字音源站拥有 `16,721` 条国乐数字音源，并提供 `wav/mp3`
- 判断：
  - 这是当前最值得认真推进的官方扩展域之一
  - 需要 service key 和 item-level sample audit
  - 一旦打通，文化代表性会很强

### 3. Arab-Andalusian

- 来源：
  - CompMusic corpora: <https://compmusic.upf.edu/corpora>
  - Dunya developers: <https://dunya.compmusic.upf.edu/developers/>
  - Andalusian API docs: <https://dunya.compmusic.upf.edu/docs/andalusian.html>
- 关键事实：
  - `338` 录音、`112` 小时（CompMusic corpora 页面）
  - Dunya developers 页明确：`Arab Andalusian` 的 `audio` 是 `open`
  - API 文档里明确有 `download_mp3`
- 判断：
  - 文化辨识度强
  - 规模够
  - 研究味道强
  - 非常适合做扩展域，甚至有机会升成主实验域

## B. 很有价值，但更偏“受限/需申请/需额外工程”的候选

### 4. Spain: Flamenco / COFLA

- 来源：<https://computationalethnomusicology.wordpress.com/datasets/>
- 关键事实：
  - `corpus COFLA` 含 `1800+` flamenco recordings 的 audio descriptors 和 meta-information
  - 音频 `shared on request for research purposes`
- 判断：
  - 学术相关性极强
  - 很适合你的跨文化叙事
  - 但需要研究访问申请，接入周期较长

### 5. Japan: NDL Historical Recordings Collection

- 来源：
  - <https://www.ndl.go.jp/en/news/fy2020/210127_01.html>
  - <https://www.ndl.go.jp/en/news/fy2021/220225_01.html>
  - <https://www.ndl.go.jp/en/aboutus/pdf/digitized_contents_en.pdf>
- 关键事实：
  - Historical Recordings Collection 总量约 `48,700-50,000`
  - 其中约 `5,500-6,000` 可在线提供
  - 内容含 `music, storytelling performances, classic music, speeches`
- 判断：
  - Japan 线最有文化重量的一条
  - 但不是现成 ML-ready dataset
  - 需要更细的 item-level 筛选和访问策略

### 6. Ireland: ITMA

- 来源：
  - <https://www.itma.ie/>
  - <https://www.itma.ie/ga/fuinn/>
- 关键事实：
  - ITMA 自称是世界上最全面的 Irish traditional music archive
  - 强调 `free universal access`
  - 自 1993 年以来录制了 `1300+` singers, instrumentalists, dancers
- 判断：
  - 文化代表性极强
  - 但更像 archive / digital library，不是直接可下载的数据集
  - 适合当“可访问但需要二次整理”的候选域

### 7. Turkey: Dunya Makam / CompMusic

- 来源：
  - <https://compmusic.upf.edu/corpora>
  - <https://dunya.compmusic.upf.edu/developers/>
  - <https://dunya.compmusic.upf.edu/makam/stats>
- 关键事实：
  - Turkish makam corpus 约 `6500` 录音、`412` 小时
  - Dunya developers 页明确：`Makam` 的 `audio` 为 `restricted`
  - 但配套公开数据和 test datasets 很丰富
- 判断：
  - 如果愿意走 Dunya 学术访问流程，这是一条非常强的扩展路线
  - 比现在那个 license 不清的 HF Turkey 源更学术、更稳

### 8. Carnatic CC / Hindustani CC

- 来源：
  - <https://dunya.compmusic.upf.edu/developers/>
  - <https://compmusic.upf.edu/corpora>
- 关键事实：
  - Dunya developers 页明确：
    - `Carnatic CC collection`：`audio open`
    - `Hindustani CC collection`：`audio open`
  - 整体 Carnatic corpus 约 `1889` recordings / `397h`
  - 整体 Hindustani corpus 约 `1124` recordings / `305h`
- 判断：
  - 这两条线非常值得补做
  - 但真正可直接用的 CC 子集规模，需要拿 API token 后再精确统计
  - 作为扩展域或替换域都很有价值

## C. 可作为备用，但当前不如前面稳

### 9. Persia: solo traditional instrument set

- 来源：<https://huggingface.co/datasets/Razavipour/persian-traditional-instruments>
- 关键事实：
  - `512` 行
  - `Audio + Text`
  - 数据查看器可见真实音频字段
  - 但 README 基本为空，公开许可字段不清晰
- 判断：
  - 工程上可接
  - 文化域也清楚
  - 但合规证据还不够硬
  - 很像 `Turkey HF` 的问题，需要额外核许可

### 10. Greece: Lyra / Greek Folk

- 来源：
  - <https://arxiv.org/abs/2211.11479>
  - <https://zenodo.org/records/15470305>
- 关键事实：
  - `Lyra` 论文写到约 `1570` pieces，但依赖 YouTube timestamped links
  - 另一个 Zenodo `Greek Folk Music Dataset` 主要是 `MIDI + lyrics + metadata`
- 判断：
  - 学术价值高
  - 但当前不是“可直接用于统一音频 embedding”的强候选
  - 更适合作为后续 metadata / symbolic 扩展，而不是现在就进主实验域

### 11. France: INA dataset project

- 来源：<https://www.ina.fr/institut-national-audiovisuel/research/dataset-project>
- 关键事实：
  - 当前公开可见的是广播/新闻研究语料
  - 示例数据集包括 `277` 和 `804` 个音频文档
  - 明确是 scientific research access
- 判断：
  - 机构访问路线是稳的
  - 但音乐专属性不够强
  - 不如 Korea / Spain / Japan 适合你的主问题

## D. 明确不推荐直接当“音乐文化域”主候选

下面这些虽然也有音频、规模也不小，但不适合你现在这篇音乐推荐论文直接拿来当文化域：

- `KSS / Korean speech corpora`
- `YodaLingua-*`
- `Pazhvak`
- `CORAA`
- `Japanese singing voice` 这类明显偏 TTS / vocal-only 语料

原因：

- 它们主要是语音或歌声任务
- 不是完整音乐文化域
- 会削弱你的 MIR / recommendation 叙事

## 最推荐的新增策略

### 如果只加 1 个域

优先级：

1. `norway`
2. `korea`
3. `arab_andalusian`

### 如果加 2 个域

优先级组合：

1. `norway + korea`
2. `norway + arab_andalusian`
3. `korea + spain`

### 如果接受受限访问和较长推进周期

最有研究味道的组合：

- `korea`
- `spain`
- `japan`

## 当前最现实的建议

如果你想继续扩域，但不想把项目重新拖进数据泥潭，我建议：

1. 先把 `norway` 接进来
2. 同时推进 `korea` 的 API sample audit
3. 把 `spain / japan` 继续保留为第二梯队

这样能同时兼顾：

- 扩域速度
- 文化多样性
- 论文说服力
- 工程可控性
