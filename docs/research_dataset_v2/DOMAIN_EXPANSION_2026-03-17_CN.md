# V2 Domain Expansion Candidates (2026-03-17)

本文档在当前 `v2_main` 基础上，重新整理一轮“还能继续扩哪些文化域”的候选清单。

当前主实验域是：
- `china`
- `india`
- `anglo_pop`
- `kazakhstan`
- `germany`

本轮筛选标准：
- 必须是 **真实音乐音频**，而不是纯文本、纯语音、纯 MIDI、纯特征
- 优先考虑 **100+** 条可用音频，最好能达到 `100-300+`
- 优先考虑 **公开可下载**；如果不是公开下载，则至少要有 **清楚的研究访问路径**
- 优先考虑 **文化域定义清楚**，而不是泛平台或泛国家入口

## 结论先看

如果现在要继续扩域，我最推荐优先考虑：

1. `korea`
2. `arab_andalusian`
3. `georgia`
4. `norway`
5. `spain`
6. `japan`
7. `ireland`
8. `persia`
9. `turkey_dunya`

其中：
- `korea` 是当前最值得推进的新域，因为已经出现了多条官方音频/音频清单路线，而且总量有机会尽快超过 `100`
- `arab_andalusian` 是最有研究味道、同时公开性也不错的一条线
- `georgia` 是一个小而干净、文化辨识度很强的新增域
- `norway` 是最容易立刻接进来的一条开放音频线
- `spain / japan / ireland` 都很强，但更偏 archive / restricted-access 路线

## A. 第一梯队：最值得优先推进

### 1. Korea

代表来源：
- `재단법인국악방송_전국8도민요MR_20240301`
  - <https://www.data.go.kr/data/15098241/fileData.do>
- `재단법인국악방송_국악_MR_정보조회서비스`
  - <https://www.data.go.kr/data/15098387/openapi.do>
- `국가유산청 국립무형유산원_소장 음원자료 정보 목록 조회 서비스`
  - <https://www.data.go.kr/data/15094324/openapi.do>
- 区域型文件数据：
  - 경기민요 39 条：
    <https://www.data.go.kr/data/15142755/fileData.do>
  - 남도민요 15 条：
    <https://www.data.go.kr/data/15142756/fileData.do>
  - 서도민요 19 条：
    <https://www.data.go.kr/data/15142758/fileData.do>

为什么值得做：
- `전국8도민요MR` 公开页面明确写了 `전체 행 105`
- `국악_MR_정보조회서비스` 明确是民谣 / 国乐 MR 音频查询 API，且 `이용허락범위 제한 없음`
- `국립무형유산원` 的音源资料 API 明确说可用于音源资料检索，并描述为可在线阅览、研究、教育使用
- 文化域辨识度很强，适合作为 `korean_traditional / gugak` 方向

主要风险：
- 现在我们还没有做 item-level 批量导入，所以“能不能稳定批量拿到音频 URL”还要继续验证
- MR/伴奏类音频要注意和“完整成品音乐”区分；如果路线最终偏 MR，需要在论文里讲清楚

我的判断：
- **很值得优先推进**
- 如果下一步只新增一个域，我会把 Korea 放在前列

### 2. Arab-Andalusian

代表来源：
- CompMusic corpora:
  <https://compmusic.upf.edu/corpora>
- Dunya developers:
  <https://dunya.compmusic.upf.edu/developers/>
- Andalusian corpus info:
  <https://dunya.compmusic.upf.edu/andalusian/info>
- Andalusian API docs:
  <https://dunya.compmusic.upf.edu/docs/andalusian.html>

为什么值得做：
- CompMusic 官方语料页写明 Arab-Andalusian corpus 包含 `338 recordings (112 hours)`
- Dunya developers 页明确写了 `Arab Andalusian` 的 `Audio = open`
- Andalusian corpus info 页又给出一个更易直接理解的公开子集：`156 audio recordings (songs)`
- 明确有 `download_mp3` API 文档，工程接入非常像我们已经熟悉的 CompMusic 路线

主要风险：
- 文化定义是 “Arab-Andalusian / Maghreb / Moorish Spain heritage”，需要我们在论文里把域定义写清楚
- 语料结构可能会更偏 archive corpus，而不是现代录音库

我的判断：
- **强烈推荐**
- 学术味道非常好，而且比单纯继续补欧美流派更符合你的“跨文化”主张

### 3. Georgia

代表来源：
- Erkomaishvili dataset:
  <https://zenodo.org/records/6900390>
- TISMIR article:
  <https://transactions.ismir.net/articles/10.5334/tismir.44>
- Dezrann corpus status:
  <https://doc.dezrann.net/status>

为什么值得做：
- TISMIR 文章明确写到公开音频集合包含 `101 recordings`
- Dezrann 状态页进一步说明该 corpus 含 `101 pieces`、`404 recordings`
- 许可证是 `CC-BY-NC-4.0`
- 文化域非常清楚：traditional Georgian sacred / polyphonic singing

主要风险：
- 规模不是特别大，适合做第 6 域或 exploratory 域，不适合拿来当超大主域
- 以多声部宗教/民间声乐为主，风格相对集中

我的判断：
- **很适合补成一个“小而强”的新域**
- 如果你想要一个合法、研究味道强、规模刚过线的新域，它比 France 之类更现实

### 4. Norway

代表来源：
- HF2 Hardanger fiddle:
  <https://huggingface.co/datasets/Bots4M/HF2-Hardanger-fiddle-dataset>

为什么值得做：
- 公开可下载
- `119` audio-MIDI pairs
- `WAV`
- `CC-BY-4.0`
- 域定义清楚：Norwegian folk / Hardanger fiddle

主要风险：
- 规模刚过 `100`
- 域比较窄，主要围绕 Hardanger fiddle

我的判断：
- **最容易立刻接入**
- 但从论文文化广度角度，学术冲击力略低于 Korea / Arab-Andalusian

## B. 第二梯队：很有价值，但更偏受限访问或档案访问

### 5. Spain

代表来源：
- Corpus COFLA:
  <https://computationalethnomusicology.wordpress.com/datasets/>

为什么值得做：
- 页面明确写有 `1800+ flamenco recordings` 的 metadata / descriptors
- `flamenco` 作为文化域非常强，也很适合你这个题目

主要风险：
- 音频不是开箱即用下载，而是 `shared on request for research purposes`
- 推进速度取决于对方回复

我的判断：
- **非常好，但推进会慢**
- 如果你愿意走研究访问流程，它是最值得保留的强扩展域之一

### 6. Japan

代表来源：
- NDL Historical Recordings Collection:
  <https://www.ndl.go.jp/en/news/fy2020/210127_01.html>
  <https://www.ndl.go.jp/en/news/fy2021/220225_01.html>

为什么值得做：
- 官方说明整个 Historical Recordings Collection 规模接近 `50,000`
- 其中约 `5,500-6,000` 条可在线提供
- 对 Japan 文化域来说，官方性和代表性都很强

主要风险：
- 不是 ML-ready dataset
- item-level 导出和下载策略仍然复杂
- 需要更细的访问与筛选工作

我的判断：
- **值得继续保留**
- 但更适合做第二阶段扩展，而不是现在立刻冲主版

### 7. Ireland

代表来源：
- ITMA:
  <https://www.itma.ie/>
  <https://www.itma.ie/ga/fuinn/>

为什么值得做：
- ITMA 明确写了 `free universal access`
- 同时说明自 1993 年以来记录了 `1,300+ singers, instrumentalists, and dancers`
- 作为 Irish traditional music domain，文化域辨识度极强

主要风险：
- 它更像 archive / digital library，不是现成数据集
- 需要自己设计抓取/访问与元数据整理策略

我的判断：
- **是个很有分量的扩展域**
- 但工程成本会高于 Norway / Georgia

## C. 第三梯队：潜力不错，但当前有关键缺口

### 8. Persia

代表来源：
- `Razavipour/persian-traditional-instruments`
  <https://huggingface.co/datasets/Razavipour/persian-traditional-instruments>

为什么值得做：
- `512 rows`
- Audio modality 明确
- 域定义清楚：Persian traditional instruments

主要风险：
- README 基本为空
- 公共卡片没有稳定可引用的 license 字段

我的判断：
- **工程上可做，合规上还不够稳**
- 它现在很像我们之前对 Turkey 的判断

### 9. Turkey via CompMusic / Dunya

代表来源：
- CompMusic corpora:
  <https://compmusic.upf.edu/corpora>
- Dunya developers:
  <https://dunya.compmusic.upf.edu/developers/>
- Makam stats:
  <https://dunya.compmusic.upf.edu/makam/stats>
- Turkish makam section dataset:
  <https://compmusic.upf.edu/node/234>
- Turkish makam acapella sections dataset:
  <https://compmusic.upf.edu/turkish-makam-acapella-sections-dataset>

为什么值得做：
- CompMusic corpus 级别写了 `6601 audio recordings`
- `Makam` 在 Dunya developers 中是 `Audio = restricted`
- 附带多个公开子数据集，例如：
  - section dataset `257 audio recordings`
  - acapella sections dataset `12 performances`
  - şarkı vocal dataset `12 performances`

主要风险：
- 完整语料音频不是公开开放，而是 restricted academic access
- 公开子集规模分散，需要拼装

我的判断：
- **如果你愿意重新走 Dunya 学术访问，这条线会比当前 Turkey HF 源更稳**
- 但短期成本不低

## D. 不建议现在优先推进的

### France

代表来源：
- INA research dataset project:
  <https://www.ina.fr/institut-national-audiovisuel/research/dataset-project>

问题：
- 更像研究访问平台，不是音乐专用语料
- 当前没找到一个像 Korea / Arab-Andalusian / Georgia 那样清楚的“音乐文化域 + 音频 + 规模”组合

我的判断：
- **暂时不如前面的候选**

### 纯 vocal-only / 纯 speech / 纯 symbolic 候选

比如：
- Japanese singing voice gated datasets
- Common Voice / ASR / TTS
- 只有 MIDI、歌词、频谱图的数据

问题：
- 不适合你现在的“跨文化音乐推荐”主叙事
- 容易把文化域定义带偏

## 最终建议

### 如果只加 1 个域

优先级：
1. `korea`
2. `arab_andalusian`
3. `georgia`
4. `norway`

### 如果加 2 个域

优先级组合：
1. `korea + arab_andalusian`
2. `korea + georgia`
3. `arab_andalusian + norway`

### 如果你想要“低风险、快推进”

优先选：
- `norway`
- `georgia`

### 如果你想要“论文上更有跨文化说服力”

优先选：
- `korea`
- `arab_andalusian`

## 当前我最推荐的实际动作

如果我们真的要继续扩域，而不是只停在文档层，我建议顺序是：

1. 先推进 `korea`
2. 再推进 `arab_andalusian`
3. 如果还想再加一个低摩擦域，再接 `georgia` 或 `norway`

这样最平衡：
- 文化跨度更大
- 数据规模仍然可控
- 不会把项目拖回“重新找一堆弱来源”的状态
