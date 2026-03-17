# Research Dataset V2 扩展文化域建议

更新时间：2026-03-17

本文档用于回答一个更具体的问题：

- 如果当前 `v2-main` 已经稳定在 `china / india / anglo_pop / kazakhstan / germany`
- 我们还想再增加 `1-2` 个文化域
- 那么哪些候选域最值得补，且不会把项目重新拖回“找不到数据、许可说不清、工程迟迟落不了地”的状态

---

## 1. 当前判断标准

本轮扩展检索继续沿用以下筛选标准：

- 必须有**真实音频**
- 最好能达到 **100-250** 条的可用规模
- 许可需要**明确**，或者至少能被记录成“受限但合法”的研究访问模式
- 最终要能进入当前 `audio -> metadata -> embedding` 流水线

因此，单纯有 metadata、只有特征、只有歌词、只有 MIDI 的来源都不进入优先候选。

---

## 2. 当前最值得补的候选域

### 2.1 Korea

优先级：高  
推荐类型：**强可执行扩展域**

当前最有价值的来源：

- National Gugak Center digital audio site (`digitaleum`)  
  <https://www.gugak.go.kr/digitaleum>
- National Gugak Center Korean traditional music information API  
  <https://www.data.go.kr/en/data/3062269/openapi.do>
- National Gugak Center digital audio OpenAPI  
  <https://www.data.go.kr/data/15097515/openapi.do>
- National Gugak Center archive file data  
  <https://www.data.go.kr/en/data/15083294/fileData.do>

为什么值得补：

- 官方公告与媒体报道都明确说明 `digitaleum` 提供 **wav / mp3** 高音质国乐数字音源，且支持创作使用
- 官方文化机构来源，文化定义清楚
- 与当前主域相比，能显著增强东亚文化覆盖
- 已经不只是“可能有资源”，而是出现了**公开音源站 + OpenAPI + 归档线**三层结构
- 如果 sample audit 证实 item-level 可稳定访问，它会是一个很有说服力的新增域

当前风险：

- 仍需对 item-level 音频可访问性做抽样审计
- 仍需更清楚地确认“API 返回记录”与“可用音频对象”之间的映射关系
- 若走 OpenAPI 路线，仍需 service key

结论：

- **Korea 现在已经不是单纯受限候选，而是当前最值得优先补的新增域之一。**

---

### 2.2 Norway

优先级：高  
推荐类型：**开放下载候选**

当前来源：

- Bots4M/HF2-Hardanger-fiddle-dataset  
  <https://huggingface.co/datasets/Bots4M/HF2-Hardanger-fiddle-dataset>

已确认点：

- 明确公开为音频数据
- 数据集卡许可证为 `CC-BY-4.0`
- 规模约 `119` 条

为什么值得补：

- 这是当前新增候选里**最干净、最容易接入**的一条开放音频路线
- 不需要走额外申请、注册、馆藏访问或 gated 流程
- 如果我们的目标是“再加一个不拖慢进度的文化域”，它的性价比很高

当前限制：

- 规模低于当前 `v2-main` 其他主域的 `250`
- 音乐文化定义较窄，基本围绕 Hardanger fiddle / 挪威民间器乐传统

结论：

- **如果我们想快速、低风险地增加一个开放文化域，Norway 是当前最佳候选。**

---

### 2.3 Spain

优先级：中高  
推荐类型：**受限研究访问候选**

当前来源：

- Corpus COFLA  
  <https://computationalethnomusicology.wordpress.com/datasets/>

为什么它仍然重要：

- `flamenco` 作为文化域定义非常清楚
- 规模上远强于很多开放小语料
- 学术上与跨文化音乐推荐叙事很契合

当前问题：

- 音频不是开放直接下载
- 需要研究访问申请
- 不适合作为“马上就能接入”的下一个域

结论：

- **如果我们愿意走受限访问并做访问日志记录，Spain 是一个比 France 更值得优先争取的文化域。**

---

### 2.4 Persia

优先级：中  
推荐类型：**开放下载但许可待核实**

当前来源：

- Razavipour/persian-traditional-instruments  
  <https://huggingface.co/datasets/Razavipour/persian-traditional-instruments>

为什么它有吸引力：

- 数据集中可见约 `512` 条音频
- 文化定义清楚
- 如果许可证得到确认，它会是一个非常好的中东 / 波斯文化域候选

当前问题：

- 公共 dataset card / API 没有暴露稳定许可证字段
- 目前和 Turkey 的问题类似：工程可用，但论文合规性还不够硬

结论：

- **Persia 是很好的备选扩展域，但在许可证证据补强前，不建议直接拉进主实验域。**

---

## 3. 当前不建议优先补的域

### France

虽然 `INA research dataset project` 在“受限研究访问”模式下是现实可用的：

- <https://www.ina.fr/institut-national-audiovisuel/research/dataset-project>

但它更像访问平台而不是一个已经冻结好的音乐语料。  
在当前阶段，France 的接入成本和后续整理成本都明显高于 Korea / Norway / Spain。

当前判断：

- **France 仍不建议优先于 Korea / Norway / Spain。**

---

## 4. 最推荐的两种扩展策略

### 策略 A：最稳、最省力

新增：

- `norway`

原因：

- 开放下载
- 许可清楚
- 能最快接入现有流水线

适合目标：

- 希望尽快把主域从 `5` 扩到 `6`
- 不想让工程节奏被新的访问申请拖住

---

### 策略 B：兼顾说服力与文化覆盖

新增：

- `norway`
- `korea`

原因：

- `norway` 负责提供一个低摩擦的开放新增域
- `korea` 负责增加东亚文化覆盖和项目叙事的丰富度

适合目标：

- 希望最终扩成 `6-7` 个域
- 接受“一个开放域 + 一个受限研究域”的组合

---

### 策略 C：更偏文化辨识度

新增：

- `korea`
- `spain`

原因：

- 两者都比 France 更有文化域辨识度
- 对论文“跨文化推荐系统”叙事更强

限制：

- 都需要受限研究访问模式
- 会拉长真正进入 embedding 阶段的时间

适合目标：

- 更重视最终论文的文化覆盖叙事
- 不介意数据接入周期更长

---

## 5. 当前建议

如果只加 **一个** 域：

- 选 `norway`

如果加 **两个** 域：

- 优先 `norway + korea`

如果你愿意接受更慢的访问流程、但希望文化辨识度更强：

- 再考虑 `spain`

不建议当前优先：

- `france`
- `persia`（在许可证证据补强之前）

---

## 6. 与当前 `v2-main` 的关系

当前主实验域仍然是：

- `china`
- `india`
- `anglo_pop`
- `kazakhstan`
- `germany`

对应文件：

- [metadata_v2_main.csv](E:/Desktop/Echo/storage/public/research_dataset_v2/metadata_v2_main.csv)

新增域建议是为了下一步扩展，而不是否定当前主版。  
当前最稳的做法仍然是：

1. 保留 `v2-main`
2. 额外新增 `1-2` 个扩展域
3. 再决定是否将它们升级为正式主实验域
