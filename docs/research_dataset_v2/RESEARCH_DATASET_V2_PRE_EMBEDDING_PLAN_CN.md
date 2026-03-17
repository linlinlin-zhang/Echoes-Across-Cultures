# Research Dataset V2 预嵌入阶段方案

更新时间：2026-03-13  
适用项目：Echo / DCAS  
当前范围：从文化域定义开始，到“统一生成 embedding 数据库”之前为止。  
不包含内容：Gemini / CultureMERT embedding 实际批量生成、DCAS 训练、推荐评测、PAL 回灌。

---

## 1. 这份文档的目标

这份文档用于冻结 `research_dataset_v2` 的前置设计，避免后续出现以下问题：

- 文化域定义过宽，导致同一域内部风格过杂
- 不同来源的元数据字段不兼容
- 音频导入完成后才发现 license 或数据结构不适合
- 已经开始做人类标注，结果数据版本又大改
- embedding 生成之后才发现数据源混乱，导致整套向量库要重做

本阶段的成功标准不是“先下载最多的数据”，而是先把以下三件事固定住：

1. 每个文化域到底代表什么音乐域  
2. 每个文化域允许接入哪些公开来源  
3. 所有原始音频和元数据如何统一到同一个 schema

---

## 2. V2 的核心原则

### 原则 A：先定义文化域，再导入数据

文化域必须是“可描述、可筛选、可复现”的音乐域，而不是空泛的国家名。

### 原则 B：每个文化域内部尽量一致

同一文化域内部不应同时混入过多子传统，否则 DCAS 的 `zs / za` 更容易学乱。

### 原则 C：raw dataset 与 embedding dataset 解耦

`research_dataset_v2` 应该先作为统一的音频与元数据底座存在。  
Gemini embedding 只是后续在同一底座上派生出来的一种表示版本，而不是数据集本身。

### 原则 D：所有域最终必须用同一个 backbone 统一重嵌入

后续如果决定用 Gemini，就主域中的全部文化域都用 Gemini。  
不要出现“有些域用 CultureMERT，有些域用 Gemini”的混合向量空间。

### 原则 E：在 embedding 生成前，先把 license / 来源 / 字段统一好

这能最大程度降低返工成本。

---

## 3. V2 的 6 个目标文化域

当前正式主域按最新决策冻结为以下 6 域：

- `germany`
- `china`
- `japan`
- `india`
- `turkey`
- `anglo_pop`

但在工程与论文里，这 6 个域都必须进一步加上“音乐域定义”，避免国家标签过宽。

### 3.1 germany

建议定义：

- 首选：`german_folk`
- 备选：`german_art_song`

要求：

- 二选一，不要同时混合民歌传统和艺术歌曲传统
- 如使用 `german_folk`，优先保留器乐或民歌传统录音
- 如使用 `german_art_song`，需保证录音语义相对统一，不混现代流行或朗诵

### 3.2 china

建议定义：

- `chinese_traditional`

要求：

- 以中国传统器乐 / 民乐 / 民族器乐为主
- 不混现代华语流行
- 尽量避免“纯技法演示数据”占比过高；若必须使用技法数据，要在元数据中保留 `substyle` 或 `instrument_family`

### 3.3 japan

建议定义：

- `japanese_music_audio`

要求：

- 当前工程阶段允许放宽 Japan 域定义，以保证样本规模
- 优先保留日本传统音乐方向，但不把“必须纯传统”作为硬约束
- 在元数据中尽量保留 `substyle / instrument / language`

### 3.4 india

建议定义：

- `hindustani_raag`

要求：

- 以 Hindustani classical / raag-based 音频为主
- 不与宝莱坞流行或通用印度流行混合

### 3.5 turkey

建议定义：

- `turkish_music_audio`

要求：

- 当前阶段允许较宽定义
- 若后续主来源更清晰，可再收窄到 `makam` 或传统子域

### 3.6 anglo_pop

建议定义：

- `anglo_pop`

要求：

- 以英语现代流行作为锚点文化域
- 不混重金属、纯电子舞曲、说唱等过强子风格，除非明确作为子域
- 优先选择有清晰 `genre=pop` 或相近标签的数据

---

## 4. 每个文化域的接入要求

一个来源只有在满足以下要求时，才允许进入 `v2`：

### 必备要求

- 公开可获取
- 可用于研究
- 能拿到实际音频文件
- 能稳定生成 `track_id`
- 能映射到本项目 schema
- 能解释为什么它属于目标文化域

### 强烈推荐要求

- 有稳定元数据
- 有较清楚的乐器 / 风格 / 子标签
- 样本量至少能支持筛出 `100-200` 条较干净样本
- 录音不是明显的语音/解说/教学对白数据

### 暂不接入的来源

- license 不清晰
- 只能拿到文本描述、拿不到音频
- 数据内部文化语义极其混乱
- 单一说话人语音数据伪装成“音乐域”

---

## 5. V2 的目标规模

### 第一阶段目标规模

- 每域目标：`400-600` 首
- 5 域总量目标：`2000-3000` 首

### 为什么不一开始追求更大

- 你当前最缺的是“定义清楚、元数据统一、后续 PAL 能接”的数据底座
- 不是先追求 `5000+` 样本
- 一旦文化域内部不一致，量越大，后续清洗成本越高

### 验收线

- 每域最终保留样本不少于 `400`
- 文化域之间总体分布不严重失衡
- 全库 track_id 无重复

---

## 6. 目录结构（embedding 生成前）

推荐目录如下：

```text
storage/public/research_dataset_v2/
  README.md
  metadata_merged.csv
  metadata_merged.csv.merge_report.json
  source_inventory.csv
  germany/
    raw/
    audio/
    metadata.csv
    import_report.json
  china/
    raw/
    audio/
    metadata.csv
    import_report.json
  japan/
    raw/
    audio/
    metadata.csv
    import_report.json
  india/
    raw/
    audio/
    metadata.csv
    import_report.json
  turkey/
    raw/
    audio/
    metadata.csv
    import_report.json
  anglo_pop/
    raw/
    audio/
    metadata.csv
    import_report.json
```

说明：

- `raw/`：保留原始导入文件或下载日志
- `audio/`：标准化后供脚本消费的音频文件
- `metadata.csv`：该文化域单域元数据
- `metadata_merged.csv`：全局统一元数据

---

## 7. 统一 metadata schema

### 强制字段

- `track_id`
- `culture`
- `audio_path`
- `source_dataset`
- `source_split`
- `source_index`

### 强烈推荐字段

- `label`
- `substyle`
- `instrument`
- `language`
- `title`
- `artist`
- `duration_sec`
- `license`
- `license_note`

### 可选分析字段

- `region`
- `instrument_family`
- `era`
- `notes`
- `url`

### 字段设计原则

- `culture` 必须是本项目固定值之一，不允许直接写原始来源标签
- `label` 用于来源内标签保留，不要求跨源完全统一
- `substyle` 用来保存域内风格细分，避免信息在标准化时丢失
- `license` 和 `license_note` 必须保留，便于论文和数据卡说明

---

## 8. 从现在到 embedding 生成前的完整阶段

本阶段一共只做 5 步。

### 阶段 A：冻结文化域定义

要做什么：

- 确认 6 域
- 为每个域写一句“音乐域定义”
- 确认每域允许与不允许混入的内容

验收标准：

- 能用一句话向 reviewer 解释每个域是什么
- 不存在“国家名很宽，但实际样本很杂”的情况

### 阶段 B：建立 source inventory

要做什么：

- 给每个域列出候选来源
- 记录来源状态：`ready / provisional / blocked`
- 记录 license、样本量、风险、接入方式

验收标准：

- 每域至少有 1 个可执行来源
- 每域至少有 1 个备选来源

### 阶段 C：单域导入与单域 metadata 生成

要做什么：

- 使用现有工具链优先接入 Hugging Face 音频数据
- 对非 HF 源，手工整理为同样的 `audio/ + metadata.csv` 结构

验收标准：

- 每域都能独立生成 `metadata.csv`
- 每条样本都能追溯到具体来源与原始索引

### 阶段 D：全局合并与 schema 标准化

要做什么：

- 合并 6 域 metadata
- 规范字段名
- 修复重复 track_id
- 生成 `metadata_merged.csv`

验收标准：

- 合并后 schema 固定
- 无空 `audio_path`
- 无重复 `track_id`

### 阶段 E：前置质量检查

要做什么：

- 检查缺失字段
- 检查无效音频路径
- 检查极端短音频或非音乐样本
- 检查域内分布是否过于单一

验收标准：

- 所有样本都可以被后续 embedding 脚本读取
- 每域有足够样本
- 问题样本有日志

到这里为止，就进入后续的“统一生成 embedding 数据库”阶段。

---

## 9. 当前推荐的执行策略

### 当前建议

- 先保留 `research_dataset_v1` 不动，作为旧基线
- 新建 `research_dataset_v2`
- 先完成本文件范围内的 5 个阶段
- 只有在 `metadata_merged.csv` 和 source inventory 稳定后，才开始批量 Gemini embedding

### 不建议的做法

- 边下载边改文化域定义
- 只给某一个文化域单独换 backbone
- 在 merged metadata 未稳定前先做人类标注
- 先生成 embedding，之后再回头改 schema

---

## 10. 当前阶段的直接交付物

本阶段应至少产出以下文件：

- `docs/research_dataset_v2/RESEARCH_DATASET_V2_PRE_EMBEDDING_PLAN_CN.md`
- `docs/research_dataset_v2/RESEARCH_DATASET_V2_SOURCE_INVENTORY.csv`
- `docs/research_dataset_v2/RESEARCH_DATASET_V2_METADATA_SCHEMA.csv`

如果这三样东西稳定了，说明 `v2` 的“预嵌入阶段”就已经走上正轨。

---

## 11. 后续紧接着要做的事（不在本文件范围内）

一旦完成本文件范围内的工作，下一阶段就是：

1. 用同一个 backbone 统一生成全库 embedding  
2. 生成 `tracks.npz` 和 manifest  
3. 做 split / interaction / baseline  
4. 再接 DCAS 和 PAL

也就是说，这份文档是为“后面的统一 embedding 生成”打地基。
