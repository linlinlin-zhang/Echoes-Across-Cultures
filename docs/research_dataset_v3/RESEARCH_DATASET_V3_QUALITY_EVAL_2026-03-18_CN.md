# Research Dataset V3 质量评估

日期：2026-03-18

## 1. 总体结论

这版 `research_dataset_v3` 已经达到“可用、可训练、可进入项目全流程”的标准，但它更适合作为：

- 一个强可用的 `V3 baseline`
- 一个文化域对比实验主集
- 一个能支撑 embedding / DCAS / 推荐 / PAL 的工程数据集

而不是一个已经完全“语义纯净、元数据充分、来源无偏”的最终版数据集。

一句话判断：

- 工程可用性：高
- 文化语义纯度：中等
- 元数据完整度：中等偏低
- 推荐评测即插即用程度：中等
- 论文级“文化本体论严谨性”：中等偏弱，仍需继续打磨

## 2. 数据规模与平衡性

主表：

- `1049` 条
- `9` 个主文化域
- 总时长约 `94.71` 小时
- 文化域最大/最小规模比约 `1.43`

各主域条数：

- `china`: `146`
- `france`: `105`
- `germany`: `105`
- `great_britain`: `105`
- `india`: `108`
- `italy`: `105`
- `modern_english_pop`: `120`
- `russia`: `105`
- `turkey`: `150`

结论：

- 从训练角度看，类别平衡性是好的。
- 这版不会因为极端类不平衡而天然拖垮 `DCAS` 的文化分类头。

## 3. 音频质量与技术一致性

全表检查结果：

- 缺失音频文件：`0`
- `metadata.duration_sec` 与音频头信息不一致（>0.2s）：`0`
- 精确重复音频组：`2` 组，共 `4` 条，均出现在 `turkey`

总体音频格式：

- `.mp3`: `903`
- `.wav`: `146`

采样率分布：

- `44100`: `878`
- `16000`: `120`
- `48000`: `50`
- `96000`: `1`

声道分布：

- 双声道：`807`
- 单声道：`242`

结论：

- 技术上是可读、可处理的。
- 来源异构性很明显，但现有 embedding 代码会统一混成 mono 并重采样，因此对 embedding 阶段影响可控。
- 真正需要警惕的不是“能不能读”，而是“不同来源的制作条件是否会变成文化代理变量”。

## 4. 元数据完整度

全表填充率：

- `track_id / culture / audio_path / source_dataset / source_split / source_index / duration_sec / region / era`: `100%`
- `label`: `85.70%`
- `substyle`: `49.95%`
- `instrument`: `17.64%`
- `language`: `13.44%`
- `title`: `74.26%`
- `artist`: `66.63%`
- `license`: `68.06%`
- `license_note`: `74.26%`
- `instrument_family`: `13.92%`
- `url`: `50.05%`

结论：

- 对 embedding / 训练本身已经足够。
- 对“解释性评测、元数据驱动分析、下游论文报告”还不够强。
- 最弱的是 `language / instrument / instrument_family / artist / title` 的跨域一致性。

## 5. 各域质量画像

### 5.1 India

- `108` 条，约 `43.53` 小时
- `artist` 填充 `100%`
- `label` 填充 `100%`
- `substyle` 填充 `100%`
- 但 `language` 为 `0%`
- 中位时长约 `1028.73s`

判断：

- 传统语义强，主域价值高。
- 但由于曲目极长，而当前 embedding 默认只看前 `30s`，原始时长优势几乎不会被模型利用。

### 5.2 Turkey

- `150` 条，约 `1.25` 小时
- 时长几乎全是 `30s`
- `artist/title/language/license` 近乎全空
- 存在 `2` 组精确重复音频

判断：

- 工程上最好用，因为长度统一。
- 语义解释性最弱，元数据明显不足。
- 如果后续做解释性分析或论文附录，这一域会比较吃亏。

### 5.3 China

- `146` 条，约 `4.33` 小时
- `title`、`substyle` 填充高
- `artist` 只有约 `45.2%`
- `language` / `license` 约 `55.5%`
- `unique_artists_nonempty = 5`
- `era = traditional`

判断：

- 文化辨识度强，但目前几乎就是“传统中国音乐域”。
- 不是现代中文流行域。
- 艺术家多样性偏低，容易让模型更偏“曲种/声腔/器乐”而不是更广义的中国文化域。

### 5.4 五个 FMA 国家域

共同特点：

- `artist/title/license` 基本完整
- 时长普遍充足
- 都来自同一个大源 `Free Music Archive`
- `language` 覆盖偏低
- `era = mixed`

判断：

- 这是当前 V3 元数据最完整、最适合做国家层文化对照的一部分。
- 但要注意：它们本质上是“FMA 长尾独立音乐中的国家关联样本”，不是严格意义的国家文化真值语料。

### 5.5 Modern English Pop

- `120` 条，约 `1.00` 小时
- 几乎全是 `30s`
- `artist/title/license` 均为空
- `substyle`、`instrument` 完整
- 全部 `16k mono`

判断：

- 非常适合作为“现代英语流行对比基准”
- 但它同时也是最明显的“来源风格域”，不是元数据丰富的音乐目录域

## 6. 结构性风险

### 6.1 来源混杂会带来 source confound

当前 9 个主域来自 6 个数据源：

- `Free Music Archive`: `525`
- `bilal63/turkish_music_emotion_dataset`: `150`
- `vtsouval/mtg_jamendo_autotagging`: `120`
- `saraga_hindustani`: `108`
- `compmusic_jingju_acappella`: `81`
- `ccmusic-database/CTIS`: `65`

风险：

- 模型可能学到“来源差异”，而不全是“文化差异”
- 尤其是 `modern_english_pop / turkey / india / china` 几乎各自绑定独立来源

### 6.2 30 秒裁切会显著压缩原始语义

当前项目的 Gemini 和 CultureMERT embedding 都默认 `max_seconds = 30`。

这意味着：

- 原始总时长 `94.71` 小时
- 实际进入 embedding 的有效时长上限只有 `8.74` 小时
- 只保留了约 `9.23%` 的原始时长

各域受影响最明显的是：

- `india`: 只保留约 `2.07%`
- `great_britain`: 约 `7.51%`
- `france/germany/italy/russia`: 约 `10%`

结论：

- 如果保持当前 30 秒策略，这个数据集更准确地说是“1049 个 30 秒文化片段”，不是“94.7 小时完整音乐库”。
- 对项目主流程依然可用，但对长结构音乐的文化表示是不充分的。

### 6.3 推荐评测默认配置会失真

`synthesize_interactions.py` 默认按 `label` 精确分组采样。

问题是：

- `india` 的 `label` 非常细，`84` 个唯一 label 对应 `108` 条，`77%` 是 singleton
- `china` 也是高稀疏，`74` 个唯一 label 对应 `146` 条，`77%` 是 singleton
- `modern_english_pop` 也有较高稀疏度

实际默认合成交互结果：

- `180` 个用户
- 只有 `1600` 条交互
- 每用户交互数 `min/median/max = 1 / 2 / 50`

按文化域看更明显：

- `india`: 平均每用户 `1.45`
- `china`: 平均每用户 `1.4`
- `turkey`: 平均每用户 `50`

结论：

- 推荐链路能跑
- 但默认交互生成会极度偏向 `turkey`
- 如果不改这一步，推荐评测结论会被交互构造方式污染

## 7. 与当前项目后续流程的匹配度

### 7.1 Metadata -> Embedding

匹配度：高

原因：

- 必需列完整
- 音频路径有效
- 所有主表音频均可读取
- 抽样 dry-run 已验证不同文化域都能被 Gemini 预处理统一成 `16k / mono / 30s`

但要注意：

- 当前 Gemini 配置依然是 `30s` 裁切
- 每条 dry-run payload 约 `960,044` bytes
- 全表 raw wav payload 约 `0.94 GiB`
- base64 后约 `1.25 GiB`

### 7.2 Embedding -> Tracks NPZ -> DCAS Training

匹配度：高

原因：

- 主表 `9` 域都在 `100+`
- 类别失衡比只有 `1.43`
- `train.py` 只要求 `track_id / culture / embedding`
- 这版数据不会卡在训练入口

限制：

- `affect_label` 缺失，因此 affect 相关分支默认不会启用

### 7.3 Disentanglement Evaluation

匹配度：中等

原因：

- `culture` 和 `era` 可以直接评
- `source_dataset` 也能评

问题：

- 默认 `factors=culture,label` 时，`turkey` 会被整个排除，因为 `label` 缺失
- `instrument`、`language` 的覆盖太低，评测样本会掉得很厉害

更稳的 factor 组合应该是：

- `culture,era`
- `culture,source_dataset`
- `culture,substyle`（只在部分域上做）

### 7.4 Recommendation / PAL

匹配度：中等偏低，主要不是因为主表不行，而是因为默认辅助脚本设置不合适。

问题集中在：

- 默认 synthetic interactions 被稀疏 label 拖坏
- `artist/title` 缺失会降低 PAL 标注和人工复核体验

如果要把 V3 用在推荐和 PAL，建议先补两件事：

1. 增加一个 `coarse_label` 或 `genre_bucket`
2. 针对 `turkey / modern_english_pop` 补 `artist/title/license`

### 7.5 Waveform Style Transfer

匹配度：高

原因：

- 音频路径齐全
- 文件均可读
- 脚本本身会重采样并截断到更短窗口

但要注意：

- 不同来源的录音条件差异仍可能被迁移进去
- 这会让“文化风格迁移”与“录音条件迁移”部分混在一起

## 8. 最终判断

这版 V3 对你当前项目来说是**能直接推进后续主流程**的，尤其适合：

- 文化域 embedding 实验
- DCAS 训练
- 风格空间分析
- 初版推荐与 PAL 闭环

但它目前最像：

- 一个高可用的工程版文化域数据集

而不是：

- 一个已经完全清除了来源偏差和元数据缺口的终版研究语料

## 9. 我建议立刻做的三件事

1. 先用这版 V3 跑通 `embedding -> tracks.npz -> train -> eval` 主链路。
2. 在 metadata 上新增一个跨域统一的 `coarse_label`，替代默认稀疏 `label` 去生成 synthetic interactions。
3. 清理 `turkey` 的 2 组重复音频，并优先补 `turkey / modern_english_pop` 的 `artist/title/license`。

## 10. 原始评估产物

原始 JSON 指标输出：

- `tmp/v3_quality_report.json`

默认 synthetic interactions：

- `tmp/v3_interactions_default.csv`
