# Domain Profiles and Korea Search Guide (2026-03-17)

本文档回答两个问题：

1. 当前已基本确定的文化域，大概都是什么类型的音乐
2. Korea 文化域接下来应该怎么继续搜

## 1. China

当前来源：
- CTIS:
  <https://huggingface.co/datasets/ccmusic-database/CTIS>

当前在主版中的大致类型：
- **中国传统器乐**
- 明显不是流行音乐，也不是西方意义上的“古典乐库”
- 更像传统民族器乐 / 戏曲伴奏器乐 / 地方乐种相关器乐集合

从本地 metadata 能看到：
- `cname` 里大量是乐器名，例如：
  - 黔胡
  - 壳仔弦
  - 高音板胡
  - 渔胡
  - 二胡
  - 四弦

所以 China 域最准确的理解是：
- **instrument-centered Chinese traditional music domain**

## 2. India

当前来源：
- Hindustani raag small:
  <https://huggingface.co/datasets/neerajaabhyankar/hindustani-raag-small>

当前在主版中的大致类型：
- **Hindustani classical / raag-based music**
- 这是最接近“古典体系”的一个域
- 不是流行音乐

从本地 metadata 能看到：
- `label` 全部是 raag 名称，例如：
  - AheerBhairav
  - AlhaiyaBilawal
  - Bhairav
  - Bageshree
  - Bhoopali

所以 India 域最准确的理解是：
- **North Indian classical raga domain**

## 3. Anglo-pop

当前来源：
- HF parquet 版 MTG-Jamendo:
  <https://huggingface.co/datasets/vtsouval/mtg_jamendo_autotagging>
- 原始项目页：
  <https://mtg.github.io/mtg-jamendo-dataset/>

当前在主版中的大致类型：
- **现代西方流行 / 流行摇滚 / 电流行 / 民谣流行混合域**
- 它是当前数据集里唯一明确的“现代流行锚点”

从本地 metadata 能看到：
- `label` 里大量是：
  - `pop`
  - `poprock`
  - `electronic`
  - `folk`
  - `dance`
  - `indie`
- `instrument` 里常见：
  - guitar
  - piano
  - drums
  - synthesizer
  - bass
- `mood_theme` 里常见：
  - relaxing
  - emotional
  - commercial
  - energetic
  - cool

所以 Anglo-pop 域最准确的理解是：
- **pop-like Western commercial music anchor domain**

## 4. Kazakhstan

当前来源：
- Kazakh traditional audio:
  <https://huggingface.co/datasets/rtrk/kazakh-traditional-audio>

当前在主版中的大致类型：
- **哈萨克传统器乐 / kui**
- 不是流行音乐
- 更接近传统器乐语料

从本地 metadata 能看到：
- `label` 全部是 `kui`
- `language` 全部是 `kk`
- 标题都是哈萨克传统曲目名称

所以 Kazakhstan 域最准确的理解是：
- **Kazakh traditional kui domain**

## 5. Germany

当前来源：
- Europeana Westphalian Folk Song and Sound Archive:
  <https://www.europeana.eu/es/collections/organisation/1815-westphalian-folk-song-and-sound-archive>

当前在主版中的大致类型：
- **德国民歌 / 社区歌曲 / 仪式与节庆歌曲 / 行进曲 / 部分宗教/圣歌**
- 不是现代流行音乐
- 也不是一个纯古典乐曲库
- 更像档案型地方民俗音乐语料

从本地 metadata 的标题能看到：
- wedding / shooting festival / folk songbook / marches / sacred song 等线索
- 例如：
  - `Ave Maria; Musica sacra, Marienlob`
  - `Marsch und Volkstänze aus Geseke`
  - 各类 handwritten songbook entries

所以 Germany 域最准确的理解是：
- **archival German folk and community song domain**

## 6. Norway

当前来源：
- HF2 Hardanger fiddle dataset:
  <https://huggingface.co/datasets/Bots4M/HF2-Hardanger-fiddle-dataset>

当前作为新增候选域的大致类型：
- **挪威民间音乐 / Hardanger fiddle**
- 不是现代流行
- 也不是大型综合古典库
- 是一个相对窄但很干净的民间器乐域

从本地 metadata 能看到：
- `119` 条音频
- `label` 主要是：
  - `archival`
  - `processed`
- 一部分曲目还有情绪变体：
  - `original`
  - `happy`
  - `angry`
  - `sad`
  - `tender`

所以 Norway 域最准确的理解是：
- **Norwegian Hardanger fiddle folk domain**

## 7. 当前主版整体的音乐结构

如果把当前 5 域主版和 Norway 这个候选域一起看，大致可以分成：

- **传统器乐域**
  - China
  - Kazakhstan
  - Norway
- **古典体系域**
  - India
- **档案型民歌/社区歌曲域**
  - Germany
- **现代流行锚点**
  - Anglo-pop

所以这套数据并不是：
- “每个国家都随机抽点歌”

而是：
- **用不同类型的文化音乐域去构成一个跨文化推荐环境**

## 8. Korea 应该怎么继续搜

Korea 当前最值得搜的，不是随便搜 “Korean music dataset”，而是这几条更具体的方向：

### 推荐方向 A：gugak / minyo / digital audio

建议关键词：
- `국악 디지털 음원`
- `국악 음원 OpenAPI`
- `전국8도민요 MR`
- `국악 MR 정보조회서비스`
- `국립무형유산원 음원자료`

### 推荐方向 B：官方机构

重点机构：
- `국립국악원`
- `국가유산청`
- `국립무형유산원`
- `국악방송`

### 推荐方向 C：你需要重点核查什么

如果你打开 Korea 相关页面，最重要的不是先看介绍，而是看：

1. 有没有 **真实音频对象**
   - mp3
   - wav
   - streaming audio

2. 有没有 **可下载或稳定访问的链接**

3. 许可写的是不是：
   - `KOGL Type 1`
   - `이용허락범위 제한 없음`
   - 或其他明确允许研究/教育使用的条款

4. 是不是能凑出：
   - 至少 `100+` 条
   - 而且不是纯 MR / 纯伴奏

## 9. Korea 当前最值得追的来源

### 9.1 全国 8 道民谣 MR
- <https://www.data.go.kr/data/15098241/fileData.do>

目前已知：
- 页面公开说明有 `105` 条
- 文化域辨识度强

需要你继续确认：
- 是否可实际下载
- 是完整民谣音频还是主要偏伴奏

### 9.2 国乐 MR 信息查询服务
- <https://www.data.go.kr/data/15098387/openapi.do>

目前已知：
- 是国乐相关音源查询 API

需要你继续确认：
- API 返回里是不是有真实音频 URL

### 9.3 国立无形遗产院音源资料
- <https://www.data.go.kr/data/15094324/openapi.do>

目前已知：
- 更偏馆藏/资料型

需要你继续确认：
- 能不能定位到可播放或可下载的音频对象

## 10. 我对 Korea 的建议

如果你要自己继续查，我建议按这个顺序：

1. 先看 `15098241`
2. 再看 `15098387`
3. 最后看 `15094324`

判断标准很简单：
- 能拿到真实音频
- 能形成 `100+`
- 许可写得清楚

只要满足这三条，Korea 就值得推进成正式扩展域。
