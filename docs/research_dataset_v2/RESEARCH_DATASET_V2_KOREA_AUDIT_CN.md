# Research Dataset V2 Korea 文化域专项审计

更新时间：2026-03-17

## 1. 结论先行

这次对 `korea` 的重新审计结果比之前明显更强。

最重要的新发现不再只是泛泛的“韩国传统音乐信息库”，而是三条可以互相支撑的路线：

- **National Gugak Center digital audio site (`digitaleum`)**  
  <https://www.gugak.go.kr/digitaleum>
- **National Gugak Center digital audio OpenAPI**  
  <https://www.data.go.kr/data/15097515/openapi.do>
- **National Gugak Center archive / metadata support lines**  
  <https://www.data.go.kr/en/data/3062269/openapi.do>  
  <https://www.data.go.kr/en/data/15083294/fileData.do>

从当前公开页面可以直接确认的信息看，它具备以下几个特征：

- 类型明确是 **digital audio**
- 所属机构是 **National Gugak Center**
- 页面标注 **application automatic approval**
- 开发阶段可用流量是 **10,000**
- 页面给出了公共版权标识 **KOGL Type 1**

来自公开检索的额外证据包括：

- 2021 年官方上线报道，明确写有 **16,721** 条高音质国乐数字音源，并提供 `wav / mp3`  
  <https://www.koreatimes.co.kr/www/culture/2024/04/135_315698.html>
- 站点公告与媒体报道中明确写到：
  - 提供单음与 악구音源
  - 可供创作与教育利用
  - 标明出处后可自由使用  
  参考：  
  <https://www.newsis.com/view/NISX20210901_0001569332>  
  <https://www.yna.co.kr/view/AKR20210901087800005>

这意味着 `korea` 已经从“需要强假设的受限访问候选”，提升为：

- **非常值得优先推进的新增文化域**

它现在的状态介于：

- `norway` 这种最省力的开放下载域
- `spain / japan / france` 这种需要更多访问协调的受限域

之间。

更准确地说：

- `korea` 现在已经是一个 **强 OpenAPI 候选域**
- 还差最后一步：**拿到 service key 后做 item-level sample audit**

---

## 2. 这次最关键的来源

### 2.1 National Gugak Center digital audio site (`digitaleum`)

链接：

- <https://www.gugak.go.kr/digitaleum>

为什么这条线非常重要：

- 它直接指向数字音源站，而不是抽象的目录页
- 外部报道已经说明站内包含 `wav / mp3` 高音质音源
- 如果站内下载/播放对象可以稳定解析，它甚至可能比 OpenAPI 更适合作为实际抓取入口

当前不足：

- 仍需 item-level 下载/播放对象审计
- 页面结构可能存在动态加载或反爬策略

当前判断：

- **这是 Korea 目前最值得优先做样本审计的公开入口。**

---

### 2.2 National Gugak Center digital audio OpenAPI

链接：

- <https://www.data.go.kr/data/15097515/openapi.do>

当前页面直接能确认的关键信息：

- 标题含义：`국립국악원_국악디지털음원`
- 访问形态：OpenAPI
- 审批方式：`신청즉시승인`
- 开发流量：`10000`
- 版权/使用标识：`공공저작물_유형1`

为什么这条线重要：

- 它不是泛 metadata API，而是直接指向 **digital audio**
- 它来自官方国乐机构
- 页面看起来已经足够接近“可实际拉取样本”的状态

当前不足：

- 还未验证返回字段是否直接给出音频 URL、播放对象或可解析的 archive key
- 还未验证实际可用条目数量

当前判断：

- **这是 Korea 目前最值得推进的主来源。**

---

### 2.3 National Gugak Center Korean traditional music information API

链接：

- <https://www.data.go.kr/en/data/3062269/openapi.do>

页面当前能确认：

- 说明涉及 multimedia / image / audio / video
- 页面文案包含 `The use permission range is limitless`

为什么它有价值：

- 它比纯静态目录更接近可编程接入
- 即便数字音频 API 后面只给出基础字段，这条线也可能作为 metadata enrichment 使用

当前不足：

- 更像一个广义文化信息 API，而不是明确的音频主语料 API

当前判断：

- **适合作为 Korea 的辅助 API，而不是第一主源。**

---

### 2.4 National Gugak Center archive file data

链接：

- <https://www.data.go.kr/en/data/15083294/fileData.do>

页面当前能确认：

- 描述中提到 archive materials transferred to the center
- 页面文案中明确写到 `copyright-free`

为什么它有价值：

- 它提供了额外的馆藏侧证据
- 对 Korea 域的合法性和文化可信度很有帮助

当前不足：

- 仍然需要验证 file data 和实际音频对象之间的映射关系

当前判断：

- **适合作为 Korea 的 backup / archive support line。**

---

## 3. Korea 和其他扩展域相比怎么样

### 对比 France

- `France` 现在主要还是 `INA research dataset project`
- 更像研究访问平台，不像已收敛的音乐音频语料

相比之下：

- `Korea` 现在已经出现了更接近直接接入的 **digital audio API**

结论：

- **Korea 明显优先于 France**

---

### 对比 Spain

- `Spain` 最强的是 `Corpus COFLA`
- 学术上很强，但还是要研究访问申请

相比之下：

- `Korea` 的 official OpenAPI 路线更像能快速进入工程验证

结论：

- 如果目标是“先补一个能真正推进的域”，**Korea 比 Spain 更快**
- 如果目标是“文化辨识度和学术故事更强”，Spain 仍然很有价值

---

### 对比 Norway

- `Norway` 的 Hardanger fiddle 数据集仍然是最省力的开放新增域

相比之下：

- `Korea` 的文化覆盖更有代表性
- 但工程接入仍然比 Norway 稍复杂

结论：

- **Norway 更适合立刻加**
- **Korea 更适合加成“有说服力的扩展域”**

---

## 4. 当前最合理的定位

如果现在要决定 `korea` 在 V2 里的位置，我建议这样定：

- 当前状态：`strong_public_audio_and_openapi_candidate`
- 推荐位置：**扩展域第一梯队**
- 是否立刻升为主域：**先不直接升**
- 触发升级条件：
  - 跑通 `digitaleum` 小批 sample audit，或拿到 OpenAPI service key
  - 跑通小批 sample audit
  - 确认存在稳定音频字段 / URL / archive playback object

---

## 5. 下一步怎么做

最自然的执行顺序是：

1. 优先对 `digitaleum` 站内条目做小批样本审计
2. 同时在 `data.go.kr` 申请或确认 `15097515` 的 API key / service key
3. 拉取一批 sample records
3. 检查：
   - 是否真的返回音频对象
   - 返回字段是否稳定
   - 是否足够筛出 `100-250` 条
4. 如果通过，再决定：
   - 作为 `v2-main+1` 的第 6 域
   - 或先作为扩展实验域

---

## 6. 当前建议

如果你现在就想决定这条线是否值得继续，我的判断是：

**值得。**

更具体地说：

- `Korea` 现在已经比 `France` 更值得优先补
- 在“受限或半开放域”里，它的可执行性已经接近 `Spain`
- 在工程推进优先级上，它应该排在：
  - `Norway` 之后
  - `Spain / France` 之前

所以如果下一步要新增域，我建议优先顺序是：

1. `norway`
2. `korea`
3. `spain`
