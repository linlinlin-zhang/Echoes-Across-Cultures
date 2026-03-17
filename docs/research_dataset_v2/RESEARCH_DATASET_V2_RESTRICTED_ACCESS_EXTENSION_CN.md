# Research Dataset V2 受限研究使用扩展说明

更新时间：2026-03-16  
适用范围：`research_dataset_v2` 在“允许受限但合法的研究使用、不分发原始音频”的前提下，对候选文化域进行扩展与治理的执行说明。

---

## 1. 为什么要增加这条策略

如果我们只接受“完全开源、可直接下载、许可极清楚”的音频源，那么当前最稳的主域主要集中在：

- `china`
- `india`
- `anglo_pop`
- `germany`

这会让项目推进更快，但会限制跨文化覆盖。

一旦我们允许以下条件成立：

- 数据来源合法
- 可用于非商业学术研究
- 原始音频不再对外分发
- 论文和文档中明确写出访问条件与限制

那么原本较难的文化域会明显变得更可做，尤其是：

- `japan`
- `spain`
- `france`
- `korea`

---

## 2. 新增或重估后的文化域

### 2.1 Japan

当前最值得继续推进的两条线：

- `National Diet Library Historical Recordings Collection`
- `tts-dataset/japanese-singing-voice`（gated）

Japan 的现实结论是：

- 如果坚持只用完全开放直链音频，Japan 很难尽快变成强主域
- 如果接受“受限访问 + 不再分发原始音频”，Japan 会明显更容易补到可用状态

因此 Japan 已经从“边缘候选”升级为：

- **restricted-access priority domain**

---

### 2.2 Spain

`Corpus COFLA` 仍然是西班牙 / flamenco 方向最合适的学术源。

在旧标准下，它因为音频需要申请访问而被视为阻塞；
在新标准下，它已经可以被视为：

- **restricted-access candidate**

前提是：

- 研究用途访问被明确授予
- 访问条件被写入项目文档
- 原始音频不再对外分发

---

### 2.3 France

France 以前之所以一直进不了主域，不是因为完全没有资源，而是因为：

- 缺少一个像 HF 音频数据集那样的“开箱即用开放源”

在新标准下，`INA research dataset project` 让 France 重新变成一个现实候选：

- 可注册
- 面向研究项目访问
- 更适合作为“受限研究使用域”

因此 France 现在不再是“完全不推荐”，而是：

- **restricted-access candidate**

---

### 2.4 Korea

Korea 这轮最大的变化来自官方公共数据与国乐相关档案源：

- `National Gugak Center Korean traditional music information API`
- `National Gugak Center archive file data`

它们仍然需要做 direct-audio 样本审计，但已经说明：

- Korea 并不是“找不到源”
- 而是更适合以“官方平台 / API / 研究使用”方式接入

因此 Korea 也应该被纳入：

- **restricted-access candidate**

---

## 3. 现在项目里应该怎么用这条策略

推荐分层如下：

### 3.1 公开直取主域

这些域优先用于先跑通 embedding 数据库与 DCAS 主链：

- `china`
- `india`
- `anglo_pop`
- `germany`

### 3.2 受限研究扩展域

这些域优先进入“申请 / 审计 / 协议记录”流程：

- `japan`
- `spain`
- `france`
- `korea`

### 3.3 使用原则

- 原始音频只在本地受控目录保存
- 不对外发布原始音频
- 对外只发布：
  - metadata
  - 处理脚本
  - 统计信息
  - 可公开的 embeddings（如许可允许）
- 每个受限源都必须保存：
  - 来源页面
  - 访问日期
  - 许可或访问说明截图/文本
  - 是否允许再分发

---

## 4. 对论文叙事的影响

这条策略让我们可以更稳地说：

- 项目不只依赖少数完全开放源
- 数据路线同时覆盖“公开可复现音频源”和“受限但合法的研究音频源”
- 在跨文化推荐场景下，数据治理本身就是系统设计的一部分

这会比简单地“偷偷用非开源音乐但不说明来源”稳很多。

---

## 5. 当前建议

如果后续 Japan 的 gated 访问获批，或 Spain / France / Korea 的研究访问通道被确认，建议将这些域纳入 `v2` 的第二阶段扩展计划，而不是继续只围绕完全开放直取域做数据集扩展。
