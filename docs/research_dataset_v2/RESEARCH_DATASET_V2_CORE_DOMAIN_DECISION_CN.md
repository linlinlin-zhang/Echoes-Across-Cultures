# Research Dataset V2 主域决策说明

更新时间：2026-03-14  
适用范围：`research_dataset_v2` 的主实验文化域冻结决策。  
说明：这是在多轮来源与许可审计后的执行版决策文档，用来替代此前更偏探索状态的“5 域草案”。

---

## 1. 决策结果

当前决定将 `research_dataset_v2` 的主域收敛为 **6 个文化域**：

- `china`
- `germany`
- `japan`
- `india`
- `turkey`
- `anglo_pop`

这 6 个域中：

- `china`：保留，且当前最稳
- `germany`：保留，因外国语学院德语方向具有实际协同价值
- `japan`：保留，因外国语学院日语方向具有实际协同价值
- `india`：新增为强主域，因为来源清楚、规模充足、许可明确
- `turkey`：新增为主域，作为原有跨文化主线的延续
- `anglo_pop`：保留，作为现代流行锚点域

---

## 2. 为什么从原先方案调整为这 6 域

### 2.1 为什么不继续坚持 Spain

`spain / flamenco` 在学术上方向没有问题，但当前存在两个现实障碍：

- 最合适的 COFLA 音频不是“开箱即用”的开放音频主来源
- 可合法、可批量导入的开放 flamenco 音频目前更像小规模 seed set，而不是主域库

因此在当前工程阶段，把 Spain 继续放在主域里会明显拖慢 `v2-lite` 落地速度。

### 2.2 为什么 Germany 和 Japan 仍然保留

虽然 Germany 和 Japan 当前也不是最稳的两个域，但它们保留有现实价值：

- 与外国语学院后续挂靠方向一致
- 有潜在的跨学科协作价值（德语、日语专业）
- 对论文叙事和后续标注组织有帮助

因此这两个域不因为“当前不够完美”而放弃，而是通过适当收窄或放宽定义，争取补到可用状态。

### 2.3 为什么加入 India

India 当前是最强的新主域之一，理由很直接：

- 音频可得
- License 清楚
- 样本量足够
- 文化定义清楚
- 与现有跨文化音乐研究和 DCAS 原型链兼容性强

主来源：
- <https://huggingface.co/datasets/neerajaabhyankar/hindustani-raag-small>

### 2.4 为什么加入 Turkey

Turkey 的文化和方法相关性仍然很强：

- 与旧版本项目路线连续
- 在跨文化推荐叙事中很自然
- 和现有 DCAS 原型历史结果有衔接

虽然当前公开原始音频许可仍需进一步确认，但它仍值得保留为主域，并进入后续更细的来源审计。

---

## 3. 六个主域的正式工程定义

### 3.1 china

工程定义：
- `chinese_traditional`

解释：
- 中国传统器乐 / 民乐 / 民族器乐音频，不混现代华语流行

当前状态：
- `ready`

---

## 3.2 germany

工程定义：
- `german_folk`

解释：
- 优先采用德语民歌传统或相关民间录音资源
- 暂不与德语艺术歌曲、现代德语流行混合

当前状态：
- `provisional`

当前关键动作：
- 对 Europeana Westphalian collection 做 item-level rights 与音频可下载性审计

---

## 3.3 japan

工程定义：
- `japanese_music_audio`

解释：
- 出于当前开放来源限制，Japan 域从原先严格的 `japanese_traditional` 放宽到 `japanese_music_audio`
- 优先包含日本音乐音频，尽量保留传统日本方向，但不再把“必须纯传统”作为硬约束

这样做的原因：
- 如果坚持“传统日本音乐”硬定义，当前主来源规模不足
- 放宽后，Japan 域才更有机会补到可用状态

当前状态：
- `provisional`

---

## 3.4 india

工程定义：
- `hindustani_raag`

解释：
- 以 Hindustani classical / raag-based music 为主

当前状态：
- `ready`

---

## 3.5 turkey

工程定义：
- `turkish_music_audio`

解释：
- 先使用 Turkish music audio 的较宽工程定义
- 后续若来源许可与内容更明确，可收窄到 makam / traditional 子方向

当前状态：
- `provisional`

---

## 3.6 anglo_pop

工程定义：
- `anglo_pop`

解释：
- 英语现代流行锚点域

当前状态：
- `provisional but highly usable`

---

## 4. 主域状态分层

### 第一层：已可直接推进

- `china`
- `india`

### 第二层：通过额外审计后应优先补齐

- `germany`
- `japan`
- `turkey`
- `anglo_pop`

其中：

- `anglo_pop` 的问题主要是过滤策略，不是来源缺失
- `germany / japan / turkey` 的问题主要是来源冻结和许可确认

---

## 5. 当前执行策略

### 不再追求“先把 6 域全部完美冻结再开工”

更现实的做法是：

1. 立即推进 `china` 和 `india` 的导入 probe  
2. 并行推进 `anglo_pop` 的过滤方案设计  
3. 继续补 `germany / japan / turkey` 的来源审计  
4. 等至少 4 个域稳定后，再进入统一 embedding 生成前的冻结阶段

### 为什么这样更稳

因为它兼顾了：

- 主域数量足够表达“跨文化”
- 工程推进不会被单个难域卡死
- Germany / Japan 仍有保留和补强空间

---

## 6. 当前的 v2-lite 目标规模

考虑到你现在接受“小而干净”的版本，当前建议：

- 每域目标：`100-200`
- 六域总量目标：`800-1200`

说明：

- 不要求六域完全一样大
- 但建议尽量控制在 `2:1` 以内
- 如果某个域最终只能拿到 `80` 左右样本，应重新评估它是否适合进入主域

---

## 7. 一句话总结

`research_dataset_v2` 当前正式主域决策为：

**China + Germany + Japan + India + Turkey + Anglo-pop**

其中：

- China / India 已经接近可直接推进
- Germany / Japan / Turkey 继续补来源与许可
- Anglo-pop 保留为现代流行锚点并做后续过滤

这版方案兼顾了：

- 外院协同需求
- 工程可执行性
- 跨文化系统叙事
