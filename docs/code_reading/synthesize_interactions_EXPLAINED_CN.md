# `synthesize_interactions.py` 代码说明

对应源码：[E:/Desktop/Echo/dcas/scripts/synthesize_interactions.py](E:/Desktop/Echo/dcas/scripts/synthesize_interactions.py)

## 1. 这个文件在数据链路里负责什么
这个脚本负责从 `metadata.csv` 合成“弱交互数据”。

项目当前并没有直接使用真实平台用户日志，因此它需要一种可控方式，构造出后续推荐评测可用的 `interactions.csv`。这个脚本就是在做这件事。

它生成的不是“真实用户行为”，而是“带结构假设的模拟交互”。

## 2. 它想模拟什么
脚本支持两种用户行为模式：

### `single_culture`
- 用户主要只消费自己文化域内的内容

### `mixed_culture`
- 用户有 home culture
- 但也会接触若干 secondary cultures

这两种模式分别对应：

- 更单一、更保守的文化消费
- 更混合、更跨文化的消费

## 3. 核心函数 `synthesize_interactions(...)`
### 参数解释
`metadata_csv`

- 输入 metadata 主表

`out_csv`

- 输出 interactions 文件路径

`users_per_culture`

- 每个文化域要合成多少个用户

`tracks_per_user`

- 每个用户最多抽多少条交互

`min_weight`

- 交互权重下界

`max_weight`

- 交互权重上界

`genre_column`

- 用哪一列作为“偏好流派”的参考字段
- 默认是 `label`

`mode`

- 交互生成模式
- 支持 `single_culture` 和 `mixed_culture`

`secondary_cultures`

- `mixed_culture` 模式下，每个用户还会额外接触多少个其他文化域

`home_share`

- `mixed_culture` 模式下，home culture 样本占总交互的比例

`seed`

- 随机种子
- 用于结果可复现

## 4. 它的整体流程
1. 读取 metadata
2. 检查是否有 `track_id / culture`
3. 按 `culture` 分组
4. 如果存在 `genre_column`，再按 `(culture, genre)` 分组
5. 初始化随机数生成器
6. 为每个文化域生成若干用户
7. 为每个用户采样一批 track
8. 写出 `user_id, track_id, weight`

## 5. 辅助函数流程说明
这个文件没有定义很多顶层辅助函数，但在 `synthesize_interactions(...)` 内部定义了两个关键小函数，它们刚好构成交互合成时的局部流程：

1. `_pick_rows(...)`
   - 先从候选池里抽样
   - 若存在 `preferred_genre`，优先从该 genre 子池里取
2. `_extend_unique(...)`
   - 如果前面没抽满目标数量，再从回退池里补足
   - 同时尽量避免重复 `track_id`

所以你可以把单个用户的采样过程理解成：

- 先“按偏好尽量精准地抽”
- 再“按唯一性要求尽量补齐”

这两个小函数虽然简单，但实际上决定了合成交互既不是纯随机，也不会因为样本池太小而很快塌掉。

## 6. 内部两个小函数各自做什么
### `_pick_rows(pool, n_pick, preferred_genre)`
作用：

- 从某个候选池里抽样
- 如果指定了 `preferred_genre` 且该文化内确实有这种 genre，则优先从这个 genre 子池里采样

意义：

- 它让“用户偏好”不只是随机散射，而是带一点“喜欢某类内容”的结构

### `_extend_unique(selected, fallback_pool, target_size)`
作用：

- 如果前面抽样没抽满目标数量，就从回退池里补齐
- 但只补未出现过的 `track_id`

意义：

- 保证采样尽量唯一
- 同时在数据较小的文化域里仍尽量凑够目标交互数

## 7. `single_culture` 模式是怎么工作的
在这种模式下：

1. 对每个文化域生成 `users_per_culture` 个用户
2. 每个用户只从本文化域的样本池里抽取
3. 如果该用户有 `preferred_genre`，优先从这个 genre 采样
4. 不足时再从同文化回退池补齐

这种模式模拟的是“本文化消费主导”的理想化用户。

## 8. `mixed_culture` 模式是怎么工作的
这是这个脚本更有意思的部分。

对于一个属于文化 `A` 的合成用户：

1. 先决定还会接触多少个其他文化域
2. 从其他文化域集合里随机挑选 `secondary_cultures` 个
3. 根据 `home_share` 决定 home culture 要抽多少条
4. 剩余条数在 secondary cultures 之间尽量平均分配
5. 每个次文化也会尝试随机挑一个偏好 genre
6. 最后再通过 `_extend_unique(...)` 补齐

这让交互数据不再是纯单域，而是带有某种跨文化曝光结构。

## 9. 权重 `weight` 是怎么来的
每条交互都会先从 `[min_weight, max_weight]` 中均匀随机采样一个基础权重。

如果是 `mixed_culture` 模式：

- home culture 的样本权重会乘以 `1.15`
- 非 home culture 的样本权重会乘以 `0.9`

这表达了一个弱假设：

- 用户对本文化内容的偏好平均会更强一点

这不是统计估计，而是任务设计中的先验。

## 10. 输出文件长什么样
输出 CSV 只有三列：

- `user_id`
- `track_id`
- `weight`

这是一个非常克制的设计，目的就是让下游训练和验证尽量简单。

## 11. 这个脚本的重要性
如果没有真实交互日志，推荐任务就没有监督信号。

这个脚本的意义在于：

- 先构造一个可控、可复现、结构清楚的弱监督交互层
- 让项目能够评估推荐模型或文化泛化模型

它本质上是“从内容数据构造任务监督”的一层。

## 12. 局限与注意点
### 局限 1：这不是现实用户行为
它只是结构化模拟，不应被误解为真实世界分布。

### 局限 2：偏好结构比较简单
主要依赖文化域和 genre，未建模更复杂的长期偏好。

### 局限 3：权重是随机先验，不是点击/停留时长之类的真实行为量

## 13. 面试时怎么讲
可以这样说：

“`synthesize_interactions.py` 负责从内容侧 metadata 合成弱交互监督。它按文化域生成合成用户，支持单文化和混合文化两种消费模式；在每个模式下又通过 genre 偏好、home culture 占比和随机权重来构造更有结构的交互分布。它不是在还原真实用户日志，而是在为推荐与泛化实验提供一个可控、可复现的任务层。”

## 14. 建议重点看哪里
优先看：

1. `mode` 两种分支
2. `_pick_rows(...)`
3. `_extend_unique(...)`
4. `weight` 在 `mixed_culture` 模式下的调整逻辑

这几处就构成了整个脚本的设计核心。
