
---

## 4. 实验设置

### 4.1 数据集

#### 4.1.1 V4 main 数据集

我们构建了V4跨文化音乐数据集，包含1,122首曲目，涵盖10种文化传统和8个数据来源。完整统计如表1所示。

**表1：V4 main 数据集详细统计**

| 文化 | 曲目数 | 主要来源 | 来源曲目数 |
|---|---|---|---|
| turkey | 150 | turkish_music_emotion | 150 |
| china | 145 | CTIS(65) + OpenCpop(50) + jingju_acappella(30) | 145 |
| modern_english_pop | 120 | mtg_jamendo | 120 |
| india | 108 | saraga_hindustani | 108 |
| france | 105 | Free Music Archive | 105 |
| germany | 105 | Free Music Archive | 105 |
| great_britain | 105 | Free Music Archive | 105 |
| italy | 105 | Free Music Archive | 105 |
| russia | 105 | Free Music Archive | 105 |
| indonesia | 74 | gamelan(55) + FMA(19) | 74 |
| **总计** | **1,122** | | |

**来源分布**：

| 来源 | 曲目数 | 占比 |
|---|---|---|
| Free Music Archive | 544 | 48.5% |
| turkish_music_emotion | 150 | 13.4% |
| mtg_jamendo | 120 | 10.7% |
| saraga_hindustani | 108 | 9.6% |
| CTIS | 65 | 5.8% |
| gamelan | 55 | 4.9% |
| OpenCpop | 50 | 4.5% |
| jingju_acappella | 30 | 2.7% |
| **总计** | **1,122** | **100%** |

**合成交互数据**：9,600条交互记录，240个用户，每用户40条交互，100%曲目覆盖率。

**源混淆度**：weighted\_source\_predictability\_from\_culture = 0.911765，表明文化标签能高度预测数据来源，这是一个严重的混杂因素。

#### 4.1.2 routeA\_small 子集

routeA\_small 是一个高混淆度子集，用于测试框架在极端偏差场景下的鲁棒性：

- 640首曲目，4种文化，4个来源
- 96个用户，384个评估对
- 源混淆度 = 1.0（完美可预测）

### 4.2 骨干模型与训练配置

**骨干模型**：
- CultureMERT \cite{kanatas2025culturemert}：768维嵌入，针对跨文化音乐持续预训练
- Gemini：Google的多模态基础模型，用于对比实验

**训练配置**：
- 优化器：AdamW
- 学习率：0.002
- 批次大小：128
- 训练轮数：10
- 随机种子：42
- 正则化预热：3个epoch
- 成对约束：从epoch 2开始，预热2个epoch
- 排序约束：从epoch 4开始，预热2个epoch

### 4.3 基线方法

我们在每个基准上评估了以下基线方法：

1. **popularity**：基于全局流行度的推荐，作为最简单的基线
2. **cosine**：基于预训练嵌入的余弦相似度推荐
3. **knn**：K近邻推荐，使用预训练嵌入空间中的距离
4. **lightfm\_like**：类似LightFM的混合推荐，结合内容和协同过滤
5. **bpr\_mf**：贝叶斯个性化排序矩阵分解 \cite{rendle2009bpr}
6. **bpr\_two\_stage\_hybrid**：两阶段BPR混合推荐
7. **bpr\_listwise\_hybrid**：Listwise BPR混合推荐 \cite{qu2025listwise}
8. **bpr\_lambdamart\_hybrid**：LambdaMART增强的BPR混合推荐 \cite{burges2010lambdamart}
9. **dcas\_full\_ot**：完整DCAS模型，仅OT检索，无重排序
10. **dcas\_full\_ot\_calibrated\_target**：DCAS目标校准操作点
11. **dcas\_full\_ot\_calibrated\_minor**：DCAS少数群体聚焦操作点

### 4.4 指标与统计协议

**评估指标**：

- **惊喜度（Serendipity）**：度量推荐结果的意外性和有用性，计算为用户未交互但相关的推荐曲目比例
- **校准KL散度（Calib KL）**：度量推荐分布与用户历史偏好分布之间的KL散度 \cite{steck2018calibrated}，值越低表示校准越好
- **少数文化覆盖率（Minority@k）**：推荐列表Top-k中来自少数文化（曲目数 < 100）的曲目比例

**统计协议**：

- Bootstrap置信区间：1000次重采样
- 排列检验：1000次随机排列
- Bonferroni校正：$\alpha = 0.05/10 = 0.005$（10次比较）
- 典型显著性水平：$p = 0.004975124$（1/201）
