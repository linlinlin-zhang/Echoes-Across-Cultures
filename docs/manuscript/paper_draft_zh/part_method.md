
---

## 3. 方法

### 3.1 问题形式化定义

设 $\mathcal{X} = \{x_i\}_{i=1}^N$ 为包含 $N=1,122$ 首曲目的音乐集合，每首曲目 $x_i$ 通过预训练音频编码器（如CultureMERT \cite{kanatas2025culturemert} 或 MERT \cite{li2023mert}）提取为768维嵌入向量 $e_i \in \mathbb{R}^{768}$。每首曲目关联一组元数据属性：

- 文化标签 $c_i \in \mathcal{C} = \{1, \ldots, 10\}$，涵盖turkey(150)、china(145)、modern\_english\_pop(120)、india(108)、france(105)、germany(105)、great\_britain(105)、italy(105)、russia(105)、indonesia(74)共10种文化
- 来源标签 $s_i \in \mathcal{S} = \{1, \ldots, 8\}$，涵盖Free Music Archive(544)、turkish\_music\_emotion(150)、mtg\_jamendo(120)、saraga\_hindustani(108)、CTIS(65)、gamelan(55)、OpenCpop(50)、jingju\_acappella(30)共8个来源
- 情感标签 $a_i \in \mathcal{A} = \{1, \ldots, 8\}$（8类情感）

此外，我们有一组合成交互数据 $\mathcal{D}_{\text{int}} = \{(u_j, \mathcal{I}_j)\}_{j=1}^{M}$，其中 $u_j$ 表示第 $j$ 个用户，$\mathcal{I}_j \subset \mathcal{X}$ 为用户 $j$ 的交互曲目集合。在我们的设置中，$M=240$ 个用户，共9,600条交互记录，100%曲目覆盖率。

**目标**：学习一个映射 $f: \mathbb{R}^{768} \to \mathbb{R}^{d_c} \times \mathbb{R}^{d_s} \times \mathbb{R}^{d_a}$，将输入嵌入 $e$ 分解为三个独立的潜变量：

$$f(e) = (z_c, z_s, z_a)$$

其中 $z_c \in \mathbb{R}^{32}$ 表示内容因子（曲目的音乐语义身份），$z_s \in \mathbb{R}^{32}$ 表示风格因子（文化传统与录音特征），$z_a \in \mathbb{R}^{16}$ 表示情感因子（跨文化情感维度）。

在此基础上，构建一个推荐函数 $R: \mathcal{U} \times \mathcal{X} \to \mathbb{R}$，为用户 $u$ 生成排名靠前的曲目列表，同时满足以下多目标约束：

1. **惊喜度最大化**：推荐用户未预期但相关的曲目
2. **校准度最大化**：推荐分布与用户历史偏好分布相匹配
3. **少数文化覆盖率最大化**：确保非主导文化曲目获得公平曝光
4. **内容保真度**：推荐结果在音乐语义上保持一致性

### 3.2 DCAS框架架构

DCAS框架由以下核心组件构成：

#### 3.2.1 共享编码器

编码器 $E_\phi: \mathbb{R}^{768} \to \mathbb{R}^{256}$ 是一个三层MLP，将768维预训练嵌入压缩为256维共享隐表示 $h$：

$$h = E_\phi(e) = \text{MLP}_{\phi}(e; \text{hidden\_dim}=256, \text{depth}=3, \text{dropout}=0.1)$$

具体地，编码器由三层线性变换组成，每层后接ReLU激活和Dropout（比率0.1）：

$$h^{(1)} = \text{Dropout}(\text{ReLU}(W^{(1)} e + b^{(1)})), \quad W^{(1)} \in \mathbb{R}^{256 \times 768}$$
$$h^{(2)} = \text{Dropout}(\text{ReLU}(W^{(2)} h^{(1)} + b^{(2)})), \quad W^{(2)} \in \mathbb{R}^{256 \times 256}$$
$$h = h^{(3)} = \text{Dropout}(\text{ReLU}(W^{(3)} h^{(2)} + b^{(3)})), \quad W^{(3)} \in \mathbb{R}^{256 \times 256}$$

#### 3.2.2 三因子高斯潜空间头

从共享隐表示 $h$ 出发，三个独立的高斯头分别输出各因子的均值和方差：

**内容头（Content Head）**：
$$\mu_c = W_{\mu_c} h + b_{\mu_c}, \quad W_{\mu_c} \in \mathbb{R}^{32 \times 256}$$
$$\log \sigma_c^2 = W_{\sigma_c} h + b_{\sigma_c}, \quad W_{\sigma_c} \in \mathbb{R}^{32 \times 256}$$
$$z_c = \mu_c + \sigma_c \odot \epsilon_c, \quad \epsilon_c \sim \mathcal{N}(0, I_{32})$$

**风格头（Style Head）**：
$$\mu_s = W_{\mu_s} h + b_{\mu_s}, \quad W_{\mu_s} \in \mathbb{R}^{32 \times 256}$$
$$\log \sigma_s^2 = W_{\sigma_s} h + b_{\sigma_s}, \quad W_{\sigma_s} \in \mathbb{R}^{32 \times 256}$$
$$z_s = \mu_s + \sigma_s \odot \epsilon_s, \quad \epsilon_s \sim \mathcal{N}(0, I_{32})$$

**情感头（Affective Head）**：
$$\mu_a = W_{\mu_a} h + b_{\mu_a}, \quad W_{\mu_a} \in \mathbb{R}^{16 \times 256}$$
$$\log \sigma_a^2 = W_{\sigma_a} h + b_{\sigma_a}, \quad W_{\sigma_a} \in \mathbb{R}^{16 \times 256}$$
$$z_a = \mu_a + \sigma_a \odot \epsilon_a, \quad \epsilon_a \sim \mathcal{N}(0, I_{16})$$

其中重参数化技巧（reparameterization trick）\cite{kingma2014auto} 使梯度能够通过随机采样操作反向传播。

#### 3.2.3 解码头

解码器 $D_\psi: \mathbb{R}^{80} \to \mathbb{R}^{768}$ 将拼接的潜变量 $z = [z_c; z_s; z_a] \in \mathbb{R}^{80}$ 重建回原始嵌入空间：

$$\hat{e} = D_\psi(z_c, z_s, z_a) = \text{MLP}_{\psi}([z_c; z_s; z_a])$$

重建损失使用均方误差（MSE）：

$$\mathcal{L}_{\text{recon}} = \|e - \hat{e}\|_2^2$$

#### 3.2.4 辅助判别头

**文化判别器（带梯度反转）**：
$$p(c|z_a) = \text{Softmax}(\text{MLP}_{\text{culture}}(\text{GRL}(z_a)))$$
$$\mathcal{L}_{\text{domain}} = -\frac{1}{N} \sum_{i=1}^N \log p(c_i | z_a^{(i)})$$

其中GRL在前向传播时为恒等映射 $f(x) = x$，在反向传播时将梯度乘以 $-\lambda_{\text{grl}}$。这鼓励 $z_a$ 学习文化不变的表征。

**来源判别器**：
$$p(s|z_a) = \text{Softmax}(\text{MLP}_{\text{source}}(z_a))$$
$$\mathcal{L}_{\text{source}} = -\frac{1}{N} \sum_{i=1}^N \log p(s_i | z_a^{(i)})$$

**情感分类器**：
$$p(a|z_a) = \text{Softmax}(\text{MLP}_{\text{affect}}(z_a))$$
$$\mathcal{L}_{\text{affect}} = -\frac{1}{N} \sum_{i=1}^N \log p(a_i | z_a^{(i)})$$

### 3.3 三阶段课程训练策略

DCAS采用三阶段课程训练策略，逐步增强解耦质量和跨文化对齐能力。

#### 3.3.1 阶段一：因子化变分预训练（Epoch 1-10）

阶段一的目标是学习基本的解耦表征。总损失为：

$$\mathcal{L}_{\text{stage1}} = \mathcal{L}_{\text{recon}} + \beta_{\text{KL}} \mathcal{L}_{\text{KL}} + \lambda_{\text{cov}} \mathcal{L}_{\text{cov}} + \lambda_{\text{tc}} \mathcal{L}_{\text{tc}} + \lambda_{\text{hsic}} \mathcal{L}_{\text{hsic}} + \lambda_{\text{contrastive}} \mathcal{L}_{\text{contrastive}}$$

**KL散度正则化**（$\beta_{\text{KL}} = 1.0$）：
$$\mathcal{L}_{\text{KL}} = \sum_{f \in \{c,s,a\}} \frac{1}{2} \sum_{j=1}^{d_f} \left(\mu_{f,j}^2 + \sigma_{f,j}^2 - \log \sigma_{f,j}^2 - 1\right)$$

**协方差正则化**（$\lambda_{\text{cov}} = 0.05$）：
$$\mathcal{L}_{\text{cov}} = \sum_{f \neq g \in \{c,s,a\}} \|\text{Cov}(z_f, z_g)\|_F^2$$

**总相关正则化**（$\lambda_{\text{tc}} = 0.05$）：
$$\mathcal{L}_{\text{tc}} = D_{\text{KL}}(q(z_c, z_s, z_a) \| q(z_c)q(z_s)q(z_a))$$

**HSIC正则化**（$\lambda_{\text{hsic}} = 0.02$）：
$$\mathcal{L}_{\text{hsic}} = \sum_{f \neq g \in \{c,s,a\}} \text{HSIC}(z_f, z_g)$$
$$\text{HSIC}(z_f, z_g) = \frac{1}{(m-1)^2} \text{tr}(K_{z_f} H K_{z_g} H)$$

**对比预测编码损失**（$\lambda_{\text{contrastive}} = 0.20$）：
$$\mathcal{L}_{\text{contrastive}} = -\log \frac{\exp(\text{sim}(z_c^{(i)}, z_c^{(i+)})/\tau)}{\sum_{j=1}^B \exp(\text{sim}(z_c^{(i)}, z_c^{(j)})/\tau)}$$

**正则化预热**：所有正则化权重从0线性增加到目标值，预热周期为3个epoch。

#### 3.3.2 阶段二：成对与排序约束（Epoch 2-10）

从第2个epoch开始，引入成对约束，预热2个epoch：

$$\mathcal{L}_{\text{pair}} = \frac{1}{|\mathcal{P}|} \sum_{(i,j,k) \in \mathcal{P}} \max\left(0, d(z_c^{(i)}, z_c^{(j)}) - d(z_c^{(i)}, z_c^{(k)}) + 0.5\right)$$

从第4个epoch开始，引入排序约束，预热2个epoch：

$$\mathcal{L}_{\text{rank}} = \frac{1}{|\mathcal{R}|} \sum_{(i,j) \in \mathcal{R}} \log\left(1 + \exp\left(-(s(i) - s(j))\right)\right)$$

#### 3.3.3 阶段三：最优传输跨文化对齐

OT损失使用Sinkhorn算法 \cite{cuturi2013sinkhorn}，$\epsilon = 0.1$，200次迭代。给定两个文化的潜分布样本，定义代价矩阵 $C_{ij} = \|z^{(c_1)}_i - z^{(c_2)}_j\|_2^2$，求解：

$$\text{OT}_{\epsilon}(p, q) = \min_{T \in \mathcal{U}(p,q)} \langle T, C \rangle - \epsilon H(T)$$

Sinkhorn迭代：
$$a^{(t+1)} = p \oslash (K b^{(t)}), \quad b^{(t+1)} = q \oslash (K^T a^{(t+1)}), \quad K = \exp(-C/\epsilon)$$

#### 3.3.4 总损失与权重配置

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{recon}} + \beta_{\text{KL}} \mathcal{L}_{\text{KL}} + \lambda_{\text{pair}} \mathcal{L}_{\text{pair}} + \lambda_{\text{rank}} \mathcal{L}_{\text{rank}} + \lambda_{\text{domain}} \mathcal{L}_{\text{domain}} + \lambda_{\text{contrastive}} \mathcal{L}_{\text{contrastive}} + \lambda_{\text{cov}} \mathcal{L}_{\text{cov}} + \lambda_{\text{tc}} \mathcal{L}_{\text{tc}} + \lambda_{\text{hsic}} \mathcal{L}_{\text{hsic}} + \lambda_{\text{source}} \mathcal{L}_{\text{source}} + \lambda_{\text{OT}} \mathcal{L}_{\text{OT}}$$

权重配置：$\lambda_{\text{pair}}=0.15$, $\lambda_{\text{rank}}=0.12$, $\lambda_{\text{domain}}=0.50$, $\lambda_{\text{contrastive}}=0.20$, $\lambda_{\text{cov}}=0.05$, $\lambda_{\text{tc}}=0.05$, $\lambda_{\text{hsic}}=0.02$, $\lambda_{\text{source}}=0.10$, $\beta_{\text{KL}}=1.0$。

优化器：AdamW，$lr=0.002$，$\text{batch\_size}=128$，10 epochs，$\text{seed}=42$。

### 3.4 OT检索与校准感知重排序

#### 3.4.1 OT检索

用户偏好表示：
$$\bar{z}_c^{(u)} = \frac{1}{|\mathcal{I}_u|} \sum_{i \in \mathcal{I}_u} \mu_c^{(i)}$$

基础推荐得分：
$$s_{\text{content}}(u, j) = \text{cosine}(\bar{z}_c^{(u)}, \mu_c^{(j)})$$

OT重加权：
$$s_{\text{OT}}(u, j) = s_{\text{content}}(u, j) + \alpha \cdot \sum_{i \in \mathcal{I}_u} T^*_{ij}$$

#### 3.4.2 校准感知重排序

$$s_{\text{final}}(u, j) = w_{\text{rel}} s_{\text{rel}} + w_{\text{nov}} s_{\text{nov}} + w_{\text{aff}} s_{\text{aff}} + w_{\text{min}} s_{\text{min}} + w_{\text{src}} s_{\text{src}} + w_{\text{div}} s_{\text{div}}$$

目标校准操作点权重：

| 维度 | 相关性 | 新颖性 | 目标亲和力 | 少数群体 | 来源均衡 | 多样性 |
|---|---|---|---|---|---|---|
| 权重 | 0.48 | 0.10 | 0.22 | 0.14 | 0.06 | 0.03 |

### 3.5 参与式主动学习（PAL）流程

**流程**：
1. **不确定性采样**：选择判别头预测熵最高的样本
2. **主动标注包**：打包 $k$ 首曲目及其元数据
3. **专家标注**：文化归属、情感标签、风格描述、相对相似度
4. **模型更新**：微调与标注文化相关的参数
5. **迭代循环**：直到收敛或预算耗尽
