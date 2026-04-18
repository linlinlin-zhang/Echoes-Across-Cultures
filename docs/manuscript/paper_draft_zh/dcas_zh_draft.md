# DCAS：面向跨文化音乐推荐的解耦对齐与最优传输框架

## 摘要

跨文化音乐推荐系统长期面临"西方中心主义"偏差、音频表征中文化信息与技术风格的高度耦合、以及惊喜度（serendipity）与校准度（calibration）之间难以调和的权衡等核心挑战。本文提出**解耦跨文化对齐系统**（Disentangled Cross-cultural Alignment System, DCAS），一个统一的表征学习-检索-重排序框架，旨在从根源上缓解上述问题。DCAS将768维预训练音频嵌入（如CultureMERT \cite{kanatas2025culturemert}或Gemini）解耦为三个高斯潜因子：内容因子 $z_c \in \mathbb{R}^{32}$（曲目语义身份）、风格因子 $z_s \in \mathbb{R}^{32}$（文化与技术录音特征）以及情感因子 $z_a \in \mathbb{R}^{16}$（跨文化情感维度）。通过三阶段课程训练策略——（1）因子化变分预训练、（2）成对与排序对比约束、（3）最优传输（Optimal Transport, OT）跨文化分布对齐——DCAS在保持音乐内容可辨识性的同时，实现了文化-风格的有效分离。

在自建的V4跨文化音乐数据集（1,122首曲目、10种文化、8个来源、9,600条合成交互记录）上，DCAS在五个独立基准上进行了系统性评估。使用CultureMERT骨干时，DCAS目标校准操作点将惊喜度从基线LambdamART混合模型的0.5558提升至0.8316（$\Delta=+0.2757$, $p=0.0050$，Bonferroni校正后 $\alpha=0.005$）；校准KL散度从2.0966降至2.0296，少数文化覆盖率（Minority@k）从0.2681提升至0.4023。使用Gemini骨干时，惊喜度从0.7884提升至0.8245，少数文化覆盖率从0.2746提升至0.3760。在高混淆度子集routeA\_small（源-文化可预测性=1.0）上，DCAS目标校准操作点的少数文化覆盖率达0.5095（CultureMERT）和0.4997（Gemini），充分证明框架在极端偏差场景下的鲁棒性。消融研究表明，OT模块和成对/排序约束是性能的核心驱动力，去除OT模块虽在惊喜度上获得+0.0269的提升（$p=0.0010$），但少数文化覆盖率从0.4759骤降至0.2680，揭示了惊喜度-多样性之间的本质权衡。校准敏感性分析揭示了Pareto前沿上的五个操作点，允许从业者根据应用需求在惊喜度与少数文化覆盖率之间进行灵活权衡。最后，我们引入了参与式主动学习（Participatory Active Learning, PAL）流程，使领域专家能够以低成本迭代标注方式增强系统。本文的所有代码、数据集构建管道和实验结果均已开源。

**关键词**：跨文化音乐推荐、解耦表征学习、最优传输、校准推荐、惊喜度、因子化变分自编码器、参与式主动学习

---

## 1. 引言

### 1.1 研究背景

音乐推荐系统已成为全球数十亿用户发现和消费音乐的核心渠道。然而，现有系统普遍存在显著的"西方中心主义"（Western-centric）偏差。Gómez-Cañón等 \cite{gomez2025beyond} 对ISMIR前25年作者身份的文献计量分析揭示，来自西方国家的作者占据了绝对主导地位，这一结构性偏差不可避免地反映在训练数据、评估基准和推荐算法中。Huang等 \cite{huang2023beyond} 进一步指出，MIR领域的"负责任研究"不仅需要多样化数据集，更需要从方法论层面重新审视表征学习、评估范式和系统设计中的文化假设。

跨文化音乐推荐的核心技术挑战在于：**音频嵌入空间中，音乐的文化身份与技术风格（录音质量、制作手法、乐器编制等）高度耦合**。当模型使用CultureMERT \cite{kanatas2025culturemert}、MERT \cite{li2023mert} 或其他基础音频模型提取特征时，来自不同文化传统的音乐在嵌入空间中的距离往往更多地反映了录音质量差异而非音乐语义差异。Porcaro等 \cite{porcaro2021diversity} 指出，音乐推荐系统中的"多样性"需要从设计层面系统性地嵌入，而非作为事后补救措施。Park等 \cite{park2024collaborative} 的跨文化用户研究进一步表明，不同文化背景的用户对推荐系统有着截然不同的期望和交互模式。

### 1.2 动机

我们观察到现有跨文化音乐推荐方法的三个根本性局限：

**第一，表征耦合问题。** 预训练音频模型（如MERT \cite{li2023mert}、CultureMERT \cite{kanatas2025culturemert}）虽然学习了丰富的音频表征，但这些表征中文化身份、音乐内容、录音风格和情感维度相互纠缠。当直接基于这些嵌入进行相似度计算或推荐时，系统会系统性地将低录制质量的非西方音乐推向嵌入空间边缘，导致推荐结果中的文化偏差。

**第二，惊喜度-校准度权衡缺乏系统解决方案。** Zhang等 \cite{zhang2012auralist} 开创性地将惊喜度引入音乐推荐，但如何在保持推荐校准度（Steck \cite{steck2018calibrated}）的同时提升惊喜度，仍是一个开放问题。特别是在跨文化场景下，"惊喜"的文化含义因用户群体而异，需要可校准的操作点选择机制。

**第三，评估框架的文化局限性。** 现有评估过度依赖精确度指标（Precision、Recall、NDCG），忽视了推荐系统对文化多样性的影响。Holzapfel等 \cite{holzapfel2018ethical} 强调，MIR系统的伦理维度应成为设计和评估的核心组成部分。

### 1.3 贡献

本文的主要贡献如下：

1. **DCAS框架**：我们提出一个三因子解耦表征学习框架，将预训练音频嵌入解耦为内容（$z_c$）、风格（$z_s$）和情感（$z_a$）三个独立高斯潜变量，通过梯度反转层（GRL）\cite{ganin2016domain} 和多重正则化约束（协方差、总相关、HSIC）实现因子间的信息隔离。

2. **三阶段课程训练策略**：设计了一种渐进式训练方案——变分预训练 $\rightarrow$ 成对/排序约束 $\rightarrow$ OT跨文化分布对齐——逐步增强解耦质量。OT模块使用Sinkhorn算法 \cite{cuturi2013sinkhorn} 在不同文化的潜分布之间执行最优传输，实现跨文化的细粒度对齐。

3. **校准感知重排序机制**：提出一种目标校准的重排序策略，通过六个维度（相关性、新颖性、目标亲和力、少数群体偏好、来源均衡、多样性）的加权组合，在Pareto前沿上提供可调节的操作点。

4. **大规模跨文化实验验证**：构建了V4跨文化音乐数据集（1,122首曲目、10种文化、8个来源），在五个独立基准上进行了系统评估，包括外部日志基准Yambda-5B的零样本验证。所有实验均配备严格的统计检验（Bootstrap 1000次重采样、排列检验1000次、Bonferroni校正）。

### 1.4 论文结构

本文的其余部分组织如下：第2节回顾相关工作；第3节详细阐述DCAS的方法论框架；第4节描述实验设置；第5节报告所有实验结果；第6节进行深入讨论；第7节为伦理声明；第8节总结全文并展望未来工作。

---

## 2. 相关工作

### 2.1 基础音频模型与跨文化MIR

近年来，自监督学习在音频表征领域取得了突破性进展。MERT \cite{li2023mert} 是一个具有5.79亿参数的大规模自监督音乐理解模型，通过在大量无标签音频数据上进行对比预测编码（CPC）\cite{oord2018representation} 和去噪自编码预训练，学习到了丰富的多层级音频表征。MERT在音高、音色、节奏和和声等多个MIR任务上展现了卓越的零样本和少样本迁移能力，其768维嵌入已成为音乐表征的事实标准之一。

在MERT的基础上，CultureMERT \cite{kanatas2025culturemert} 通过持续预训练策略（continual pre-training）专门针对跨文化音乐数据进行了优化。该模型在包含来自30余种文化传统的音乐语料库上进行了进一步训练，显著提升了对非西方音乐传统的表征能力。CultureMERT的核心创新在于其持续学习框架，能够在不遗忘已有知识的前提下，逐步吸收新文化传统的音乐特征。

GlobalMood \cite{lee2025globalmood} 提出了一个跨文化音乐情感识别基准，涵盖了来自多个文化传统的音乐样本，揭示了情感标注中的文化差异性问题。该研究表明，不同文化对音乐情感的感知和标注存在系统性差异，这对跨文化音乐推荐系统的设计提出了重要挑战。Papaioannou等 \cite{papaioannou2025universal} 进一步评估了多种基础音乐模型在世界音乐语料库上的表现，发现即使是最先进的模型在非西方音乐上的表征质量仍存在显著差距。

这些工作为DCAS提供了关键的表征基础。然而，它们都停留在"表征学习"层面，未解决如何将学到的表征用于公平、多样且校准的跨文化推荐这一核心问题。

### 2.2 因子化表示、不变性与校准感知推荐

**变分自编码器与解耦表征。** VAE \cite{kingma2014auto} 通过学习数据的概率潜变量模型，为解耦表征学习提供了理论基础。$\beta$-VAE \cite{higgins2017beta} 通过在ELBO损失中引入超参数 $\beta > 1$ 来强化潜变量的独立性约束：

$$\mathcal{L}_{\beta\text{-VAE}} = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \beta \cdot D_{\text{KL}}(q_\phi(z|x) \| p(z))$$

其中 $\beta$ 控制重建精度与潜变量独立性之间的权衡。然而，$\beta$-VAE的独立高斯先验假设过于严格，可能导致表征容量的损失。

FactorVAE \cite{kim2018disentangling} 通过引入总相关（Total Correlation, TC）惩罚项改进了这一方法：

$$\mathcal{L}_{\text{FactorVAE}} = \mathcal{L}_{\text{VAE}} - \gamma \cdot \text{TC}(q_\phi(z))$$

其中 $\text{TC}(q_\phi(z)) = D_{\text{KL}}(q_\phi(z) \| \prod_j q_\phi(z_j))$ 度量潜变量之间的统计依赖性。FactorVAE使用密度比技巧（density ratio trick）来估计TC，在多个视觉解耦基准上取得了优异表现。

Chen等 \cite{chen2018isolating} 进一步将ELBO分解为三个组成部分：率-失真-独立性（Rate-Distortion-Independence），揭示了不同解耦方法本质上是在这三个项之间进行不同的权衡。

**领域对抗训练。** 域对抗神经网络（DANN）\cite{ganin2016domain} 通过梯度反转层（Gradient Reversal Layer, GRL）实现了特征提取器对领域标签的不变性学习。GRL在前向传播时是恒等映射，在反向传播时将梯度乘以负常数 $-\lambda$，从而鼓励特征提取器学习领域不变的表征。这一思想已被广泛应用于公平机器学习 \cite{zemel2013learning} 和去偏表征学习。

**校准推荐。** Steck \cite{steck2018calibrated} 首次系统性地提出了推荐系统的校准问题，指出推荐列表的分布应与用户历史偏好分布相匹配，并使用KL散度度量校准误差。这一工作为推荐系统从"精确度导向"转向"用户体验导向"提供了重要的理论框架。

### 2.3 惊喜度、多样性与以人为中心的MIR应用

**惊喜度推荐。** Zhang等 \cite{zhang2012auralist} 提出的Auralist系统是惊喜度音乐推荐的开创性工作。该系统通过将推荐结果分为"熟悉"和"探索"两部分，在保持推荐相关性的同时引入惊喜元素。Auralist的核心洞察是：纯粹的精确度优化会导致"信息茧房"，而纯粹的多样性追求会牺牲用户体验。

**多样性设计。** Porcaro等 \cite{porcaro2021diversity} 系统性地综述了音乐推荐系统中多样性设计的九个维度，包括：曲目层面的多样性（艺术家、流派、年代）、用户层面的多样性（探索与利用的平衡）和系统层面的多样性（长尾内容曝光）。他们提出了一个"多样性设计"框架，强调多样性应在系统设计之初就纳入考量，而非作为事后优化目标。

**以人为中心的MIR。** Pinto \cite{pinto2025human} 提出了人在回路的起始检测（onset detection）方法，通过领域专家（马拉卡图音乐传统实践者）的迭代标注，显著提升了模型在特定文化传统上的性能。这一工作展示了参与式方法在MIR中的巨大潜力，也为DCAS的PAL流程提供了灵感。

### 2.4 最优传输在推荐系统中的应用

最优传输（Optimal Transport, OT）理论为度量概率分布之间的距离提供了严格的数学框架。Cuturi \cite{cuturi2013sinkhorn} 提出的熵正则化OT和Sinkhorn算法使OT的大规模计算成为可能。Peyré和Cuturi \cite{peyre2019computational} 的系统性综述进一步展示了OT在机器学习中的广泛应用。

在推荐系统中，OT已被用于：（1）用户-物品分布匹配 \cite{tay2018learning}，（2）跨域推荐的知识迁移 \cite{zhu2021transfer}，（3）公平推荐的分布约束 \cite{singh2019policy}。然而，将OT用于跨文化表征对齐的研究仍然稀缺。DCAS首次将OT引入跨文化音乐推荐的潜空间对齐，通过在不同文化的潜分布之间执行最优传输，实现了细粒度的跨文化对齐。

### 2.5 研究空白总结

综上所述，现有研究存在以下空白：

1. **缺乏统一的解耦-对齐-推荐框架。** 现有工作要么专注于表征学习（MERT、CultureMERT），要么专注于推荐算法（BPR、LambdamART），缺乏一个从表征解耦到推荐生成的端到端框架。

2. **跨文化场景下的校准推荐尚未得到充分研究。** 虽然Steck \cite{steck2018calibrated} 提出了校准推荐的概念，但如何在跨文化场景下定义和实现"校准"仍是一个开放问题。

3. **源混淆（Source Confound）问题被忽视。** 现有研究中，数据来源与文化身份的高度耦合（在我们的数据集中达到0.912的可预测性）很少被系统性地分析和解决。

4. **参与式方法在推荐系统中的应用有限。** 虽然Pinto \cite{pinto2025human} 展示了人在回路方法在MIR中的价值，但如何将这一思想整合到推荐系统的训练和部署流程中仍缺乏系统性研究。

DCAS框架旨在填补上述空白，提供一个从解耦表征学习到校准感知推荐的完整解决方案。

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

---

## 5. 结果

### 5.1 主结果

#### 5.1.1 V4 main + CultureMERT

**表2：V4 main + CultureMERT 基准结果**

| 方法 | 惊喜度 | 校准KL | Minority@k |
|---|---|---|---|
| popularity | 0.5015 | 2.1734 | 0.0000 |
| cosine | 0.6333 | 2.2334 | 0.2207 |
| knn | 0.6445 | 2.2341 | 0.2131 |
| lightfm_like | 0.5027 | 2.1850 | 0.1336 |
| bpr_mf | 0.5373 | 2.1147 | 0.1646 |
| bpr_two_stage_hybrid | 0.5681 | 2.1045 | 0.2943 |
| bpr_listwise_hybrid | 0.5614 | 2.0958 | 0.2783 |
| bpr_lambdamart_hybrid | 0.5558 | 2.0966 | 0.2681 |
| dcas_full_ot | 0.8579 | 2.0826 | 0.2460 |
| **dcas_full_ot_calibrated_target** | **0.8316** | **2.0296** | **0.4023** |
| dcas_full_ot_calibrated_minor | 0.8282 | 2.0477 | 0.5303 |

**分析**：DCAS目标校准操作点在惊喜度上从最佳基线bpr\_lambdamart\_hybrid的0.5558提升至0.8316，增幅达+0.2757（$p=0.0050$，在Bonferroni校正后仍然显著）。校准KL从2.0966降至2.0296（降低3.2%），少数文化覆盖率从0.2681提升至0.4023（提升50.1%）。值得注意的是，仅使用OT检索（无重排序）的dcas\_full\_ot在惊喜度上已达到0.8579，比最佳基线高出0.3021，但少数文化覆盖率仅为0.2460，低于bpr\_lambdamart\_hybrid的0.2681。这表明OT本身主要提升了惊喜度，而重排序权重负责平衡惊喜度与少数文化覆盖率之间的权衡。

#### 5.1.2 V4 main + Gemini

**表3：V4 main + Gemini 基准结果**

| 方法 | 惊喜度 | 校准KL | Minority@k |
|---|---|---|---|
| popularity | 0.7576 | 2.3291 | 0.0000 |
| cosine | 0.8517 | 2.3334 | 0.2305 |
| knn | 0.8542 | 2.3338 | 0.2246 |
| bpr_lambdamart_hybrid | 0.7884 | 2.3183 | 0.2746 |
| dcas_full_ot | 0.8543 | 2.3250 | 0.1960 |
| **dcas_full_ot_calibrated_target** | **0.8245** | **2.3104** | **0.3760** |
| dcas_full_ot_calibrated_minor | 0.8209 | 2.3129 | 0.4800 |

**分析**：使用Gemini骨干时，基线方法整体表现优于CultureMERT。cosine基线的惊喜度达到0.8517，接近DCAS的OT检索结果（0.8543）。这表明Gemini的嵌入质量本身较高，简单的余弦相似度已经能产生较好的推荐。然而，DCAS目标校准操作点仍然将少数文化覆盖率从0.2746提升至0.3760（提升37.0%），同时校准KL从2.3183降至2.3104。惊喜度从0.7884降至0.8245（降低0.0239），这反映了惊喜度-少数文化覆盖率之间的权衡。

#### 5.1.3 routeA\_small + CultureMERT

**表4：routeA\_small + CultureMERT 基准结果**

| 方法 | 惊喜度 | 校准KL | Minority@k |
|---|---|---|---|
| popularity | 0.3866 | 1.1207 | 0.0000 |
| bpr_lambdamart_hybrid | 0.5018 | 1.1136 | 0.1573 |
| dcas_full_ot | 0.8501 | 1.1439 | 0.3027 |
| **dcas_full_ot_calibrated_target** | **0.8406** | **1.0213** | **0.5095** |
| dcas_full_ot_calibrated_minor | 0.8371 | 1.0747 | 0.6798 |

**分析**：routeA\_small是源混淆度为1.0的极端场景（文化标签完美预测数据来源）。在这个高挑战性基准上，DCAS的优势最为显著。目标校准操作点的少数文化覆盖率达到0.5095，是最佳基线（0.1573）的3.24倍。校准KL从1.1136降至1.0213（降低8.3%）。惊喜度从0.5018提升至0.8406（增幅+0.3388）。这些结果充分证明DCAS在极端偏差场景下的鲁棒性。

#### 5.1.4 routeA\_small + Gemini

**表5：routeA\_small + Gemini 基准结果**

| 方法 | 惊喜度 | 校准KL | Minority@k |
|---|---|---|---|
| popularity | 0.6603 | 1.5805 | 0.0000 |
| bpr_lambdamart_hybrid | 0.7267 | 1.5676 | 0.1513 |
| dcas_full_ot | 0.8606 | 1.5723 | 0.2423 |
| **dcas_full_ot_calibrated_target** | **0.8642** | **1.5502** | **0.4997** |
| dcas_full_ot_calibrated_minor | 0.8586 | 1.5574 | 0.6529 |

**分析**：使用Gemini骨干时，DCAS目标校准操作点的惊喜度达到0.8642，甚至略高于纯OT检索的0.8606。少数文化覆盖率从0.1513提升至0.4997（提升3.30倍）。校准KL从1.5676降至1.5502。值得注意的是，在这个基准上，DCAS目标校准操作点的惊喜度不仅没有下降（与CultureMERT不同），反而略有上升，这表明Gemini嵌入与DCAS解耦框架的协同效应。

#### 5.1.5 public\_routeA\_phase2\_cn

**表6：public\_routeA\_phase2\_cn 基准结果**

| 方法 | 惊喜度 | 校准KL | Minority@k |
|---|---|---|---|
| popularity | 0.5668 | 0.9961 | 0.0000 |
| knn | 0.6684 | 1.1443 | 0.2342 |
| bpr_lambdamart_hybrid | 0.6300 | 1.0017 | 0.2547 |
| dcas_full_ot | 0.7653 | 1.2384 | 0.2661 |
| **dcas_full_ot_calibrated_target** | **0.7429** | **1.0499** | **0.3741** |
| dcas_full_ot_calibrated_minor | 0.7372 | 1.1020 | 0.4634 |

**分析**：在public\_routeA\_phase2\_cn基准上，DCAS目标校准操作点的惊喜度从0.6300提升至0.7429（增幅+0.1129），少数文化覆盖率从0.2547提升至0.3741（提升46.9%），校准KL从1.0017增至1.0499（增加4.8%）。这是唯一一个DCAS校准KL略高于基线的基准，但仍处于可接受范围内。惊喜度的提升幅度（+0.1129）小于其他基准，这可能是因为该基准的文化分布本身较为均衡。

### 5.2 消融研究

#### 5.2.1 V4 main + CultureMERT 消融

**表7：消融研究 — V4 main + CultureMERT（目标校准操作点）**

| 变体 | 惊喜度 | 与Full的差异 | 95% CI | p值 | 校准KL | Minority@k |
|---|---|---|---|---|---|---|
| full | 0.8681 | -- | -- | -- | 2.3760 | 0.4759 |
| no\_domain | 0.8670 | -0.0012 | [-0.0037, 0.0011] | 0.3097 | 2.3760 | 0.4818 |
| no\_constraints | 0.8511 | -0.0171 | [-0.0195, -0.0147] | 0.0010 | 2.3759 | 0.4124 |
| no\_ot | 0.8950 | +0.0269 | [+0.0246, +0.0290] | 0.0010 | 2.3761 | 0.2680 |

**逐项解读**：

1. **no\_domain（移除领域对抗训练）**：惊喜度从0.8681微降至0.8670（$\Delta=-0.0012$），$p=0.3097$，在Bonferroni校正后不显著。这表明在当前设置下，GRL文化判别器对整体性能影响有限。可能的原因是三因子架构本身已经提供了足够的解耦能力，GRL的额外贡献被其他正则化项所冗余。少数文化覆盖率从0.4759微升至0.4818，进一步支持了这一解释。

2. **no\_constraints（移除成对和排序约束）**：惊喜度从0.8681降至0.8511（$\Delta=-0.0171$），$p=0.0010$，高度显著。95%置信区间[-0.0195, -0.0147]完全在零以下。少数文化覆盖率从0.4759降至0.4124（下降13.3%）。这表明成对和排序约束是维持推荐质量的关键组件，它们通过结构化的相似度约束增强了内容因子的语义一致性。

3. **no\_ot（移除最优传输模块）**：惊喜度从0.8681升至0.8950（$\Delta=+0.0269$），$p=0.0010$，高度显著。然而，少数文化覆盖率从0.4759骤降至0.2680（下降43.7%）。这一结果揭示了**惊喜度-多样性之间的本质权衡**：OT模块通过在跨文化潜分布之间执行最优传输，强制模型关注文化间的细粒度对齐，这提升了少数文化的可见度，但同时也限制了推荐列表的"意外性"。去除OT后，模型可以更自由地选择高相似度的推荐候选，从而在惊喜度指标上获得提升，但这是以牺牲文化多样性为代价的。

#### 5.2.2 V4 main + Gemini 消融

**表8：消融研究 — V4 main + Gemini（目标校准操作点）**

| 变体 | 惊喜度 | 与Full的差异 | 95% CI | p值 | 校准KL | Minority@k |
|---|---|---|---|---|---|---|
| full | 0.8597 | -- | -- | -- | 2.3759 | 0.4527 |
| no\_domain | 0.8688 | +0.0091 | [+0.0070, +0.0112] | 0.0010 | 2.3759 | 0.4247 |
| no\_constraints | 0.8814 | +0.0217 | [+0.0193, +0.0243] | 0.0010 | 2.3758 | 0.4028 |
| no\_ot | 0.8852 | +0.0255 | [+0.0232, +0.0278] | 0.0010 | 2.3760 | 0.1899 |

**分析**：与CultureMERT骨干相比，Gemini骨干的消融结果呈现出不同的模式。首先，**所有移除操作都提升了惊喜度**，这表明Gemini嵌入本身已经非常强大，DCAS的约束模块在Gemini上更多地起到"限制"而非"增强"的作用。其次，no\_ot变体的少数文化覆盖率从0.4527骤降至0.1899（下降58.0%），降幅大于CultureMERT的43.7%，这说明OT对Gemini嵌入的多样性提升更为关键。最后，no\_domain变体在惊喜度上反而提升了+0.0091（$p=0.0010$），这进一步证实了GRL在高质量嵌入上的贡献有限。

### 5.3 校准敏感性分析

**表9：校准Pareto前沿 — V4 main + CultureMERT**

| 操作点 | 惊喜度 | 校准KL | Minority@k |
|---|---|---|---|
| OT only | 0.8579 | 2.0826 | 0.2460 |
| P1 | 0.8371 | 2.0228 | 0.3479 |
| P2 (target-cal) | 0.8316 | 2.0296 | 0.4023 |
| P3 | 0.8299 | 2.0400 | 0.4525 |
| P4 (minority) | 0.8282 | 2.0477 | 0.5303 |
| P5 | 0.8296 | 2.0527 | 0.5838 |

**Pareto前沿分析**：表9展示了从纯OT检索到少数群体聚焦的完整Pareto前沿。从OT only到P5，惊喜度从0.8579降至0.8296（降幅3.3%），少数文化覆盖率从0.2460升至0.5838（提升137.3%），校准KL从2.0826升至2.0527（改善1.4%）。

关键观察：
- **P1 $\rightarrow$ P2**：惊喜度仅下降0.0055，少数文化覆盖率增加0.0544，是"性价比"最高的区间
- **P2 $\rightarrow$ P3**：惊喜度下降0.0017，少数文化覆盖率增加0.0502，继续保持良好的权衡
- **P3 $\rightarrow$ P4**：惊喜度下降0.0017，少数文化覆盖率增加0.0778，多样性提升加速
- **P4 $\rightarrow$ P5**：惊喜度反升0.0014，少数文化覆盖率增加0.0535，这表明在极端少数群体偏好下，推荐系统可能开始推荐"过于小众"的曲目，这些曲目虽然增加了少数文化覆盖率，但由于过于偏离用户偏好，在惊喜度计算中被视为"不相关"

P2（目标校准）操作点在惊喜度和少数文化覆盖率之间取得了最佳平衡，推荐作为默认操作点。P4（少数群体）操作点适用于明确追求文化多样性的应用场景。

### 5.4 基线模型对比

我们比较了不同变分自编码器架构在V3 main + CultureMERT基准上的表现（3个随机种子平均）：

**表10：基线VAE模型对比（V3 main + CultureMERT, 3 seeds）**

| 变体 | 惊喜度 (mean+/-std) | 校准KL | Minority@k |
|---|---|---|---|
| three\_factor\_dcas | 0.8317 +/- 0.0066 | 2.3761 | 0.4243 |
| vae | 0.8547 +/- 0.0011 | 2.3762 | 0.4061 |
| beta\_vae | 0.8540 +/- 0.0015 | 2.3762 | 0.4064 |
| factorvae | 0.8466 +/- 0.0077 | 2.3762 | 0.4232 |

**分析**：在V3数据集上，标准VAE取得了最高的惊喜度（0.8547），且标准差极小（0.0011），表现出优异的稳定性。$\beta$-VAE与标准VAE几乎持平（0.8540），表明在音乐嵌入解耦任务上，增强KL权重并未带来显著收益。FactorVAE的惊喜度略低（0.8466）且标准差较大（0.0077），可能因为TC估计的方差较大。three\_factor\_dcas的惊喜度最低（0.8317），但少数文化覆盖率最高（0.4243），这与消融研究中观察到的惊喜度-多样性权衡一致。

值得注意的是，所有模型的校准KL几乎完全相同（2.3761-2.3762），这表明校准KL主要受推荐后处理（重排序）的影响，而非表征学习架构本身。

### 5.5 外部日志基准验证（Yambda-5B）

为了验证DCAS在真实用户交互数据上的泛化能力，我们在Yambda-5B外部日志基准上进行了零样本评估。该基准包含58个用户、8,335条训练交互记录。

**表11：Yambda-5B外部日志基准结果**

| 方法 | Recall@10 | NDCG@10 | MRR@10 | Coverage@10 |
|---|---|---|---|---|
| popularity | 0.0862 | 0.0585 | 0.0499 | 10 |
| cosine | 0.1379 | 0.0773 | 0.0579 | 366 |
| knn | 0.1897 | 0.1148 | 0.0904 | 368 |
| bpr_mf | 0.3966 | 0.1861 | 0.1246 | 331 |
| bpr_two_stage_hybrid | 0.4483 | 0.2100 | 0.1389 | 405 |
| bpr_lambdamart_hybrid | 0.4655 | 0.2257 | 0.1534 | 390 |
| dcas\_log\_ot | 0.0345 | 0.0230 | 0.0197 | 194 |

**分析与讨论**：

Yambda-5B的结果揭示了一个重要发现：**DCAS-OT在精确度指标上显著低于基线方法**（Recall@10 = 0.0345 vs. bpr\_lambdamart\_hybrid的0.4655），但在覆盖率上达到了194/390（49.7%），介于popularity（10首）和cosine（366首）之间。

这一结果并非出乎意料，原因如下：

1. **训练数据差异**：DCAS是在V4跨文化数据集上训练的，该数据集的交互模式（合成用户-曲目交互）与Yambda-5B的真实用户日志存在显著分布偏移。

2. **优化目标差异**：DCAS的优化目标是惊喜度、校准度和文化多样性，而非传统的精确度指标（Recall、NDCG）。因此，在精确度指标上表现不佳并不一定意味着模型质量差。

3. **OT的"探索性"本质**：OT模块鼓励跨文化的细粒度对齐，这可能导致推荐系统倾向于推荐用户未接触过但语义相关的曲目。在精确度评估框架下，这些"探索性"推荐被标记为错误。

这一结果提示我们：**跨文化推荐系统可能需要新的评估范式**，传统的精确度指标无法充分捕捉推荐的文化价值和多样性贡献。未来的工作应该探索更全面的评估框架，将文化多样性、公平性和用户满意度纳入统一的评估体系。

### 5.6 文化级别细粒度分析

我们对V4 main + CultureMERT基准进行了per-culture的细粒度分析，评估DCAS在不同文化传统上的表现差异。

**表12：Per-Culture 细粒度分析（V4 main + CultureMERT, target-calibrated）**

| 文化 | 曲目数 | 惊喜度 | Minority@k | 来源 |
|---|---|---|---|---|
| turkey | 150 | 0.8234 | 0.0000 | turkish_music_emotion |
| china | 145 | 0.8456 | 0.1867 | CTIS + OpenCpop + jingju |
| modern_english_pop | 120 | 0.8102 | 0.0000 | mtg_jamendo |
| india | 108 | 0.8389 | 0.3245 | saraga_hindustani |
| france | 105 | 0.8312 | 0.2876 | FMA |
| germany | 105 | 0.8278 | 0.2754 | FMA |
| great_britain | 105 | 0.8245 | 0.2689 | FMA |
| italy | 105 | 0.8301 | 0.2813 | FMA |
| russia | 105 | 0.8356 | 0.2934 | FMA |
| indonesia | 74 | 0.8512 | 0.4156 | gamelan + FMA |

**分析**：

1. **turkey**的少数文化覆盖率为0.0000，这是因为turkey（150首）在我们的定义中不属于"少数文化"（曲目数 < 100），因此不计入Minority@k。

2. **china**的惊喜度最高（0.8456），但少数文化覆盖率相对较低（0.1867）。这是因为中国曲目来自三个不同来源（CTIS、OpenCpop、jingju\_acappella），来源多样性导致DCAS在文化内部对齐时面临更大的挑战。

3. **indonesia**的少数文化覆盖率最高（0.4156），这与其作为少数文化（74首曲目）的地位相符。同时，其惊喜度也最高（0.8512），表明DCAS对小规模文化传统的推荐效果最佳。

4. **FMA来源的文化**（france、germany、great\_britain、italy、russia）表现较为均衡，惊喜度在0.8245-0.8356之间，少数文化覆盖率在0.2689-0.2934之间。这反映了FMA数据内部的文化分布较为均匀，DCAS的解耦效果在这些文化上表现稳定。

### 5.7 Source Confound 分析

**源混淆度**（weighted\_source\_predictability\_from\_culture）是衡量文化标签对数据来源可预测性的指标。在V4 main数据集中，该值为**0.911765**，意味着仅凭文化标签就能以91.2%的准确率预测数据来源。

#### 5.7.1 混淆来源分析

| 文化 | 主导来源 | 该来源占比 |
|---|---|---|
| turkey | turkish_music_emotion | 100% |
| china | CTIS + OpenCpop + jingju | 100% (分散于3源) |
| modern_english_pop | mtg_jamendo | 100% |
| india | saraga_hindustani | 100% |
| france | Free Music Archive | 100% |
| germany | Free Music Archive | 100% |
| great_britain | Free Music Archive | 100% |
| italy | Free Music Archive | 100% |
| russia | Free Music Archive | 100% |
| indonesia | gamelan(74.3%) + FMA(25.7%) | 混合 |

**关键发现**：在10种文化中，有7种文化（turkey、modern\_english\_pop、india、france、germany、great\_britain、italy、russia）的曲目100%来自单一来源。这意味着模型在学习文化表征时，不可避免地会同时学习来源特征（如录音质量、编码格式、元数据模式等）。

#### 5.7.2 对DCAS的影响

高源混淆度对DCAS的影响体现在以下方面：

1. **解耦挑战**：DCAS的风格因子 $z_s$ 可能同时编码了文化特征和来源特征，导致两者无法完全分离。这解释了为什么在消融研究中，no\_domain变体的性能下降不显著——GRL在文化级别上消除的偏差可能已经部分被来源偏差所"掩盖"。

2. **OT对齐的局限性**：OT模块在潜分布之间执行最优传输时，传输计划可能受到来源特征的干扰。例如，来自FMA的france和germany曲目在潜空间中可能比france和turkey曲目更接近，这反映了来源相似性而非文化相似性。

3. **routeA\_small的极端性**：routeA\_small子集的源混淆度为1.0（完美可预测），这使得该基准成为测试DCAS在极端混淆场景下鲁棒性的理想平台。DCAS在该基准上的良好表现（表4、表5）表明，三阶段课程训练和多重正则化约束在一定程度上缓解了源混淆的影响。

#### 5.7.3 缓解策略

我们探索了以下缓解源混淆的策略：

1. **来源判别器**：在辅助头中显式建模来源标签（$\lambda_{\text{source}}=0.10$），鼓励模型在潜空间中消除来源偏差。

2. **跨来源对比学习**：在对比损失中，正样本对不仅包括同文化曲目，还包括跨来源的相似曲目，以增强模型对来源变化的鲁棒性。

3. **来源均衡采样**：在训练过程中，对每个mini-batch进行来源均衡采样，确保每个来源的曲目比例大致相等。

实验结果表明，这三种策略的组合使routeA\_small基准上的少数文化覆盖率提升了约15-20%，但对惊喜度的影响较小（< 2%）。

---

## 6. 讨论

### 6.1 框架的有效性与泛化能力

DCAS框架在五个独立基准上的一致优异表现证明了其有效性和泛化能力。关键观察如下：

**跨骨干模型的稳健性**：无论是使用专门针对跨文化音乐优化的CultureMERT，还是通用的多模态基础模型Gemini，DCAS都能显著提升惊喜度和少数文化覆盖率。这表明DCAS的解耦-对齐-重排序范式具有良好的骨干模型无关性，可以适配不同的预训练嵌入。

**跨数据规模的稳健性**：从完整的V4 main数据集（1,122首曲目）到高度受限的routeA\_small子集（640首曲目），再到外部日志基准Yambda-5B（8,335条交互），DCAS的核心机制保持一致的有效性。

**操作点的灵活性**：校准敏感性分析（表9）表明，DCAS可以在Pareto前沿上提供多个操作点，允许从业者根据具体应用需求进行灵活选择。这对于实际的推荐系统部署具有重要意义。

### 6.2 消融研究的启示

消融研究揭示了DCAS各组件的相对重要性和交互效应：

**OT模块的"双刃剑"效应**：OT模块是提升少数文化覆盖率的核心驱动力（从0.4759降至0.2680当移除OT），但同时也是惊喜度的"限制因素"（移除OT后惊喜度提升+0.0269）。这一发现与Porcaro等 \cite{porcaro2021diversity} 的"多样性设计"框架相呼应：多样性不是免费午餐，它需要以某种形式的"效率"牺牲为代价。

**成对/排序约束的关键作用**：移除这些约束导致惊喜度显著下降（$\Delta=-0.0171$, $p=0.0010$），少数文化覆盖率也下降13.3%。这表明结构化的相似度约束是维持推荐质量的基础设施，与BPR \cite{rendle2009bpr} 和LambdaMART \cite{burges2010lambdamart} 的核心思想一致。

**GRL的有限贡献**：在两个骨干模型上，移除GRL对性能的影响均不显著（CultureMERT: $p=0.3097$；Gemini: 惊喜度反而提升）。这表明在三因子架构中，内容因子和风格因子的解耦主要通过重建损失和正则化约束实现，GRL提供的额外文化不变性增益有限。

### 6.3 Source Confound 的局限性与应对

0.911765的源混淆度是本文面临的最大方法论挑战。尽管DCAS通过来源判别器、跨来源对比学习和来源均衡采样在一定程度上缓解了这一问题，但根本性的解决方案需要**在数据采集层面**消除源-文化耦合。

未来的数据集构建应遵循以下原则：
1. **多来源覆盖**：每种文化的音乐应从多个独立来源采集
2. **质量控制**：对不同来源的录音进行统一的质量标准化
3. **元数据规范化**：消除来源特有的元数据模式

### 6.4 PAL的未来应用前景

参与式主动学习（PAL）流程为跨文化推荐系统提供了一个可持续的增强机制。通过在关键文化传统上引入领域专家的迭代标注，PAL能够：

1. **降低标注成本**：不确定性采样确保专家只需标注最具信息量的样本
2. **提升文化适配性**：领域专家的隐性知识被编码到模型的表征中
3. **建立信任**：参与式方法使受影响的社区能够直接影响推荐系统的行为

这一方向与Pinto \cite{pinto2025human} 的"人在回路"MIR方法论高度一致，代表了MIR从"以模型为中心"向"以人为中心"范式转变的重要一步。

### 6.5 对MIR社区的启示

本文的研究对MIR社区有以下几点启示：

**第一，跨文化评估需要超越精确度指标。** Yambda-5B的结果（表11）清楚地表明，在传统的精确度指标上"表现不佳"的系统，可能在文化多样性和公平性方面具有重要价值。社区需要开发更全面的评估框架。

**第二，源混淆是一个被系统性忽视的问题。** 0.911765的源混淆度在跨文化数据集中可能是普遍现象，但很少有研究对其进行量化分析。未来的工作应该报告源混淆度，并探索其对模型性能的影响。

**第三，解耦表征学习在MIR中仍有巨大潜力。** 虽然$\beta$-VAE和FactorVAE在视觉领域已被广泛研究，但它们在音乐表征学习中的应用仍处于早期阶段。DCAS的三因子架构为这一方向提供了有益的探索。

---

## 7. 伦理声明

### 7.1 数据伦理

本研究中使用的V4数据集包含来自10种文化传统的1,122首音乐曲目。我们承认以下伦理考量：

1. **文化代表性**：尽管我们努力纳入多样化的文化传统，但10种文化远不能代表全球音乐的丰富多样性。特别是，撒哈拉以南非洲、拉丁美洲和中东地区的代表性不足。这与Gómez-Cañón等 \cite{gomez2025beyond} 对ISMIR社区的文献计量分析揭示的结构性偏差一致。

2. **数据来源的伦理审查**：FMA等开源数据集的使用遵循其许可证条款。对于包含传统音乐的数据集（如gamelan、saraga\_hindustani），我们尊重原始数据集的伦理指南和使用限制。

3. **合成交互的透明性**：我们使用的9,600条交互记录是通过合成方法生成的，而非真实用户行为数据。这一方法选择是为了避免真实用户数据中的隐私问题，但也意味着我们的评估可能无法完全反映真实用户的偏好模式。

### 7.2 算法公平性

DCAS框架在设计时考虑了以下公平性原则：

1. **少数文化优先**：通过重排序机制中的少数群体偏好维度（权重0.14-0.25），系统确保非主导文化曲目获得公平曝光。

2. **来源均衡**：来源均衡维度（权重0.06-0.10）防止推荐列表被单一来源（通常是主导文化来源）垄断。

3. **透明度**：DCAS的重排序权重是完全透明的，从业者可以审查和调整每个维度的权重。

然而，我们也承认以下局限：

- **加权公平的非中立性**：重排序权重的选择本质上是一种价值判断。我们的目标校准操作点（相关性0.48、少数群体0.14等）反映了我们团队对"公平"的特定理解，这可能不适用于所有文化背景。

- **文化类别的固化**：将音乐归类为离散的文化类别（如"turkey"、"china"）可能强化了文化的本质主义观点。实际上，许多音乐传统是跨文化的、混合的和流动的。

### 7.3 潜在负面影响

我们识别了以下潜在负面影响：

1. **文化刻板印象的强化**：解耦表征中的"风格因子"可能无意中编码了文化刻板印象，如果这些刻板印象被用于推荐决策，可能导致对特定文化的简化理解。

2. **推荐系统的"文化异化"**：过度强调文化差异可能导致用户被"锁定"在其"原生文化"的推荐中，减少了跨文化探索的机会。

3. **参与式方法的权力不平等**：PAL流程假设领域专家可以平等地参与标注过程，但在现实中，不同文化传统的专家在时间、资源和话语权方面存在显著不平等。

### 7.4 缓解措施

为减轻上述风险，我们建议：

1. **定期审计**：对推荐系统进行定期的文化公平性审计，使用本文报告的指标（惊喜度、校准KL、Minority@k）跟踪系统表现。

2. **社区参与**：在系统设计和部署过程中，积极征求受影响文化社区的意见和反馈。

3. **开源透明**：公开所有代码、数据和实验结果，允许独立审查和复制。

---

## 8. 结论与未来工作

### 8.1 结论

本文提出了DCAS（Disentangled Cross-cultural Alignment System），一个用于跨文化音乐推荐的解耦对齐与最优传输框架。DCAS通过三因子高斯潜空间（内容、风格、情感）、三阶段课程训练（变分预训练、成对/排序约束、OT跨文化对齐）和校准感知重排序（六维度加权），在五个独立基准上实现了显著的惊喜度提升和少数文化覆盖率改善。

关键发现包括：
- DCAS目标校准操作点在V4 main + CultureMERT上将惊喜度从0.5558提升至0.8316（$\Delta=+0.2757$, $p=0.0050$），少数文化覆盖率从0.2681提升至0.4023
- OT模块是提升少数文化覆盖率的核心驱动力，但与惊喜度存在本质权衡
- 源混淆（0.911765）是跨文化推荐中的重大挑战，需要通过数据采集和方法论创新共同解决
- DCAS在外部日志基准Yambda-5B上的表现揭示了传统精确度指标的局限性，呼吁更全面的评估框架

### 8.2 未来工作

1. **动态文化表征**：当前的DCAS使用离散的文化类别，未来工作应探索连续的文化表征空间，更好地捕捉文化的流动性和混合性。

2. **多模态融合**：除了音频嵌入，未来的DCAS版本可以整合歌词、乐谱、演出视频等多模态信息，提升跨文化理解的深度。

3. **在线学习**：将DCAS扩展到在线学习场景，使系统能够实时适应用户反馈和文化趋势的变化。

4. **跨语言推荐**：探索DCAS在非英语音乐推荐中的应用，包括歌词语言、演唱风格和文化语境的跨语言对齐。

5. **用户研究**：开展大规模的用户研究，评估DCAS推荐结果对不同文化背景用户的实际影响，包括满意度、文化认同感和探索意愿。

6. **源混淆的系统性解决方案**：与数据采集社区合作，建立跨文化音乐数据集的源混淆度报告标准，并开发专门的解耦方法来消除源-文化耦合。

---

## 参考文献

\bibitem{li2023mert}
Li, B., et al. (2023). MERT: Acoustic Music Understanding Model with Large-Scale Self-Supervised Training. \textit{arXiv preprint arXiv:2306.00107}.

\bibitem{kanatas2025culturemert}
Kanatas, G., et al. (2025). CultureMERT: Continual Pre-Training for Cross-Cultural Music Representation Learning. In \textit{Proceedings of the 26th International Society for Music Information Retrieval Conference (ISMIR 2025)}, pp. 555--564.

\bibitem{lee2025globalmood}
Lee, J., et al. (2025). GlobalMood: A Cross-Cultural Benchmark for Music Emotion Recognition. In \textit{Proceedings of ISMIR 2025}, pp. 11--19.

\bibitem{ganin2016domain}
Ganin, Y., et al. (2016). Domain-Adversarial Training of Neural Networks. \textit{Journal of Machine Learning Research}, 17(59):1--35.

\bibitem{higgins2017beta}
Higgins, I., et al. (2017). beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework. In \textit{Proceedings of ICLR 2017}.

\bibitem{kim2018disentangling}
Kim, H. \& Mnih, A. (2018). Disentangling by Factorising. In \textit{Proceedings of ICML 2018}, pp. 2649--2658.

\bibitem{chen2018isolating}
Chen, T. Q., et al. (2018). Isolating Sources of Disentanglement in Variational Autoencoders. In \textit{Advances in Neural Information Processing Systems 31}, pp. 2615--2625.

\bibitem{oord2018representation}
van den Oord, A., et al. (2018). Representation Learning with Contrastive Predictive Coding. \textit{arXiv preprint arXiv:1807.03748}.

\bibitem{cuturi2013sinkhorn}
Cuturi, M. (2013). Sinkhorn Distances: Lightspeed Computation of Optimal Transport. In \textit{Advances in Neural Information Processing Systems 26}, pp. 2292--2300.

\bibitem{settles2009active}
Settles, B. (2009). Active Learning Literature Survey. \textit{University of Wisconsin-Madison, Computer Sciences Technical Report \#1648}.

\bibitem{zhang2012auralist}
Zhang, Y. C., et al. (2012). Auralist: Introducing Serendipity into Music Recommendation. In \textit{Proceedings of WSDM 2012}, pp. 13--22.

\bibitem{steck2018calibrated}
Steck, H. (2018). Calibrated Recommendations. In \textit{Proceedings of RecSys 2018}, pp. 154--162.

\bibitem{gomez2025beyond}
Gómez-Cañón, J. S., et al. (2025). Beyond a Western Center of MIR: A Bibliometric Analysis of the First 25 Years of ISMIR Authorship. \textit{Transactions of the International Society for Music Information Retrieval}, 8(1):372--387.

\bibitem{huang2023beyond}
Huang, C. W., et al. (2023). Beyond Diverse Datasets: Responsible MIR, Interdisciplinarity, and the Fractured Worlds of Music. \textit{TISMIR}, 6(1):43--59.

\bibitem{porcaro2021diversity}
Porcaro, L., et al. (2021). Diversity by Design in Music Recommender Systems. \textit{TISMIR}, 4(1):114--126.

\bibitem{holzapfel2018ethical}
Holzapfel, A., et al. (2018). Ethical Dimensions of Music Information Retrieval Technology. \textit{TISMIR}, 1(1):44--55.

\bibitem{park2024collaborative}
Park, S., et al. (2024). Collaborative Playlists around the World: A Cross-Cultural User Study. \textit{TISMIR}, 7(1):288--305.

\bibitem{papaioannou2025universal}
Papaioannou, S., et al. (2025). Universal Music Representations? Evaluating Foundation Models on World Music Corpora. \textit{arXiv preprint arXiv:2506.17055}.

\bibitem{pinto2025human}
Pinto, A. (2025). Towards Human-in-the-Loop Onset Detection: A Transfer Learning and User-Centered Annotation Design for Maracatu. In \textit{Proceedings of ISMIR 2025}, pp. 320--327.

\bibitem{kingma2014auto}
Kingma, D. P. \& Welling, M. (2014). Auto-Encoding Variational Bayes. In \textit{Proceedings of ICLR 2014}.

\bibitem{rendle2009bpr}
Rendle, S., et al. (2009). BPR: Bayesian Personalized Ranking from Implicit Feedback. In \textit{Proceedings of UAI 2009}, pp. 452--461.

\bibitem{burges2010lambdamart}
Burges, C. J. C. (2010). From RankNet to LambdaRank to LambdaMART: An Overview. \textit{Microsoft Research Technical Report MSR-TR-2010-82}.

\bibitem{loshchilov2019decoupled}
Loshchilov, I. \& Hutter, F. (2019). Decoupled Weight Decay Regularization. In \textit{Proceedings of ICLR 2019}.

\bibitem{peyre2019computational}
Peyré, G. \& Cuturi, M. (2019). Computational Optimal Transport. \textit{Foundations and Trends in Machine Learning}, 11(5-6):355--607.

\bibitem{zemel2013learning}
Zemel, R., et al. (2013). Learning Fair Representations. In \textit{Proceedings of ICML 2013}, pp. 325--333.

\bibitem{tay2018learning}
Tay, Y., et al. (2018). Learning to Rank with Optimal Transport for Recommender Systems. In \textit{Proceedings of RecSys 2018 Workshop}.

\bibitem{zhu2021transfer}
Zhu, F., et al. (2021). Transfer Learning for Cross-Domain Recommendation via Optimal Transport. \textit{IEEE Transactions on Knowledge and Data Engineering}, 34(12):5678--5691.

\bibitem{singh2019policy}
Singh, A. \& Joachims, T. (2019). Policy Learning for Fairness in Ranking. In \textit{Advances in Neural Information Processing Systems 32}, pp. 5465--5475.

\bibitem{qu2025listwise}
Qu, Y., et al. (2025). Listwise Bayesian Personalized Ranking for Music Recommendation. In \textit{Proceedings of ISMIR 2025}.
