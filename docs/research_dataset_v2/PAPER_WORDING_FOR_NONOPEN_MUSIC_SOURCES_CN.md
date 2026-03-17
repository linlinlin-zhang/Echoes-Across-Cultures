# 论文中如何表述不开源音乐来源

更新时间：2026-03-16  
适用范围：论文正文、附录、数据集说明、伦理声明。

---

## 1. 数据集章节推荐写法

### 中文模板

本研究的数据由开放音频源与受限研究访问音频源共同构成。  
开放音频源在论文和项目文档中逐项列出。  
受限音频源主要来自需要注册、机构访问、gated approval 或研究申请的馆藏与数据服务。由于版权或访问条款限制，我们不再分发这些原始音频文件，但会在许可允许范围内公开元数据、处理脚本、统计信息与实验配置。

### 英文模板

The dataset used in this study combines openly accessible audio sources and restricted-access research audio sources. Open sources are cited directly in the paper and project documentation. Restricted sources include archives, gated datasets, and research-access collections that cannot be publicly redistributed. We therefore do not release those raw audio files, but provide metadata where permitted, preprocessing scripts, dataset manifests, and experiment configurations.

---

## 2. 方法或实现章节推荐写法

### 中文模板

对于受限来源的音频，本研究仅在非商业学术研究范围内进行本地处理，并将其转换为统一的中间表示用于训练与评测。原始音频不进入公开仓库，也不作为公开数据集的一部分再次发布。

### 英文模板

For restricted-access audio sources, raw media were processed locally under non-commercial academic research conditions and converted into a unified intermediate representation for training and evaluation. The raw audio files are not redistributed and are not included in any public release of the dataset.

---

## 3. 限制性说明推荐写法

### 中文模板

由于部分文化域依赖受限研究访问音频源，当前版本的数据集无法实现完整原始音频公开复现。为减轻这一限制，我们提供来源说明、可公开元数据、预处理脚本和实验配置，并在数据说明文档中记录访问条件与限制。

### 英文模板

Because some cultural domains rely on restricted-access research audio sources, the current dataset cannot be fully redistributed as raw audio. To mitigate this limitation, we provide source documentation, releasable metadata, preprocessing scripts, and experiment configurations, together with notes on access conditions and restrictions.

---

## 4. 不建议的写法

不要写：

- “数据来自若干内部来源”  
  问题：过于模糊，无法判断合法性与可复现性。

- “由于版权原因，数据不公开”  
  问题：只说结果，不说来源类型和访问条件，透明度不够。

- “我们使用了公开网络音频，但不提供链接”  
  问题：会让读者怀疑来源合法性和研究可信度。

---

## 5. 当前项目可直接使用的推荐段落

### 中文版

本研究的跨文化音乐数据同时包含开放音频源与受限研究访问音频源。对于开放音频源，我们在项目文档中逐项列出数据来源与许可信息；对于受限音频源，我们仅在合法的非商业学术研究条件下访问原始音频，并不对外再分发这些原始文件。为保证研究透明度，我们公开来源说明、元数据字段、预处理脚本、embedding 构建流程和实验配置，使读者能够理解实验设置并在许可允许的范围内部分复现流程。

### 英文版

The cross-cultural music data used in this study include both openly accessible audio sources and restricted-access research audio sources. For open sources, we document the dataset provenance and licensing information directly in the project materials. For restricted sources, raw audio is accessed only under lawful non-commercial academic research conditions and is not publicly redistributed. To preserve transparency, we release source documentation, metadata schemas, preprocessing scripts, the embedding-construction pipeline, and experiment configurations so that the setup remains inspectable even when some raw media cannot be shared.
