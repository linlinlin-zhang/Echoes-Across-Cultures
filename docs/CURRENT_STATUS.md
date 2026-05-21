# 项目当前状态说明

> 更新日期：2026-05-22  
> 当前分支：`feature/research-v2-platform-and-results`

---

## 1. 代码仓库状态

本分支已领先远程仓库 **6 个 commit**，刚刚完成推送。工作区干净，无未提交更改。

### 最近提交概览

| Commit | 类别 | 说明 |
|---|---|---|
| `d1ccc44` | research | human pilot 盲听实验包（脚本、结果、志愿者页面） |
| `a621d33` | ui | web 前端示例原型（专辑封面、参数图表、地图界面） |
| `1e35ea8` | docs | ISMIR 2026 投稿文档（摘要、标题、pilot protocol） |
| `207351c` | tooling | PowerShell 启动脚本（iTunes / Jamendo / merge） |
| `8e49772` | data | 多源音乐爬虫与元数据合并脚本 |
| `9b281b9` | chore | 清理旧版 `web_prototype` 并更新 `.gitignore` |

### 被排除提交的文件（出于安全/体积考虑）

- `run_spotify_crawl.ps1`：**包含真实 Spotify Client ID/Secret**，已加入 `.gitignore`，请勿手动提交。
- `human_pilot_package/**/*.zip`：participant 个人包与共通包总计约 **360MB**，不适合进入 git。
- `human_pilot_package/volunteer_site/audio/`：36 段 MP3 试听片段约 **18MB**，由实验脚本本地生成/复制，已排除。

---

## 2. 数据收集策略与现状

### 2.1 数据源评估结论

| 数据源 | 状态 | 说明 |
|---|---|---|
| **Spotify Web API** | 基本不可用 | 2024年11月政策变更后，`preview_url` 完全不再返回；Development Mode 搜索上限仅 10 条/请求；音频特征和 popularity 字段亦被移除。脚本保留仅作元数据备用。 |
| **Apple iTunes Search API** | 可用，需修复 bug | 无需 API Key，返回 30 秒 AAC 预览（`.m4a`）。爬虫已写好，但存在 **下载计数始终为 0** 的 bug，待修复。 |
| **Jamendo API** | 推荐主力 | Creative Commons 授权，可合法下载完整 MP3。需自行申请 Client ID。脚本已就绪。 |
| **Free Music Archive (FMA)** | 备选补充 | 可通过 HuggingFace 下载大规模 CC 音乐数据集，用于补充冷门文化区域。 |

### 2.2 已就绪的爬虫脚本

| 脚本 | 路径 | 功能 |
|---|---|---|
| `crawl_itunes_previews.py` | `dcas/scripts/` | iTunes 30s 预览抓取，支持按国家/文化分桶、断点续传 |
| `crawl_jamendo.py` | `dcas/scripts/` | Jamendo CC 音乐抓取，支持文化标签映射、断点续传 |
| `crawl_spotify_previews.py` | `dcas/scripts/` | Spotify 元数据扫描（预览功能已失效） |
| `merge_spotify_jamendo_metadata.py` | `dcas/scripts/` | 多源元数据对齐、去重、文化别名统一化 |

所有脚本均遵循相同的**断点续传（checkpoint/resume）**架构：输出目录内保存 `state.json` + `metadata.csv`，中断后可无缝恢复。

### 2.3 启动脚本

| 脚本 | 对应爬虫 | 状态 |
|---|---|---|
| `run_itunes_crawl.ps1` | iTunes | 安全，可直接使用 |
| `run_jamendo_crawl.ps1` | Jamendo | 需用户填写 `JAMENDO_CLIENT_ID` |
| `run_merge_metadata.ps1` | merge | 安全，自动检测输入文件 |
| `run_spotify_crawl.ps1` | Spotify | **含真实凭证，勿提交 git** |

---

## 3. 已知问题与阻塞项

### 3.1 iTunes 爬虫下载逻辑 bug

- **现象**：`total_collected` 正常增长，但 `total_downloaded` 始终为 0。
- **根因推测**：`_parse_track` 中的 `preview_url` 过滤逻辑或下载批处理的集合判断可能存在问题。此前调试时对该函数进行过修改，恢复后可能未完全复原。
- **优先级**：高——这是目前获取主流商业音乐预览的最可行通道。

### 3.2 Spotify 彻底无法获取预览

- 这不是脚本 bug，而是 Spotify 平台级政策变更。
- 当前 `crawl_spotify_previews.py` 可收集歌曲元数据（曲名、艺人、专辑、发行日期等），但无法下载音频。
- 建议：仅作为 Jamendo/iTunes 元数据的补充对齐来源，不依赖其实际抓音功能。

---

## 4. 前端开发准备

### 4.1 已提交的示例原型

位于 `web/example/`，共 4 个独立 HTML 文件：

1. **专辑封面展示示例** — 网格/轮播式封面浏览界面
2. **参数及其图表界面展示示例1/2** — 两种不同风格的推荐参数可视化布局
3. **地图界面示例** — 基于地理/文化的音乐探索地图

这些是前期快速验证 UI 方向的单文件原型，尚未接后端真实数据。

### 4.2 下一步前端方向

- 选型尚未最终确定（目前无框架锁定，示例为原生 HTML/CSS/JS）。
- 需要与实际的 CultureMERT 嵌入服务和推荐 API 对接。

---

## 5. Human Pilot 实验

### 5.1 实验内容

- **目的**：为 ISMIR 论文补充 small blind listener sanity check，缓解纯合成交互信号的局限性。
- **规模**：10 位志愿者，每人 12 组 A/B 盲听任务。
- **对比方法**：BPR listwise hybrid（基线） vs PAL OT calibrated P3 balanced（实验方法）。

### 5.2 已有结果

- 已回收 8 位参与者的 CSV 结果（P01, P02, P03, P05, P06, P07, P09, P10）。
- 分析脚本位于 `human_pilot_package/researcher_private/analyze_responses.py`。
- 统计指标：`compatible_choice`（兼容性）、`discovery_choice`（发现性）、`overall_choice`（总体偏好）。

---

## 6. 下一步行动清单（建议优先级）

### 紧急（阻塞数据流）
- [ ] **修复 `crawl_itunes_previews.py` 的下载 bug**，验证能实际写入 `.m4a` 文件。
- [ ] **申请 Jamendo Client ID** 并执行小规模 Jamendo 抓取测试。

### 高优先级（构建数据集）
- [ ] 并行运行 iTunes + Jamendo 爬虫，收集第一批跨文化音乐样本（目标：各文化至少 500~1000 首）。
- [ ] 运行 `merge_spotify_jamendo_metadata.py`（如有 Spotify 元数据）或 iTunes/Jamendo 两源合并。
- [ ] 使用 `build_tracks_from_audio.py` 生成 **CultureMERT 嵌入**（`ntua-slp/CultureMERT-95M`，mean pooling，30s 截断）。

### 中优先级（前端与系统）
- [ ] 基于 `web/example/` 的验证方向，确定正式前端技术栈（React/Vue/纯原生？）。
- [ ] 设计推荐服务 API 接口（输入：seed track / user profile；输出：ranked candidates with explanations）。
- [ ] 将 CultureMERT 嵌入接入近似最近邻搜索（FAISS / Annoy / ScaNN）。

### 低优先级（论文与完善）
- [ ] 完成剩余 2 位 human pilot 志愿者数据回收（P04, P08）。
- [ ] 将 pilot 统计结果写入 ISMIR 论文相关章节。

---

## 7. 关键文件速查

| 用途 | 路径 |
|---|---|
| iTunes 爬虫 | `dcas/scripts/crawl_itunes_previews.py` |
| Jamendo 爬虫 | `dcas/scripts/crawl_jamendo.py` |
| 元数据合并 | `dcas/scripts/merge_spotify_jamendo_metadata.py` |
| iTunes 启动器 | `run_itunes_crawl.ps1` |
| Jamendo 启动器 | `run_jamendo_crawl.ps1` |
| 合并启动器 | `run_merge_metadata.ps1` |
| 前端示例 | `web/example/*.html` |
| Human Pilot 说明 | `human_pilot_package/README_RESEARCHER_CN.md` |
| ISMIR 论文计划 | `docs/ISMIR2026_PROJECT_PLAN_CN.md` |
| 项目自查与执行计划 | `docs/PROJECT_V4_SELF_AUDIT_AND_EXECUTION_PLAN_2026-03-20_CN.md` |

---

## 8. 环境与依赖备忘

- **主工作目录**：`E:\Desktop\Echo`
- **Python 虚拟环境**：项目使用 `.venv` 或 `.venv-gpu`（`.gitignore` 已排除）。
- **大文件存储**：`storage/` 目录已全局排除，爬虫输出默认指向 `./storage/public/`，需本地手动创建。
- **GPU 依赖**：CultureMERT 嵌入生成建议在有 CUDA 的环境下运行，避免 CPU 推理过慢。
