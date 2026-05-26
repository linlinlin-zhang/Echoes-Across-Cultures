# 项目当前状态说明

> 更新日期：2026-05-24  
> 当前分支：`feature/research-v2-platform-and-results`

---

## 1. 代码仓库状态

本分支已领先远程仓库 **6 个 commit**。当前工作区有本轮数据采集与 embedding 管线相关的未提交修改。

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
| **Apple iTunes Search API** | 可用，已通过 smoke test | 无需 API Key，返回 30 秒 AAC 预览（`.m4a`）。2026-05-23 已修复“只收集不下载”的集合标记 bug，并验证可实际写入 `.m4a`。 |
| **Jamendo API** | 可用，扩展抓取中 | Creative Commons 授权，可合法下载完整 MP3。Client ID 已验证通过；`run_jamendo_crawl.ps1` 与爬虫脚本均优先读取环境变量 `JAMENDO_CLIENT_ID`。 |
| **Free Music Archive (FMA)** | 备选补充 | 可通过 HuggingFace 下载大规模 CC 音乐数据集，用于补充冷门文化区域。 |

### 2.2 已就绪的爬虫脚本

| 脚本 | 路径 | 功能 |
|---|---|---|
| `crawl_itunes_previews.py` | `dcas/scripts/` | iTunes 30s 预览抓取，支持按国家/文化分桶、断点续传 |
| `crawl_jamendo.py` | `dcas/scripts/` | Jamendo CC 音乐抓取，支持文化标签映射、断点续传 |
| `crawl_spotify_previews.py` | `dcas/scripts/` | Spotify 元数据扫描（预览功能已失效） |
| `merge_spotify_jamendo_metadata.py` | `dcas/scripts/` | Spotify/Jamendo 元数据对齐、去重、文化别名统一化 |
| `merge_metadata_dedup.py` | `dcas/scripts/` | 通用多源 metadata 合并与去重；当前 `run_merge_metadata.ps1` 使用它合并 iTunes/Jamendo/Spotify 可用输入 |

所有脚本均遵循相同的**断点续传（checkpoint/resume）**架构：输出目录内保存 `state.json` + `metadata.csv`，中断后可无缝恢复。

### 2.3 启动脚本

| 脚本 | 对应爬虫 | 状态 |
|---|---|---|
| `run_itunes_crawl.ps1` | iTunes | 安全，可直接使用 |
| `run_itunes_nonwestern_supervisor.ps1` | iTunes | 扩展文化域补量守护脚本；当前按 iTunes+Jamendo 合并计数，中国 1000，其余域 700 |
| `run_jamendo_crawl.ps1` | Jamendo | 需用户填写 `JAMENDO_CLIENT_ID` |
| `run_merge_metadata.ps1` | merge | 安全，自动检测输入文件 |
| `run_spotify_crawl.ps1` | Spotify | **含真实凭证，勿提交 git** |

---

## 3. 已知问题与阻塞项

### 3.1 iTunes 爬虫下载逻辑 bug（已修复）

- **现象**：`total_collected` 正常增长，但 `total_downloaded` 始终为 0。
- **根因**：新解析出的 `track_id` 在加入待下载列表时被提前加入 `downloaded_set`，导致下载批处理认为这些记录已经下载并全部跳过。
- **修复**：分离 `seen_set` 与 `downloaded_set`；只有实际下载成功后才写入 `downloaded_set` / `downloaded_track_ids`。
- **验证**：2026-05-23 在 `storage/public/itunes_smoke_codex_source_20260523/` 运行小规模 smoke crawl，`target_total=2`，结果 `total_collected=2`、`total_downloaded=2`，metadata 指向的 `.m4a` 文件均存在。
- **当前小样本**：已在 `storage/public/itunes_crawl/` 抓取 10 条预览，覆盖 `west/japan/korea/india/brazil` 各 2 条；已合并到 `storage/public/merged/metadata_merged.csv`。
- **当前大规模任务状态**：2026-05-24 第一轮 iTunes 非西方补量已完成，`storage/public/itunes_crawl/metadata.csv` 按 `track_id` 去重为 14667 首；iTunes 侧已达到 China 700、Japan/Korea/India/Brazil/Latin/Africa/Middle East/Southeast Asia 各 500。
- **韧性修复**：`crawl_itunes_previews.py` 已改为单个 query 请求失败时保存 checkpoint 并跳过该 query，不再让整轮任务直接崩掉；未完成 query 不会写入 `completed_queries`，后续守护脚本可重试。
- **扩展补量目标**：2026-05-24 已将 supervisor 升级为按 iTunes+Jamendo 合并计数判断缺口：China 目标 1000，其余所有跟踪文化域目标 700。新增 6 个文化域：`nordic`、`eastern_europe`、`balkans`、`caribbean`、`andean`、`central_asia`；同时将已有低量 `celtic` 补到 700。
- **中国方言补量**：中国文化域 query pool 已加入 Cantonese、Hakka、Hokkien/Taiwanese Hokkien、Minnan、Teochew、Shanghainese、Sichuan dialect、Wu Chinese、Yue Chinese 等关键词。
- **第二阶段 3 万首目标**：2026-05-24 已启动新一轮 iTunes 非西方补量，west 不再补。当前总目标为 30000 首：west 保持现有 10802 首，16 个非西方文化域合计 19198 首，单域目标为 1199-1200 首。
- **当前后台任务**：`scripts/run_itunes_30k_nonwest_then_embedding.sh` 正在运行；活动日志记录在 `storage/public/merged/nonwest_30k_to_embedding_active_log.txt` 指向的文件。完成后会自动 merge、补封面/播放链接字段，并重启 CultureMERT embedding。
- **续跑注意**：旧的 `scripts/run_itunes_balanced_world_crawl.sh` 仍保留按 country 当前数量补到 `PER_COUNTRY` 的能力；文化域补量优先使用新的 supervisor，避免 west 继续扩大。

### 3.2 Spotify 彻底无法获取预览

- 这不是脚本 bug，而是 Spotify 平台级政策变更。
- 当前 `crawl_spotify_previews.py` 可收集歌曲元数据（曲名、艺人、专辑、发行日期等），但无法下载音频。
- 建议：仅作为 Jamendo/iTunes 元数据的补充对齐来源，不依赖其实际抓音功能。

### 3.3 Jamendo 凭证与扩展状态

- Jamendo Client ID 已验证可用，API 返回 `status=success`。
- 2026-05-23 已完成一轮 Jamendo 采集：77 个查询槽全部跑完，`storage/public/jamendo_crawl/metadata.csv` 按 `track_id` 去重为 1742 首，`state.json` 记录 `total_collected=1784`、`total_downloaded=1742`、失败下载 42 个。
- 主要分布：west 835、latin 289、brazil 130、celtic 116、india 110、middle_east 94、japan 61、africa 53、china 39、southeast_asia 10、korea 5。
- 2026-05-24 已扩展 Jamendo 文化域配置到 17 个：`west`、`china`、`korea`、`japan`、`india`、`latin`、`brazil`、`africa`、`middle_east`、`southeast_asia`、`celtic`、`nordic`、`eastern_europe`、`balkans`、`caribbean`、`andean`、`central_asia`。
- 中国 Jamendo 标签已补充 Cantonese、Hakka、Hokkien/Taiwanese、Minnan、Teochew、Shanghainese 等方言/区域关键词。
- Jamendo 扩展抓取已完成，`storage/public/jamendo_crawl/metadata.csv` 当前按 `track_id` 去重为 2551 首，覆盖 17 个文化域；其中新增补充较明显的是 Caribbean、Celtic、Eastern Europe、Balkans、Andean、Nordic 等。
- 也可用环境变量方式运行：

```powershell
$env:JAMENDO_CLIENT_ID = "your_client_id"
.\run_jamendo_crawl.ps1
```

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
- [x] **修复 `crawl_itunes_previews.py` 的下载 bug**，验证能实际写入 `.m4a` 文件。
- [x] **申请 Jamendo Client ID** 并执行小规模 Jamendo 抓取测试。

### 高优先级（构建数据集）
- [x] 使用 iTunes supervisor 继续扩展文化域：中国至少 1000 首，其余跟踪文化域至少 700 首，并覆盖 6 个新增文化域。
- [x] 运行 `merge_metadata_dedup.py` 进行 iTunes/Jamendo 可用输入合并。
- [ ] 当前执行第二阶段 3 万首补量：west 保持现状，剩余名额均匀分配给非西方文化域。
- [ ] 补量完成后由 `scripts/run_itunes_30k_nonwest_then_embedding.sh` 自动重跑合并、补 metadata 媒体字段并启动正式 **CultureMERT 嵌入**（`ntua-slp/CultureMERT-95M`，mean pooling，30s 截断）。
- [x] CultureMERT embedding smoke test 已通过：`storage/public/merged/tracks_culturemert_smoke3.npz`，3 首、768 维、0 错误。
- [x] 已新增 metadata enrichment：`dcas/scripts/enrich_metadata_media_links.py` 会回填/统一 `cover_art_url`、`cover_art_url_large`、`platform_track_url`、`platform_album_url`、`external_url`、`full_track_url`、`audio_is_preview` 等字段。

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
| 通用元数据合并 | `dcas/scripts/merge_metadata_dedup.py` |
| metadata 封面与播放链接补全 | `dcas/scripts/enrich_metadata_media_links.py` |
| Jamendo 完成后自动 merge + embedding | `scripts/run_post_jamendo_embedding.sh` |
| 3 万首非西方均衡补量 + enrich + embedding | `scripts/run_itunes_30k_nonwest_then_embedding.sh` |
| Spotify/Jamendo 专用合并 | `dcas/scripts/merge_spotify_jamendo_metadata.py` |
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
