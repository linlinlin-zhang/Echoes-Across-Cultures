# 艺人位置与多语言显示

本轮补查见 [2026-11-02 更新记录](ARTIST_LOCATIONS_UPDATE_2026-11-02_CN.md)：新增 24 组艺人、212 首曲目地区，线上抽查已通过。

歌曲中文描述保留，所有非中文界面共用已有的英文描述字典。
国家、城市、曲风、状态和界面标签按所选语言显示，不为每种语言重新生成歌曲描述。
描述生成缓存按 `zh` / `en` 分开；在不同非中文语言之间切换不会再次生成介绍。

网页在 head 中读取语言设置，并在首次显示正文前完成静态标签翻译。
旧版收藏及音乐台缓存中的中文地名和状态会重新从原始元数据生成。
旧音乐台缓存会从本地曲库接口刷新元数据，不调用付费 AI。

## 2026-11-02 平台地区回退

按用户更新要求，优先使用已核实的 25,996 首艺人地点；另有 3,961 首使用平台原始市场地区，标记 `catalog_market`，以首都或地区参考点显示。平台也没有地区的 43 首仍为未知。

平台回退按曲目 ID 独立保存于 `storage/public/merged/catalog_market_locations.json`，不混入艺人核实字典，不改写原始 CSV；艺人地点始终优先。原有 `artist_locations.report.json` 的覆盖统计仍只表示艺人地点覆盖，合并显示覆盖以平台回退文件的 `report` 为准。重新合并艺人资料后，运行 `scripts/build_catalog_market_locations.py` 更新回退清单。云端展示使用发布快照，需重新打包发布。

## 位置依据

`storage/public/merged/artist_locations.json` 保存有来源链接的艺人地点与曲目映射。
原始 CSV 中的 `country` 及音频文件不被改写。接口将原始市场保留为 `catalog_country`，
再按艺人地点覆盖显示用的 `country`、`city`、`lat`、`lng`。

来源与匹配规则：

- Wikidata：通过 [Apple Music 艺人 ID（P2850）](https://www.wikidata.org/wiki/Property:P2850)
  匹配；唯一、可核实的公开活动地或乐队成立地优先使用城市坐标。
- Jamendo：通过实际曲目 ID 查艺人 ID，再读取
  [艺人自报地点](https://developer.jamendo.com/v3.0/artists/locations)。
  城市查询限定在该国家内，排除非城市的同名对象；相邻的行政区与城市可合并为同一城市级位置。
  同一 Jamendo 艺人组的部分旧曲目下线时，仅在其余曲目明确指向唯一艺人 ID、
  且公开档案名与该组署名完全一致时，复用该已核实档案的位置；跨平台或多个 ID 的同名组不套用。
- MusicBrainz：对 Wikidata 已确认身份但国家有歧义的艺人，通过其 MusicBrainz ID 获取
  [艺人关联地区](https://musicbrainz.org/doc/Artist#Area)，不用出生地冒充当前活动地。
- 只有国家资料时，使用 Wikidata 中该国的首都，并标记 `country_capital`。
  查到城市时标记 `artist_city`；城市来源可能是活动地、乐队成立地或艺人自报地，
  `location_relation` 保留具体含义。这不代表实时居住位置。
- 不能确认身份或地点的条目不猜测。市场与文化属性保留用于检索和分类，
  不再拿来代替艺人所在地。未确认地点不画虚构坐标；无地点的上传音频也不显示太平洋标记。

每条记录带有来源链接，推荐卡片可点击“地点来源”。
示例：[Dario Junior](https://www.jamendo.com/artist/336788) 的自报地点为巴西 João Pessoa；
[Nick Gordy](https://www.jamendo.com/artist/478536) 为葡萄牙 Fundão。

## 更新数据

```bash
python3 scripts/enrich_artist_locations.py
python3 scripts/fetch_jamendo_artist_locations.py
python3 scripts/resolve_jamendo_artist_locations.py
python3 scripts/fetch_musicbrainz_artist_areas.py
python3 scripts/enrich_artist_locations.py
```

公开查询响应缓存位于 `storage/artist_location_cache/`，重复执行会复用。
Jamendo 使用现有的本机 client ID 配置，日志和缓存文件不包含该凭据。
MusicBrainz 请求遵守每秒最多一次的调用间隔。Wikidata 批量查询最多两个并发。
同一时间只运行一个会访问 MusicBrainz 的查询进程。
人工核实的补充可写入 `configs/artist_locations.reviewed.json`，必须包含来源 URL 和城市／国家精度。
最终覆盖数量在 `artist_locations.report.json` 中，未确认的艺人记录保留在 `unresolved` 中。
`artist_locations.unresolved.csv` 按剩余歌曲数量列出待查艺人、标准名称、示例曲目及身份核对状态。

坐标处理拒绝空值、布尔值、NaN 和越界值；合法的零纬度或零经度保留。
国家参考坐标来自已有 Natural Earth 地图，缺失的小岛参考 IANA 时区表；
这些参考点只用于明确标注的非艺人回退。地图换日线投影、重复位置标签和旧缓存居中逻辑也已修正。

## 浏览器回归检查

静态测试不请求真实 AI 服务：

```bash
python3 -m http.server 18040 --bind 127.0.0.1 --directory web
node scripts/check_web_i18n.cjs
```

Node 环境需能加载 Playwright 与 Chromium；可用 `PLAYWRIGHT_MODULE`、`CHROMIUM_PATH`
指定已有安装。检查覆盖七种语言、四个页面、首次可见帧、缓存迁移、位置字段、描述共用与零付费 AI 请求。

## 继续补齐剩余艺人

```bash
python3 scripts/fetch_canonical_artist_names.py
python3 scripts/search_remaining_artist_locations.py
python3 scripts/fill_artist_location_labels.py
python3 scripts/enrich_artist_locations.py
python3 scripts/fill_artist_location_labels.py --input storage/public/merged/artist_locations.json
```

该流程先通过 Apple Music 的艺人 ID 找到标准名称，避免把地区化译名或合作署名当作独立组合。
再查 MusicBrainz 的名称和别名，并以曲库内真实歌曲的艺人署名和标题验证身份。
名称匹配但歌曲无法核对的条目不会自动写入地理数据。
结果和核对状态保存在 `recording_verified_artist_locations.json`，重复执行跳过已核对条目；
`--retry-checked` 可重新检查。查询响应均缓存，不使用生成式 AI。
全量进度日志：`tmp/artist-locations/name-search-full.log`。
城市类型采用独立、限定候选地点的查询核验，避免资料库未返回类型字段时把城市全部降级成国家。
已保存歌曲身份与原始地区响应时，可用 `--locations-only --base <原始位置快照>` 单独重建地点。

批量搜索结果上限为 100；超过上限的组合查询会自动拆分，防止高产艺人的曲目挤掉其他结果。
对于历史查询缓存，可先恢复被截断的批次，再查别名与实际作品：

```bash
python3 scripts/repair_truncated_artist_search.py
python3 scripts/search_remaining_artist_locations.py --locations-only
python3 scripts/enrich_artist_locations.py
python3 scripts/search_artist_aliases.py
python3 scripts/search_remaining_artist_locations.py --locations-only --identity-input storage/artist_location_cache/alias_verified_candidates.json
python3 scripts/enrich_artist_locations.py
python3 scripts/fill_artist_location_labels.py --input storage/public/merged/artist_locations.json --missing-only
```

别名补查保留查询名单快照，以便网络中断后恢复相同批次；结果仍须符合“公开别名完全对应＋实际歌曲匹配＋身份唯一”。
若需针对未来新曲库重新扫描，可用 `--lookup-snapshot <新的缓存文件路径>` 保存新一轮名单。
Jamendo 整批空响应也不直接当成曲目已下线：

```bash
python3 scripts/fetch_jamendo_artist_locations.py --repair-missing
python3 scripts/resolve_jamendo_artist_locations.py --new-only
python3 scripts/enrich_artist_locations.py
```

空批次会逐步拆分重试；城市标签按 Wikidata 的多语言标签补齐，并复用 `mul` 共享名称。
描述字典不会因这些查询而重新翻译。

## 地图与线上检查

音乐地图优先使用本地 `web/vendor/countries-50m.json`，失败时回退到已有 110m 版本。
较细边界包含加那利群岛、冲绳等小岛，避免真实城市点看起来落在国家轮廓外。
国家显示补充了来源中使用的 XK 地区代码。未知位置不会通过市场或国家参考点变成已核实艺人位置。
收藏会保留城市、坐标、精度与来源；手机上可直接操作删除按钮，地图说明与状态文字各自换行。
音乐台默认优先请求可用 GPU，后端在没有可用 CUDA 设备时自动使用 CPU；
需要显式使用 CPU 时可在音乐台地址添加 `?cuda=0`。

```bash
python3 scripts/check_artist_location_data.py
node scripts/check_artist_map_coordinates.cjs
node scripts/check_map_edges.cjs
node scripts/check_live_web.cjs
```

坐标与边界交叉检查需系统 ISO 3166 数据，默认 `/usr/share/iso-codes/json/iso_3166-1.json`，
可用 `ISO_CODES_JSON` 指定其他安装位置。该检查只标记距来源国家边界超过 200 公里的疑点供复核，
不是城市级测绘精度保证。浏览器测试沿用上述 Playwright 环境变量，线上检查默认访问 `https://resonance.website`。
它在隔离的匿名会话中执行真实播放、推荐、收藏和设置操作，并禁止自动调用付费 AI。
录音测试使用浏览器提供的模拟麦克风，不采集用户真实音频。

AI 接口仅在服务器配置的地址使用服务器密钥；调用方选择其他 HTTPS 地址时须提供自己的密钥。
普通与流式接口共用相同的地址检查。可在现有 Windows Python 环境执行
`scripts/check_ai_endpoint_policy.py` 检查允许与拒绝情形，不访问上游服务；
`scripts/check_ai_interface.cjs` 用模拟响应验证流式界面、追问及失败恢复。

人工核实表也补充了公开官网及音乐机构资料，例如
[Lars-Luis Linek 的汉堡](https://larsluislinek.de/Impressum/)、
[DJ-Chart 的圣加仑](https://www.dj-chart.ch/who-s-dj-chart/)、
[哈萨克国家弦乐四重奏的阿拉木图](https://jcmf.or.jp/news_en/the-5th-prizewinner/)、
[Simon & Garfunkel 的纽约](https://www.artgarfunkel.com/context-history)。
同名城市必须核对国家／州；坐标只取城市中心，不保存个人或机构街道地址。
Patiala 的 Wikidata 坐标存在偏移，人工记录使用同一城市关联的 OpenStreetMap 城市节点校正。
