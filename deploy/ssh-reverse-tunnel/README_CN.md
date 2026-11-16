# Echo 内网穿透与巴西展览部署

当前入口：[resonance.website](https://resonance.website)，云服务器 `47.250.196.219`。HTTP 地址会转到 HTTPS 域名。

2026-11-02 已改为云端直接提供页面、只读曲库及全部 30,000 首压缩试听；推荐、上传分析、收藏与 AI 继续由中国主机提供。云端和中国主机通过两个受限 SSH 回环转发连接。

完整操作说明见 [巴西展览部署手册](../exhibition-edge/README_CN.md)，本轮测试见 [展览自检报告](EXHIBITION_CHECK_2026-11-02_CN.md)。

2026-11-16 起改为按需启动，`Echo-Intranet-Web` 登录启动任务已禁用。需要时双击项目根目录的 `启动网页应用.cmd`，用完双击 `停止网页应用.cmd` 或关闭维护窗口。运行期间自动重启异常应用、重连主／备用隧道并防止睡眠；启动后自动预热推荐模型，`/api/health` 的 `recommendation_state` 为 `ready` 后开始推荐演示。详见 [按需启动说明](MANUAL_START_CN.md)。

```powershell
Get-ScheduledTask -TaskName Echo-Intranet-Web
Get-Content tmp\intranet-web\health.json
Invoke-RestMethod https://resonance.website/api/health
Invoke-RestMethod https://resonance.website/api/edge/health
```

当前云端只读版本位于 `/srv/echo-exhibition/current`，由 `echo-exhibition.service` 运行。网页或曲库更新需要用 `scripts/package_exhibition_edge.py` 打包发布；中国主机上的 Python 修改需要先运行停止脚本，再双击启动脚本。旧的 GitHub Actions 部署 `/srv/echo`，不会更新展览快照。

艺人地点优先；未核实曲目现在按用户要求使用平台市场地区，并明确标注 `catalog_market`。参考 [地点规则](ARTIST_LOCATIONS_CN.md)。

切换前的原部署说明保留于 [历史记录](README_BEFORE_EXHIBITION_2026-11-02_CN.md)，其中管理员密钥运行单隧道、所有内容经中国主机代理的方式已由展览部署替换。
