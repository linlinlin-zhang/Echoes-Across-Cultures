# Echo 网页内网穿透部署

当前部署入口：<https://resonance.website>，云服务器 `47.250.196.219`。
也可以使用 `http://47.250.196.219`。HTTPS 请使用域名。

访问链路：浏览器 → 云端 Nginx（80/443）→ 云端 `127.0.0.1:18020`
→ SSH 反向隧道 → 本机 `127.0.0.1:8000` → 仓库中的 FastAPI 和 `web/`。
网页、音乐文件、收藏数据库和 CultureMERT/DCAS 推荐均由本机提供。
本机需要开机、联网并登录 Windows；关机或休眠期间网站不可用。

## 本机启动

依赖：Windows OpenSSH、`.venv-gpu` Python 环境、FFmpeg/FFprobe，以及
`storage/public/merged/` 和 `storage/models/` 中的本地数据和模型。

将 `configs/local_web.env.example` 复制为 `configs/local_web.env`，按需填写
已有的 API 配置。真实 `.env` 文件已被 Git 忽略。
完整网站必须使用 `ECHO_WORKER_REQUIRE_TOKEN=false`，并留空
`ECHO_MAINLINE_WORKER_URL`。原有 `configs/local_worker.env` 用于独立 worker，
不要直接用它启动完整网站。

首次连接时先核对服务器指纹，将其保存到当前 Windows 用户的 `known_hosts`。
运行脚本使用 `StrictHostKeyChecking=yes`，不会自动信任未知主机。

手动运行（PowerShell，在仓库根目录）：

```powershell
.\.venv-gpu\Scripts\python.exe scripts\run_intranet_web.py --ssh-key C:\lin-overseas.pem
```

安装登录后自动启动任务并立即启动：

```powershell
.\scripts\install_intranet_web_task.ps1
```

任务名为 `Echo-Intranet-Web`，以当前用户身份运行。任务不设最长运行时长，
允许接通电池时运行。监督进程每 5 秒检查子进程并重启退出的网页进程或 SSH。
SSH 每 15 秒发保活包，连续 3 次无响应会退出，然后由监督进程重新连接。
远端转发端口只监听云服务器的回环地址，无需新增公网端口。

`ECHO_SSH_FALLBACK_HOST` 可指定同一服务器的备用连接地址；当前部署为
`root@100.120.12.57`，使用已有 Tailscale 连接。公网地址连接失败时监督进程会切换地址。
两条通道都用主地址 `47.250.196.219` 的 `known_hosts` 记录校验服务器密钥，
不会因切换内网地址而跳过主机身份验证。

云端安装了本目录 `sshd-clientalive.conf` 到
`/etc/ssh/sshd_config.d/90-echo-clientalive.conf`，并在 `sshd -t` 通过后重载 `ssh` 服务。
新连接每 30 秒做服务端保活，连续 3 次无响应后清理失效会话，
避免旧连接一直占住 `18020` 端口、阻止新的反向隧道建立。

## 状态和停止

```powershell
Get-ScheduledTask -TaskName Echo-Intranet-Web
Get-Content tmp\intranet-web\supervisor.log -Tail 20
Get-Content tmp\intranet-web\web.log -Tail 20
Get-Content tmp\intranet-web\tunnel.log -Tail 20
```

正常停止（等待约 5 秒，监督进程会同时结束网页与隧道进程）：

```powershell
New-Item tmp\intranet-web\stop -ItemType File -Force
```

重新启动：

```powershell
Start-ScheduledTask -TaskName Echo-Intranet-Web
```

禁止下次登录自动启动：

```powershell
Disable-ScheduledTask -TaskName Echo-Intranet-Web
```

运行日志和进程记录位于已被 Git 忽略的 `tmp/intranet-web/`。

## 云端配置与回退

当前 Nginx 配置文件：`/etc/nginx/sites-available/echo`。
本目录的 `nginx.conf` 保存了本次实际部署的配置，包含现有域名和证书路径。
其中 HTTP 和 HTTPS 的 `location /` 均应使用：

```nginx
proxy_pass http://127.0.0.1:18020;
```

保留现有域名、证书、WebSocket 转发、上传限制及 900 秒超时。
配置中单独允许下载 `uploads/mainline/` 下由程序生成的音频和封面，
以支持上传后的试听；其他存储下载和研究管理接口仍被拦截。
修改后执行 `nginx -t`，通过后再 `systemctl reload nginx`。
不要覆盖同一服务器上的其他网站配置。

2026-10-30 部署前的配置备份位于：
`/root/echo-deploy-backups/20261030T094830Z/nginx-echo.conf`。
这个备份指向旧的 Tailscale 内网地址；回退配置前要先恢复旧上游。
服务器原来的 `/srv/echo`、`echo.service` 和数据未删除。
当前公开网页来自本机工作区；原 GitHub Actions 流程更新的是云端
`/srv/echo`，不会更新这台本机上的代码。修改 Python 后需要重新启动本机任务。

检查：

```bash
curl -f https://resonance.website/api/health
curl -f http://127.0.0.1:18020/api/health  # 在云服务器执行
```

如果公网出现 502，先检查本机是否已登录、监督任务是否运行，以及隧道日志。
如果隧道端口正常但网页不响应，检查 `web.log`。

## 本次验收（2026-10-30）

- HTTPS 首页、音乐页、收藏页、设置页及健康接口均返回 200。
- 曲库读取正常；音乐 Range 请求返回 206；匿名收藏初始化为 20 首。
- GPU 种子歌曲推荐成功返回 3 首推荐。
- 上传 5 秒 WAV，完成压缩、CultureMERT 嵌入及推荐，返回 3 首推荐。
- 上传后的试听请求返回 206；模型下载、路径穿越及 OpenAPI 请求返回 404。
- 正常停止监督进程后，确认网页子进程及监听端口已关闭，再重新启动任务。
- 手动结束 SSH 子进程，确认监督进程自动重启隧道，公网健康接口恢复 200。
- `nginx -t` 通过；Windows 任务 `Echo-Intranet-Web` 处于运行状态。

参考：[OpenSSH 反向转发说明](https://man.openbsd.org/ssh#R)。

2026-11-01 的位置覆盖、网页复测与连接恢复结果见
[完整自检报告](SELF_CHECK_2026-11-01_CN.md)。
