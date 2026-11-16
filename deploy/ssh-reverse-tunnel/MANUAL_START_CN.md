# Echo 按需启动（2026-11-16）

本机 `Echo-Intranet-Web` 计划任务已禁用；登录 Windows 不再自动运行 Echo 网页应用和 SSH 内网穿透。旧任务配置备份在本机 `tmp/intranet-web/manual-migration-20261116/previous-task.xml`，不含私钥正文。

## 使用

1. 双击项目根目录的 **启动网页应用.cmd**。脚本使用现有 `.venv-gpu`、`configs/local_web.env` 和用户目录下的 `.ssh/echo_exhibition_ed25519`，无需再次配置密钥，也不注册／启用计划任务。
2. 保留出现的维护窗口。启动应用并建立公网连接后会自动打开 `https://resonance.website/music.html`；模型首次预热可能稍慢。推荐就绪状态可从 `/api/health` 的 `recommendation_state=ready` 核查。
3. 用完双击 **停止网页应用.cmd**，或者关闭维护窗口。网页进程、模型和两条穿透隧道会一起结束；下次仍需手动启动。停止脚本可能等待数十秒完成清理。

重复点击启动不会重复创建应用或占用端口。启动时自动维护进程，异常退出自动重启；主／备用隧道断线会重连。关闭维护窗口时 Windows Job Object 会清理它创建的整组子进程，避免旧式窗口退出后仍残留 Python／SSH。运行期间继续防止系统自动睡眠，退出时释放。

## 日志与排查

日志位于 `tmp/intranet-web/`：`supervisor.log`、`web.log`、`tunnel.log`、`tunnel-backup.log`。缺失 Python 环境、密钥或配置会在窗口提示，脚本失败时保留错误信息。后台状态 JSON 是最后一轮探测记录，停止后不代表仍在线；使用下列命令查询当前维护进程是否运行：

```powershell
.\.venv-gpu\Scripts\python.exe scripts\manual_intranet_web.py status
Get-ScheduledTask -TaskName Echo-Intranet-Web | Select-Object TaskName, State
```

无需自动打开浏览器时，可运行 `启动网页应用.cmd --no-browser`。手动脚本不需要管理员权限；旧 `install_intranet_web_task.ps1` 用于主动安装登录启动任务，按需使用时不要运行该安装器。

## 部署范围

云服务器 `47.250.196.219` 保留现有 Nginx、静态网页、只读曲库及试听服务。停止本机后，实时推荐、上传分析、收藏与 AI 等依赖中国主机的功能不可用；再次手动启动并连通隧道后恢复。脚本不关闭云服务器，也不删除网站文件、模型、曲库或配置。

## 本机验证

2026-11-16 实际完成 7 项流程检查：禁用登录启动、双击入口启动、本机模型就绪和公网 HTTPS 连通、重复启动不多开、主动结束主隧道后自动重建、停止脚本退出并释放端口、强制关闭维护进程后由 Job Object 清理所有子进程。最终状态为 **停止**，计划任务为 **Disabled**，本机 8000 端口无监听，未残留 Echo 网页／隧道维护进程。

详细本机记录位于 `tmp/manual-start-check-20261116/report.json`，启动／停止及关闭窗口的输出位于同目录日志；这些运行记录不提交仓库。新脚本通过 Python 编译及 Ruff 检查。Windows 字节锁通过不读取已锁定字节的方式检查运行状态，重复启动和停止都在真实 Windows 环境验证。
