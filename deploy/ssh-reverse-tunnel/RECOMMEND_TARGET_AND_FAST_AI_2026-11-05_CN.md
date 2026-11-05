# 2026-11-05 恢复推荐方向选择，固定快速讲解

线上入口：https://resonance.website/music.html 。

## 原因与恢复来源

推荐目标弹窗在远程开发分支的 `6c07731`（Add recommendation target picker）中加入，展览发布仍以更早的 `0fb2746` 为基线，未合入这段功能。上次 `22ce89d` 只恢复曲库选曲，没有回迁推荐方向弹窗。

本次从同一远程分支的 `de5c9e4:web/music.html` 提取原有推荐弹窗 HTML、CSS、文化域/地域映射及提交逻辑，保留原版两级选择、随机生成、开始生成、关闭、Escape 关闭和手机底部弹层。补齐七种界面语言，并修复窄屏下原生下拉选项撑宽弹窗的问题。

点击“跨文化音乐推荐”或“重新生成推荐”时，先显示方向选择；确认之后才请求推荐。明确文化域直接传入 `target_culture`；仅选地域时从该地域支持的文化域中选择，随机探索会尽量避开种子歌曲文化域。曲库和上传音频两种推荐路径都带上所选目标。现有算法将此参数用于目标文化分布与亲和度，不把“开放推荐”改成严格限定全部曲目来自该地区。

后端已经支持目标参数，本次没有改动 Python 服务、推荐算法或模型，也没有重启后台或隧道。地理补充数据与中英歌曲描述字典保持原样。

## 分析讲解

原来默认使用快速模式。本次删除“快速／思考”切换按钮及其事件和专属样式，讲解与追问的请求固定为 `thinking_mode: fast`。仍可查看历史保存的讲解内容；没有更换模型或调用付费 AI 进行测试。

## 验证与发布

- `scripts/check_recommendation_targets.cjs`：57 项通过，覆盖七语言、四种屏幕宽度、取消无计算、地域与文化域联动、明确目标、地域内随机、全局随机、过载后恢复、上传 multipart 目标参数，以及讲解和追问始终快速。API 响应均为本地模拟。
- `scripts/check_web_i18n.cjs`：七语言四页面通过，覆盖首屏标签、切换、描述和地点显示。
- `scripts/check_live_web.cjs` 与 `scripts/check_music_catalog_layout.cjs` 已适配原目标确认流程，线上验证请求参数与推荐结果中的 `effective_target_culture` 一致。
- `scripts/check_ai_interface.cjs` 已更新为验证无切换选项、快速流式讲解、追问、内容净化和失败恢复，服务响应由浏览器模拟，避免付费调用。

公开打包器生成资源内容哈希与 gzip；经公开文件允许清单、SHA-256、gzip 内容和既有英文描述字典哈希验证后，复制前一发布目录并原子切换。完整公开包 SHA-256：`cdc8b3b955d3ca08122d40274a2d522131aaeea3ba9f8946d505eb8eee769a41`。

本机原始结果和截图保存在 `tmp/recommendation-targets-20261105/`，仓库继续推送到 `codex/exhibition-reliability-performance`。

线上复测完成：20 项曲库/布局检查通过；24 项完整网页流程通过，其中确认选择巴西后，请求 `target_culture` 与引擎返回的 `effective_target_culture` 均为 `brazil`。快速讲解界面 4 项模拟服务检查通过，所有讲解和追问请求均为 `fast`，未调用付费 AI。当前云端发布：`/srv/echo-exhibition/releases/20261105T143953Z-recommend-target`，上一版保持可回退。
