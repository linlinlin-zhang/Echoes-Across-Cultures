# 自检结束文档：第2点能力可靠性与投稿级可用性加固

时间：2026-02-27
负责人：Codex

## 自检结论
- 三项能力（CultureMERT / 风格迁移 / 动态本体）已达到“工程可运行”标准。
- 但仍未达到“可直接支撑 ISMIR 顶会投稿结论”的证据标准（主要缺公开基准实验与严谨评测）。

## 已完成加固

1) 风格迁移诊断修复（关键 bug）
- 修复前：`zc_drift/za_drift` 结构性恒为 0（比较对象错误）
- 修复后：对生成 embedding 回编码后再计算漂移与保持度
- 新增 meta 指标：
  - `style_alignment`
  - `content_preservation`
  - `affect_preservation`

2) CultureMERT 建库可追溯增强
- 新增 pooling 参数校验（`mean/cls`）
- metadata 必填列校验与空数据校验
- track_id 重复拦截
- 生成 `tracks.npz.manifest.json` 记录模型、参数、错误与维度

3) 动态本体可训练闭环增强
- 新增本体导出 pairwise constraints：
  - `OntologyStore.export_pairwise_constraints`
  - `OntologyStore.save_pairwise_constraints`
- 新增 CLI：`ontology export-constraints`
- 新增 API：`POST /api/ontology/export_constraints`

## 验证记录
- `style_transfer` CLI 冒烟通过，meta 指标不再恒零
- `build_tracks_from_audio --help` 显示新增 pooling 参数
- 本体导出约束成功产出 jsonl
- `/api/ontology/export_constraints` 路由存在
- `python -m compileall dcas dcas_server` 通过

## 现阶段可主张边界（更新）
可主张：
- 已具备真实音频 embedding 构建、embedding 级反事实迁移、动态本体到训练约束导出的可运行闭环。

不可主张：
- 已完成公开大基准上的全面实验并达到 SOTA。
- 已完成波形级风格生成和听测验证。

## 下一步建议（进入 ISMIR 证据层）
1. 建立公开基准实验脚本（GlobalMood / CultureMERT Bench）
2. 增加解纠缠指标（MIG/DCI/SAP）与统计显著性检验
3. 输出可复现实验配置与表格模板（主表 + ablation + fairness）
