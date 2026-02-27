# 第2点-子项3开始文档：动态本体（Dynamic Ontology）

时间：2026-02-27
负责人：Codex

## 目标
新增可运行的动态本体层，允许专家在系统运行中持续创建概念、关系和标注，并将其持久化。

## 范围
1. 新增本体存储模块（JSON 持久化）
2. 提供概念/关系/标注的增查接口
3. 提供轻量建议接口（基于文本相似度）
4. 接入 FastAPI

## 数据结构
- Concept: `id,name,description,parent_id,aliases,created_at`
- Relation: `id,source_id,target_id,type,weight,created_at`
- Annotation: `id,track_id,concept_id,confidence,source,rationale,created_at`

## 验收标准
- 支持运行时新增概念并持久化
- 支持关联关系与曲目标注
- 支持查询与建议，且 API 路由可用
