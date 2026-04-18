# Storage Index

## 中文说明

这个目录存放项目使用的主要数据和模型产物。

### 重要子目录

- `public/`：版本化研究数据集，包括当前主线 `research_dataset_v4/main`
- `models/`：训练后的 checkpoint，以及 PAL 派生产物
- `pal/`：PAL bundle、标注资源与相关中间产物
- `prototype/`：原型系统使用的文件
- `datasets/`、`uploads/`、`style/`、`ontology/`：运行时辅助或特定功能存储

### 主线规则

做论文主线工作时，默认重点关注 `public/research_dataset_v4/` 和 `models/`。

## English Notes

This directory stores the main data and model artifacts used by the project.

### Important subdirectories

- `public/`: versioned research datasets, including the current `research_dataset_v4/main`
- `models/`: trained checkpoints and PAL-derived checkpoints
- `pal/`: PAL bundles, annotation assets, and related intermediate materials
- `prototype/`: files used by the prototype system
- `datasets/`, `uploads/`, `style/`, `ontology/`: auxiliary runtime or feature-specific storage

### Mainline rule

For paper work, treat `public/research_dataset_v4/` and `models/` as the primary storage targets.
