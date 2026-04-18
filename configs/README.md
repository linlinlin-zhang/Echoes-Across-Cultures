# Configs Index

## 中文说明

这个目录是项目的可执行入口层。只要你想知道“现在该跑哪个配置”，通常都应该先看这里。

### 核心子目录

- `dataset/`：数据集 manifest 和构建输入
- `train/`：训练配置
- `benchmark/`：推荐评测与评估配置
- `pal/`：PAL 准备、回灌和对齐训练配置

### 主线规则

对于当前论文系统主线，默认优先使用 `V4 main` 相关配置，除非文档明确说明使用别的支线。

## English Notes

This directory is the executable entry layer of the project.

### Core subdirectories

- `dataset/`: dataset manifests and build inputs
- `train/`: model training runs
- `benchmark/`: recommender and evaluation runs
- `pal/`: PAL preparation and PAL-aligned retraining runs

### Mainline rule

For the current paper-system mainline, default to the `V4 main` configs unless a document explicitly says otherwise.
