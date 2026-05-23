# DCAS Code Index

## 中文说明

这个目录包含论文系统主线的核心研究实现。可以把它理解为：`configs/` 决定“跑什么”，而 `dcas/` 负责“怎么跑”。

### 重点区域

- `models/`：模型定义与损失函数
- `pal/`：PAL 工具、uncertainty 逻辑、约束处理
- `ot/`：最优传输相关实现
- `data/`、`embeddings/`：数据加载和 embedding 工具
- `scripts/`：构建、训练、benchmark、PAL 的命令入口
- `cli/`：命令辅助与工作流包装
- `pipelines.py`：脚本和 API 会调用的高层 pipeline 函数

### 主线角色

如果说 `configs/` 是执行控制层，那 `dcas/` 就是对应的核心实现层。

## English Notes

This directory contains the core research implementation for the paper-system mainline.

### Important areas

- `models/`: model definitions and losses
- `pal/`: PAL utilities, uncertainty logic, and constraint handling
- `ot/`: optimal transport related logic
- `data/`, `embeddings/`: data loading and embedding utilities
- `scripts/`: command-line entry scripts for build, train, benchmark, and PAL workflows
- `cli/`: command helpers and workflow wrappers
- `pipelines.py`: high-level pipeline functions used by scripts and the API

### Mainline role

If `configs/` is the executable control plane, `dcas/` is the implementation layer behind it.
