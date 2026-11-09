# NeuroLM Reproduction

复现 ICLR 2025 论文 [NeuroLM: A Universal Multi-task Foundation Model for Bridging the Gap between Language and EEG Signals](https://arxiv.org/pdf/2409.00101) 的实验结果。本仓库记录实验配置、数据来源、运行日志和与论文的对照；作者实现位于 [935963004/NeuroLM](https://github.com/935963004/NeuroLM)。

> 项目状态（2026-11-09）：**本次五数据集复现已完成并归档**，见[结项报告](docs/REPRODUCTION_COMPLETED_20261109.md)。已完成自训 VQ、20 epoch 阶段二预训练、1 epoch 阶段三指令微调及官方初始化基线对照；402,480 条 val/test 预测通过独立审计，非法答案为 0。实测五项测试 BA 均低于官方基线。论文全语料、六任务和多 seed 配置保留为本次范围外的扩展。

## 目标与进度

- 本次五数据集端到端实验已完成；权重路径、结果、曲线和运行入口统一收录于[结项报告](docs/REPRODUCTION_COMPLETED_20261109.md)。
- [论文指标对照](docs/PAPER_TARGETS.md)
- [固定实验定义](docs/EXPERIMENT_DEFINITION.md)与[机器可读配置](experiments/neurolm_b_six_tasks.yaml)
- [论文与作者代码的逐项实验细节](docs/EXPERIMENT_DETAILS.md)
- [数据清单](docs/DATASETS.md)与[作者权重](docs/WEIGHTS.md)
- [作者代码审计](docs/CODE_AUDIT.md)
- [项目进度](docs/STATUS.md)
- [复现步骤与环境记录](docs/REPRODUCIBILITY.md)
- [范围与协作](docs/PROJECT_CHARTER.md)
- [决策记录](docs/DECISIONS.md)

## 获取作者代码

```bash
git clone https://github.com/935963004/NeuroLM.git upstream
git -C upstream checkout 0cda9876d8ce6ee07ed0c43eee5e9a6f5c24b177
```

作者代码和模型权重的使用条件以各自来源为准。本仓库不会提交 EEG 原始数据、模型权重、凭据或仅在本地使用的服务器配置。

## 复现结果

| 阶段 | 状态 | 证据 |
| --- | --- | --- |
| 源码获取 | 已完成 | 固定作者提交 `0cda9876d8ce` |
| 环境建立 | 已完成 | 数据盘上的独立环境及版本快照见[复现记录](docs/REPRODUCIBILITY.md) |
| 数据与权重准备 | 本次范围已完成 | 五项 EEG 数据与 OpenWebText 已预处理并保存清单，见[数据清单](docs/DATASETS.md) |
| 五任务指令微调 | 已完成 | 完整 1 epoch、898 次参数更新；全部验证/测试及 checkpoint SHA-256 通过[结果审计](results/INSTRUCTION_FIVE_TASK_20261108.md) |
| 我们的阶段二预训练 | 已完成 | 20 epoch、16 H20，见[最终权重与训练曲线](docs/pretrain_stage2_log.md) |
| 新模型阶段三匹配对照 | 已完成 | 新旧模型各 201,240 条预测通过审计；五项测试 BA 均未提升，见[完整对照](docs/stage3_new_pretrain_comparison.md) |
| 论文全配置扩展 | 本次范围外 | SEED、论文全量预训练语料及多 seed 设置未纳入此次结项 |

本次交付物已保留数据划分、配置、随机种子、源码哈希、运行命令和日志摘要；历史检查与最终结果分别归档。
