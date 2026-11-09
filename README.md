# NeuroLM Reproduction

复现 ICLR 2025 论文 [NeuroLM: A Universal Multi-task Foundation Model for Bridging the Gap between Language and EEG Signals](https://arxiv.org/pdf/2409.00101) 的实验结果。本仓库记录实验配置、数据来源、运行日志和与论文的对照；作者实现位于 [935963004/NeuroLM](https://github.com/935963004/NeuroLM)。

> 当前状态（2026-11-09）：我们的 VQ + GPT-2 初始化模型已完成 [20 epoch 阶段二预训练](docs/pretrain_stage2_log.md)，正在执行[阶段三匹配对照](docs/stage3_new_pretrain_comparison.md)：五任务、1 epoch、seed 42，与既有官方初始化指令基线统一采用选项约束解码评估。当前尚无本轮最终测试成绩。早期[自由生成基线](results/INSTRUCTION_FIVE_TASK_20261108.md)的 TUSL 解析失败与[后续修正评估](docs/Downstream_eval_log.md)分别记录，不混合比较。论文六任务、5 epochs、三种子复现仍未完成。

## 目标与进度

- 先验证 NeuroLM-B 的完整运行链路，再按数据可用性复现六项下游任务与三次随机种子结果。
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
| 数据与权重准备 | 部分完成 | 五项 EEG 数据与 OpenWebText 已预处理并保存清单；SEED 缺输入，见[数据清单](docs/DATASETS.md) |
| 五任务指令微调 | 已完成 | 完整 1 epoch、898 次参数更新；全部验证/测试及 checkpoint SHA-256 通过[结果审计](results/INSTRUCTION_FIVE_TASK_20261108.md) |
| 我们的阶段二预训练 | 已完成 | 20 epoch、16 H20，见[最终权重与训练曲线](docs/pretrain_stage2_log.md) |
| 新模型阶段三匹配对照 | 运行中 | 相同五任务指令微调配置，随后对新旧模型各评估 201,240 个 val/test 窗口，见[运行记录](docs/stage3_new_pretrain_comparison.md) |
| 论文六任务结果 | 未产生 | 当前为不含 SEED 的五任务、单种子、1 epoch 修改协议 |

所有结果都将附带数据划分、配置、随机种子、源码提交、运行命令和原始日志；缺少证据时保持“未复现”状态。
