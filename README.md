# NeuroLM Reproduction

复现 ICLR 2025 论文 [NeuroLM: A Universal Multi-task Foundation Model for Bridging the Gap between Language and EEG Signals](https://arxiv.org/pdf/2409.00101) 的实验结果。本仓库记录实验配置、数据来源、运行日志和与论文的对照；作者实现位于 [935963004/NeuroLM](https://github.com/935963004/NeuroLM)。

> 当前状态：作者环境、权重校验以及 Workload/HMC 预处理和最小运行检查已完成；TUAB/TUEV/TUSL 原始副本正在逐文件校验，SEED 尚缺作者脚本所需输入。尚无可与论文比较的复现指标，论文数字仅作目标值。

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
| 数据与权重准备 | 部分完成 | Workload/HMC 已准备；TUH 原始副本校验中；SEED 缺输入，见[数据清单](docs/DATASETS.md) |
| 训练或推理 | 仅最小检查 | B 起始权重的加载、两条 Workload 推理及单步训练探针已运行；均非论文指标 |
| 六项任务指标 | 未产生 | 尚无可与论文比较的结果 |

所有结果都将附带数据划分、配置、随机种子、源码提交、运行命令和原始日志；缺少证据时保持“未复现”状态。
