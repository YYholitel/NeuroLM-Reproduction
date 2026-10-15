# 论文目标值与复现结果

来源：[NeuroLM 论文](https://arxiv.org/pdf/2409.00101) 第 3.3 节表 2–4。下表仅列 NeuroLM-B 的论文均值和标准差；复现列在真实实验完成前保持空白。论文对六项任务报告三次随机种子结果。

| 数据集 | 指标 | 论文 NeuroLM-B | 本项目复现 | 状态 |
| --- | --- | ---: | ---: | --- |
| TUAB | Balanced Accuracy | 0.7826 ± 0.0065 | — | 未运行 |
| TUAB | AUC-PR | 0.6975 ± 0.0081 | — | 未运行 |
| TUAB | AUROC | 0.7816 ± 0.0079 | — | 未运行 |
| TUEV | Balanced Accuracy | 0.4560 ± 0.0048 | — | 未运行 |
| TUEV | Cohen's Kappa | 0.4285 ± 0.0048 | — | 未运行 |
| TUEV | Weighted F1 | 0.7153 ± 0.0028 | — | 未运行 |
| SEED | Balanced Accuracy | 0.5554 ± 0.0075 | — | 未运行 |
| SEED | Cohen's Kappa | 0.3393 ± 0.0117 | — | 未运行 |
| SEED | Weighted F1 | 0.5599 ± 0.0068 | — | 未运行 |
| HMC | Balanced Accuracy | 0.6737 ± 0.0050 | — | 未运行 |
| HMC | Cohen's Kappa | 0.6188 ± 0.0057 | — | 未运行 |
| HMC | Weighted F1 | 0.7126 ± 0.0034 | — | 未运行 |
| Workload | Balanced Accuracy | 0.6172 ± 0.0113 | — | 未运行 |
| Workload | AUC-PR | 0.5824 ± 0.0080 | — | 未运行 |
| Workload | AUROC | 0.6253 ± 0.0160 | — | 未运行 |
| TUSL | Balanced Accuracy | 0.6734 ± 0.0436 | — | 未运行 |
| TUSL | Cohen's Kappa | 0.5107 ± 0.0617 | — | 未运行 |
| TUSL | Weighted F1 | 0.6743 ± 0.0394 | — | 未运行 |

## 核对要求

- 按论文的数据划分复现；划分细节见第 3.1 节。
- 记录每个随机种子的独立指标、均值、标准差、样本数和配置。
- 若修改数据、模型或训练设置，先列出差异，再计算与论文的差值。
