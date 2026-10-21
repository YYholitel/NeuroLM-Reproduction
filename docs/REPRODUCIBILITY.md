# 复现记录

## 固定来源

| 项目 | 记录 |
| --- | --- |
| 论文 | [arXiv:2409.00101v3](https://arxiv.org/pdf/2409.00101)；ICLR 2025 |
| 作者源码 | [935963004/NeuroLM](https://github.com/935963004/NeuroLM) |
| 作者源码提交 | `0cda9876d8ce6ee07ed0c43eee5e9a6f5c24b177` |
| 本项目仓库 | [YYholitel/NeuroLM-Reproduction](https://github.com/YYholitel/NeuroLM-Reproduction) |

## 环境差异

| 项目 | 论文或作者说明 | 本次实测（2026-10-15） |
| --- | --- | --- |
| GPU | 论文：8 × A100 80 GB | 8 × RTX 3090 24 GB |
| Python | 论文：3.11.8；作者 README：3.12 | 数据盘独立环境：3.11.8 |
| PyTorch / CUDA | 论文：2.2.2 / 12.1；作者 README：2.5.0 / 12.4 | 2.2.2 / 12.1；CUDA 已识别 |
| 包管理 | 作者 README 使用 conda | 使用数据盘上的 conda 建立独立环境；包缓存与临时文件在数据盘 |
| 存储 | 论文未说明 | 系统盘空间紧张；数据盘有可用空间，具体工作目录保留在本地运维记录 |

论文及作者 README 的环境版本不同。本次采用论文的 Python/PyTorch/CUDA 版本。`conda` 默认解出的 MKL 2025 使 PyTorch 2.2.2 导入报 `iJIT_NotifyEvent`；按 [PyTorch issue](https://github.com/pytorch/pytorch/issues/123097)固定 MKL 2023.1.0（低于有问题的 2024.1）及 NumPy 1.26.4 后，PyTorch、CUDA、MNE 和其余主要依赖导入成功，`pip check` 无冲突。作者 README 中的 torchvision/torchaudio 最初随 conda 依赖进入环境，但实验入口无需调用它们。

直接依赖版本见 [`env/requirements.txt`](../env/requirements.txt)，当前 Linux conda 包的显式列表见 [`env/conda-linux-64.lock`](../env/conda-linux-64.lock)，已安装 Python 包的版本快照见 [`env/pip-versions.lock`](../env/pip-versions.lock)。后两者是本次环境快照；在其他主机重建时仍须检查平台和包源可用性。

## 最小运行记录（2026-10-15）

1. 从固定修订下载并校验 B 与 VQ 权重。B 权重以 `scripts/smoke_model_load.py` 在加载前再次核对 SHA-256，然后对模型权重执行 `strict=True` 加载。
2. 从 [PhysioNet EEGMAT v1.0.0](https://physionet.org/content/eegmat/1.0.0/) 下载 72 份 EDF，并用来源的 `SHA256SUMS.txt` 校验。作者 Workload 预处理仅应用 [`0001-workload-paths.patch`](../patches/0001-workload-paths.patch) 参数化路径；实际输出 1508/290/290 个样本，划分摘要见[清单](../experiments/workload_split_manifest.json)。
3. `scripts/smoke_model_load.py` 使用 B 起始权重、PyTorch 2.2.2 和一张 RTX 3090，完成文本前向（输入 `[1,8]`，logits `[1,1,58496]`），再对测试集两条真实 EEG 样本贪心生成 5 个 token 并调用作者 `get_pred()`。`Subject31_1-0.pkl` 的标签为 0、解析预测为 `-1`；`Subject31_2-0.pkl` 的标签为 1、解析预测也为 `-1`。命令退出码 0，标准错误为空。运行输出保存在服务器数据盘；公开仓库只记录不含 EEG 内容的摘要。

这两条样本只验证了数据加载、模型生成及答案解析链路。`-1` 是作者解析器未识别生成答案的返回值，既不算正确预测，也不构成 Workload 测试集指标。B 权重仍需六任务联合指令微调，才能与论文表 2–4 对照。

2026-10-21 另对 TUAB、TUEV、TUSL 预处理输出分别用作者对应的 `downstream_dataset.py` 指令加载器读取一个真实样本。TUAB/TUEV 返回 7 个张量；EEG 张量为 `276×200`，文本输入与目标各为 80 token，说明输出键和训练入口的样本接口衔接。TUSL 的单文件与全量输出也通过加载检查。这些检查不产生模型预测或论文指标。

## 2026-10-21 数据准备入口

TUH 三套原始副本使用 `rsync -rtcn` 与共享来源逐文件比较通过；患者/文件划分的计数与哈希见[数据清单](DATASETS.md)。TUAB 需依次应用 [`0006`](../patches/0006-tuh-preprocess.patch) 和 [`0008`](../patches/0008-tuab-patient-split.patch)；TUSL 需应用 `0006` 和 [`0007`](../patches/0007-tusl-output-stem.patch)。这些补丁只用于固定作者提交的独立源码副本。

文本准备依次应用 [`0005`](../patches/0005-text-output-revision.patch) 和 [`0009`](../patches/0009-text-local-parquet.patch)。固定修订的 80 份 Parquet 在数据盘按原编号排序，元数据合计 8,013,769 行；其中经本机传输的 16 份已在服务器逐份通过 SHA-256 对照。运行命令模板如下，`DATA_ROOT` 与 `RUN_ROOT` 均应位于有充足空间的数据盘：

```bash
python text_dataset_maker/prepare.py \
  --local_parquet_dir "$DATA_ROOT/raw/openwebtext-parquet" \
  --out_dir "$DATA_ROOT/text" \
  --revision 79d93d786212f7344586290adb811d4ae6a1762c \
  --download_workers 1
ln -s "$DATA_ROOT/text" "$RUN_ROOT/text"
```

作者指令入口从 `--out_dir/text/` 读取 `train.bin` 和 `val.bin`，因此运行目录中的 `text` 链接指向数据目录；训练输出与原始数据仍分别存放。文本二进制文件生成完毕后使用 [`verify_text_bins.py`](../scripts/verify_text_bins.py) 记录大小、token 数和 SHA-256。

## 单步训练与显存探针（2026-10-15）

[`scripts/probe_train_step.py`](../scripts/probe_train_step.py)以已校验的 B 权重初始化模型，取一条真实的 Workload 训练样本，配一批确定性的**合成文本**，按作者训练入口计算 EEG 指令损失与文本损失之和，完成一次反向传播、梯度裁剪及 AdamW 更新。探针不写权重，不使用 OpenWebText，重复样本仅用于测显存，不构成论文训练或结果。

在一张 RTX 3090 上，以 BF16、文本长度 1024 测试了以下每卡批量。表中的显存来自 `torch.cuda.max_memory_allocated/reserved`；时间仅覆盖前向至优化器更新，不能代表训练吞吐。每次运行均核验 B 权重 SHA-256，标准输出与错误日志保存在服务器数据盘。

| EEG batch | 文本 batch | 单步结果 | 峰值已分配 / 保留显存 | 备注 |
| ---: | ---: | --- | --- | --- |
| 1 | 1 | 成功，退出码 0 | 2.988 / 3.322 GiB | 5.889 秒 |
| 16 | 4 | 成功，退出码 0 | 8.906 / 11.059 GiB | 3.103 秒 |
| 32 | 8 | 成功，退出码 0 | 16.554 / 20.562 GiB | 1.021 秒 |
| 64 | 16 | CUDA 显存不足，退出码 1 | 报错时 PyTorch 已分配 23.03 GiB | EEG 前向的 LayerNorm 继续申请 48 MiB 时失败 |

首次 16/4 测试因探针构造的文本目标张量不连续而在作者模型的 `view()` 处失败；探针现对文本切片调用 `.contiguous()`，随后 16/4 完成整个训练步。64/16 是作者训练入口的默认每卡批量，在本机 24 GB 显卡上无法直接运行。32/8 可作为后续完整数据训练的起始显存配置；若用两次梯度累积达到相同名义总批量，须在实验配置中明确记录，不能据此声称与作者原配置完全相同。报告的损失只用于确认计算图可运行，不记录为实验指标。八卡配置、完整数据及实际训练吞吐仍需独立测量。

为避免模型生成无法解析时只看到聚合指标，另准备了 [`0004-parse-failure-count.patch`](../patches/0004-parse-failure-count.patch)。它在作者评估结果中增加 `parse_failures` 计数，保持原有指标输入及公式不变；正式运行仍需保存原始生成文本以便抽查。

## 尚需补齐的证据

- 六个下游数据集的下载位置、授权、版本、校验和划分文件。
- VQ、预训练及指令微调权重的使用阶段与实际加载方式；已核对文件与哈希见[权重清单](WEIGHTS.md)。
- 每阶段的命令、配置、随机种子、GPU 占用、耗时及退出状态。
- 指标计算代码、样本数和论文表 2–4 的对照。
- 干净环境复核记录。

## 结果判定

在数据、配置与划分一致并运行独立实验之前，所有任务标记为“未复现”。若显存约束迫使修改批量或模型，记录差异，单独报告近似实验，避免与论文数字混淆。
