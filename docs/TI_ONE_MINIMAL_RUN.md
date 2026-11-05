# TI-ONE 五任务最小完整运行

本轮执行目标是 TUAB、TUEV、HMC、Workload、TUSL 联合指令微调，种子 42，完整训练 1 epoch，随后评估全部验证集、测试集并保存 checkpoint。SEED 按用户要求暂缓。本配置属于五任务修改协议实验，不能填入论文六任务复现结果表。

当前状态：**准备完成代码包，尚未提交训练任务、尚无训练指标**。腾讯 SSH 登录已验证；该 SSH 会话没有 TI-ONE Notebook STS 临时凭证，技能提供的客户端鉴权失败。SDK 参数构建 dry-run 已通过，但它不证明平台权限或 GPU 可用性。完整数据和环境迁移也尚未完成。

## 配置与执行

配置见 [`neurolm_b_five_task_minimal.yaml`](../experiments/neurolm_b_five_task_minimal.yaml)。计划申请一个 8×H20 节点，每卡 EEG/text batch 为 64/16，梯度累积 1，名义全局 batch 512/128；学习率 5e-4、最低 5e-5、warmup ratio 0.1、weight decay 0.1、裁剪 1。459,656 条训练样本在作者 `drop_last=True` 策略下产生 897 次更新，每 epoch 消费 459,264 条样本。

保留 R2 的 Python 3.11.8、PyTorch 2.2.2/CUDA 12.1、NumPy 1.26.4。基础容器镜像中自带的新版 torch 不作为训练运行时。现有环境已打成迁移包；打包时发现 conda 记录中的 `tbb-devel` 开发头文件、`llvm-openmp` 静态库缺失，因此使用 `conda-pack --ignore-missing-files` 打包当前实际文件。该操作未修改训练环境；目标机导入和版本检查尚待迁移后执行，打包成功不代表目标运行时验证通过。

[`run_five_task_minimal.sh`](../scripts/run_five_task_minimal.sh)需要私有环境变量 `NEUROLM_ROOT`、唯一的 `NEUROLM_RUN_ID`。它先核验运行时、GPU 数、15 个数据划分计数、两份权重和文本 bin 的 SHA-256；输入不齐时退出。预训练权重、数据、私有平台配置和输出均不进入 Git。

## 相对作者入口的补丁

在固定提交 `0cda9876d8ce6ee07ed0c43eee5e9a6f5c24b177` 上顺序应用 `0002`、`0004`、[`0010`](../patches/0010-instruction-task-selection-and-run-records.patch)。已有数据预处理补丁及 R2、R4–R8 的结论继续有效。

- 增加 `--datasets`、`--num_workers`；默认任务仍为作者六任务，运行时显式选择五任务。
- 对已划分文件排序；固定 Python/NumPy 随机种子，分布式训练 sampler 每 epoch 设置种子。划分成员不变，训练顺序相对作者无排序实现可能不同。
- 从 CPU 读取已校验的可信 checkpoint，显式 `weights_only=False`；恢复时用 scratch 构建后加载全部权重，避免额外下载 GPT-2。
- 调度步数取实际 DataLoader 长度，累积的末尾不足一组时正常归一化、同步和更新；本轮累积为 1。
- 验证/测试按 rank 分片且不补重复样本，汇总全部预测后调用原指标函数，保存每个 rank 的逐样本生成文本、预测和标签。
- 保存运行配置、训练步日志、每任务指标和完成标记；非有限 loss/梯度直接失败。

作者的答案标签、归一化、EEG/text 两项损失之和、生成参数和解析器保持不变。原评估代码对二分类使用离散预测计算 AUC，多分类解析失败 `-1` 会索引到最后一类；这里保留该行为并显式统计 `parse_failures`、保存原始生成文本，不能把解析失败隐藏在聚合指标里。

## 验证与完成证据

已完成：补丁语法检查、五任务 CLI 参数解析、两进程 CPU/Gloo 检查（不均匀分片、空分片、预测索引无遗漏/重复，以及两次前向后联合反传）、服务器 Bash 语法检查、实际 tikit SDK 参数构建、代码包上传 SHA-256 一致性。CPU 检查没有使用真实模型或进行 EEG 训练，不能代替 H20/NCCL 运行验证。

可重跑分布式回归检查：

```bash
python scripts/check_instruction_runtime_patch.py --author-file /path/to/patched/train_instruction.py
```

[`summarize_instruction_run.py`](../scripts/summarize_instruction_run.py)要求五任务共十个验证/测试结果、完整逐样本索引、解析失败计数一致以及 checkpoint 存在，并生成不含内部路径的 `summary.json` 和 checkpoint SHA-256。完整成功还要求 worker 退出码 0；在这些证据齐全之前，不报告训练已完成或论文指标。

平台鉴权脚本、任务 JSON、服务器路径与迁移进度保存在忽略的 `.local/` 和用户私有 CFS 目录。无需把密钥、Token 或内部地址加入公开项目。
