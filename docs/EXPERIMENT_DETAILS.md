# NeuroLM-B 六任务复现：论文与固定作者代码逐项核查

核查日期：2026-10-15。依据为 [ICLR 2025 论文 arXiv v3](https://arxiv.org/pdf/2409.00101)、[作者 README](https://github.com/935963004/NeuroLM/blob/0cda9876d8ce6ee07ed0c43eee5e9a6f5c24b177/README.md) 和作者提交 `0cda9876d8ce6ee07ed0c43eee5e9a6f5c24b177`。本文件描述**论文报告**、**固定提交的实际代码**、**本项目选择**和**待证实项**，不把它们混成一套已经运行的实验。论文目标数值在 `results/paper_neurolm_b.csv`；本项目测试结果只可写入 `results/measured_neurolm_b.csv`。

## 1. 范围、模型和权重

| 项目 | 已核实内容 |
| --- | --- |
| 论文任务 | 六数据集联合指令微调，一个最终模型分别在六个测试集评估；不是六个独立模型。论文第 2.3、3.2 节。 |
| NeuroLM-B | GPT-2 基座；含 VQ encoder 共 254M 参数。论文第 3.2 节。入口先写入 12 层、12 头、768 隐层、block size 1024、dropout 0、无 bias 的默认值，再以 checkpoint 的 `model_args` 覆盖关键字段；已读取的 B 权重实际配置 `bias=true`。 |
| 起始权重 | `NeuroLM-B.pt`，固定 Hugging Face 修订 `eddfff5c64a4139442f826d6c67c8369fd00f45a`，SHA-256 `ffe098bc...d29ab`。这是预训练起点，论文表 2–4 是指令微调后的模型。 |
| VQ 权重 | `VQ.pt` 的已核对本机 SHA-256 为 `e792c39a...5a2663a`。固定代码的 instruction `pretrained` 分支从 `NeuroLM-B.pt` 加载**整个模型**，没有把单独的 VQ 路径传给构造函数；VQ.pt 用于前一预训练阶段。见 `train_instruction.py:241-307`、`model/model_neurolm.py:48-83`。 |
| 模型输出 | `generate()` 默认 `top_k=1`，每步限制到最大 logit，最多生成 5 个新 token，再解码并解析答案。见 `model/model_neurolm.py:139-165`、`train_instruction.py:492-515`。 |

模型经历文本对齐 VQ tokenizer、自回归预训练、联合指令微调三阶段。使用已发布 NeuroLM-B 起始权重的本项目方案**不重训前两阶段**；若要做从零的完整训练，是另一套资源和数据需求。论文预训练约 25,000 小时 EEG，8 × A100 80 GB；本项目服务器是 8 × RTX 3090 24 GB。论文环境为 Python 3.11.8 / PyTorch 2.2.2 / CUDA 12.1，作者 README 给出的安装组合是 Python 3.12 / PyTorch 2.5.0 / CUDA 12.4，二者不同。

## 2. 六个数据集：论文规格与代码划分

论文表 1 的样本数是论文报告值，不是本项目实际读取数。原始采样率也来自论文表 1；代码预处理后统一到 200 Hz。指标按论文表 2–4。

| 数据集 | 任务；论文原始规格与样本数 | 论文划分 | 固定代码里的具体动作 | 三项主指标 |
| --- | --- | --- | --- | --- |
| TUAB | 异常检测；23 通道，256 Hz，10 秒，409,455 段 | 保留原测试集；训练患者随机 80%/20% 切训练/验证 | `prepare_TUAB.py:92-120` 对正常和异常患者分别 `set()`、`np.random.shuffle()` 后 80/20 切分；未显式设种子。每段 10 秒不重叠（`44-80`）。`utils.py:56-68` 读 `TUAB/processed/{train,val,test}`。 | Balanced Accuracy、AUC-PR、AUROC |
| TUEV | 事件六分类；23 通道，256 Hz，5 秒，112,491 段 | 保留原测试集；训练患者随机 80%/20% | `prepare_TUEV.py:220-241` 写死种子 4523，从 `set()` 组成的患者列表抽 20% 验证；原 `processed_eval` 拷贝到测试。生成的样本为事件周围 5 秒（`23-45`）；训练入口读 `TUEV/{processed_train,processed_eval,processed_test}`。`set()` 未排序会影响同一数值种子的实际成员。 | Balanced Accuracy、Cohen's Kappa、Weighted F1 |
| SEED | 情绪三分类；62 通道，1000 Hz，4 秒，38,475 段 | 每个 session 的 15 个 trial 按时间 9/3/3，然后合并 session | `train_instruction.py:65-70` 使用 HDF5 的 trial 百分比 0–0.6/0.6–0.8/0.8–1；`downstream_dataset.py:110-123` 以整数截断得到 trial 边界，窗口 800 点、步长 800 点，即预处理后 4 秒不重叠。输入 `h5data/seed-3.hdf5`。 | Balanced Accuracy、Cohen's Kappa、Weighted F1 |
| HMC | 睡眠五分类；4 通道，256 Hz，30 秒，137,243 段 | 前 100 人训练、接下 25 人验证、后 26 人测试 | `prepare_HMC.py:108-117` 先排序 EDF 文件，再以 `[:100]`、`[100:125]`、`[125:]` 切分；必须核对一人一文件且文件名排序等于论文受试者顺序。30 秒标注片段（`17-33`）。 | Balanced Accuracy、Cohen's Kappa、Weighted F1 |
| Workload | 负荷二分类；19 通道，500 Hz，4 秒，2,088 段 | 受试者 0–25/26–30/31–35 分训练/验证/测试 | `prepare_workload.py:105-113` 对排序后的 EDF 文件切 `[:52]`、`[52:62]`、`[62:]`；与每人两文件相符仍须核对文件名。读每文件最后 60 秒，每 2 秒步长取 4 秒窗口共 29 段（`16-26,48-50`）；文件名倒数第 5 位为 1/2，映射低/高负荷（`55-61`）。 | Balanced Accuracy、AUC-PR、AUROC |
| TUSL | 慢化事件三分类；23 通道，256 Hz，10 秒，245 段 | 60%/20%/20%，论文未给成员列表 | `prepare_TUSL.py:112-120` 对 EDF **文件**按遍历顺序取前 60% 训练、接下 20% **测试**、最后 20% **验证**；这里没有排序或随机种子。只有冻结文件列表及输出清单才能重现。 | Balanced Accuracy、Cohen's Kappa、Weighted F1 |

论文未公开上述随机划分的成员清单。本项目选的训练种子 `42/43/44` 不会自动覆盖预处理脚本的固定种子或无种子行为。特别是 TUAB/TUEV 的 `set()` 顺序与 TUSL 的文件遍历顺序，都要求把**实际成员**保存为清单并计算哈希。

## 3. EEG 预处理与输入格式

论文第 3.2 节要求去无用通道、0.1–75 Hz 带通、按地点 50/60 Hz 陷波、重采样 200 Hz、以 μV 为单位，输入再除以 100。作者各数据脚本使用 MNE 读取原始 EDF/CNT，检查并调整通道顺序。TUAB、TUEV、SEED、HMC、Workload 用 50 Hz 陷波；TUSL 用 60 Hz。`downstream_dataset.py` 中六项加载器均执行 `X / 100`，并将每 200 点作为 1 秒 EEG patch；指令训练传入 `eeg_max_len=276`、`text_max_len=80`，不足时补零。具体原始格式、通道排除列表、标签及窗口逻辑须以六个 `dataset_maker/prepare_*.py` 和 `downstream_dataset.py` 为准。

SEED 的作者预处理脚本要求原始 `.cnt`，并把 trial 起止秒数写入 HDF5 属性；已有同名 `.mat` 文件不能直接代替。所有预处理脚本含作者机器上的绝对路径，迁移到本项目服务器需参数化并保留补丁；不应直接修改固定提交后仍称“原样运行”。

TUEV 脚本先在原始根目录生成 `processed_train`、`processed_eval`，再把划分结果复制到 `processed/processed_train`、`processed/processed_eval`、`processed/processed_test`（`prepare_TUEV.py:189-241`）；训练入口却直接寻找 `--dataset_dir/TUEV/{processed_train,processed_eval,processed_test}`（`utils.py:34-52`）。两处路径不能仅靠默认值自动衔接，运行前须明确最终 `--dataset_dir` 布局并核对三份文件清单。

## 4. 联合指令微调与文本数据

| 设置 | 论文附录 D 表 9 | 固定 `train_instruction.py` 默认值或行为 |
| --- | --- | --- |
| EEG / 文本 batch | 全局 512 / 128 | 每进程 64 / 16；8 进程、梯度累积 1 时名义全局 512 / 128。GPU 数或累积步数变化时需重新计算。 |
| 学习率 | B 峰值 5e-4，最小 5e-5，余弦衰减 | 相同；`--decay_lr` 默认启用，但该开关写成 `store_false`，传它反而关闭衰减。 |
| 优化器 | AdamW，β=(0.9,0.95)，weight decay 0.1 | `model.configure_optimizers()`，CLI 默认相同。 |
| 训练周期 | 5 epoch；warmup ratio 0.1 | 5 epoch、warmup ratio 0.1，同时 `warmup_epochs=1`；调度器在 `warmup_steps>0` 时优先使用 ratio。 |
| 梯度裁剪 | 1 | 1；CLI 梯度累积默认 1。 |
| 保存及选择 | 用最后一个微调模型测试 | 每 epoch 更新 `ckpt.pt`，默认每 5 epoch 另存 `ckpt-{epoch}.pt`；每 epoch 都打印验证和测试指标，正式报告必须取最终模型测试结果。 |
| 训练采样 | 多任务联合 | 六训练集按 TUAB、TUEV、SEED、HMC、Workload、TUSL 顺序 `ConcatDataset`，无显式任务均衡；每进程 DataLoader 10 workers，训练 `drop_last=True`。 |

每个 EEG 训练批次还取一批文本，两个 loss 相加后反向传播。文本是 OpenWebText，经 GPT-2 tiktoken 编码。`text_dataset_maker/prepare.py` 以种子 2357 留出 0.05% 作为验证，生成约 17 GB `train.bin` 与约 8.5 MB `val.bin`；作者脚本写到自己的目录，而训练入口要求 `--out_dir/text/train.bin`，路径必须衔接。即使 `--eval_only`，入口也先构建六数据集并读取首个文本训练批次，故不能拿单数据集和一个 checkpoint 直接运行原入口。

训练入口默认种子是 `1337`，本项目将显式用 `42/43/44`。DDP 各进程使用 `seed + rank`；代码只显式设 `torch.manual_seed()`，并非所有随机源的完整确定性锁定。`if compile:` 引用 Python 内置函数而非 `args.compile`，会尝试 `torch.compile()`；在 24 GB 卡上须先单独验证显存并记录任何修复。`--tokenizer_path` 默认错写为 `checkpoints/VQ.py`，但 instruction 的 `pretrained` 分支不会用这个路径初始化 VQ。

## 5. 提示词、答案和指标

代码把特殊 token `50257` 作为 EEG/文本分隔符，答案末尾使用 GPT-2 `<|endoftext|>`。训练 loss 只覆盖答案部分。TUAB 与 Workload 是 Yes/No（Yes=1）；SEED 是 Positive/Neutral/Negative；TUEV 为 A–F 六类，HMC 为 A–E 五类，TUSL 为 A–C 三类。完整原文提示和类别映射见 `downstream_dataset.py:81-90,248-253,355-364,454-460,546-554,640-645`，论文附录 B 表 5。

验证和测试调用 `generate(max_new_tokens=5)`，默认 `top_k=1` 逐 token 贪心生成，解析得到整数类别。二分类 `pr_auc` 使用 sklearn `average_precision_score`，`roc_auc` 使用 `roc_auc_score`；输入实际是解析后的 0/1 类别，不是连续概率。Balanced Accuracy、Cohen's Kappa、Weighted F1 分别由 sklearn 对预测类别计算。因此论文的 AUC-PR 文字定义和代码的 AP 实现存在口径差异，应按作者代码重算并注明。多分类解析失败返回 `-1`，随后 `np.eye(num_classes)[-1]` 会将其算作**最后一类**；二分类失败则把 `-1` 传入分数函数。应额外统计解析失败数，不能静默掩盖。

论文报告三次随机种子的均值和标准差，但不公开种子数值，也没有说明标准差采用总体或样本定义。本项目三种子值是项目约定，实际每次值、样本数、划分哈希、最终模型哈希、源码提交和日志必须分别保留。论文值与实测值分文件保存。

## 6. 前两阶段的论文参数（用于界定复现范围）

| 阶段 | 论文附录 D 参数摘要 |
| --- | --- |
| VQ tokenizer，表 7 | encoder 12 层、decoder 3 层、隐藏维 768、12 头、codebook 8192×128；EEG/text batch 512/128；AdamW β=(0.9,0.999)、weight decay 1e-4；lr 5e-5→1e-5 余弦；50 epoch、5 warmup、无梯度裁剪。 |
| NeuroLM-B 自回归预训练，表 8 | 12 层、隐藏维 768、12 头、254M 参数；EEG/text batch 480/32；AdamW β=(0.9,0.95)、weight decay 0.1；lr 6e-4→6e-5 余弦；20 epoch、2 warmup、梯度裁剪 1。 |

## 7. 当前可执行性和证据缺口

- 作者源码提交已固定，运行补丁仅应用于独立副本。服务器 B 与 VQ 两份权重的完整 SHA-256 已验证，且两份 checkpoint 均已读取结构；B 权重已严格加载，完成文本前向和两条真实 Workload EEG 输入的生成。两条答案均未解析为 Yes/No；这仍不是论文结果。
- Workload 的 PhysioNet EEGMAT v1.0.0 原始 72 份 EDF 已按官方清单逐文件校验；作者预处理已完成，得到 train/eval/test 为 1508/290/290 个样本，且受试者无交叉。其余五项任务尚无可直接训练的预处理数据，因此六任务联合微调和论文对照尚不能开始。
- 数据盘上的 Python 3.11.8、PyTorch 2.2.2/CUDA 12.1 已导入并识别 GPU；主要依赖已安装，`pip check` 通过，环境包快照已保存。PyTorch 2.2.2 与默认选中的 MKL 2025 不兼容，改用 MKL 2023.1.0 与 NumPy 1.26.4 后通过导入检查。
- 六数据集均需来源、许可、版本、原始文件校验、预处理日志、输出样本数及三划分成员清单。SEED 共享卷候选只发现 `.mat`，不是脚本需要的原始 `.cnt` 或最终 HDF5，来源与授权范围也尚待核对。
- 8 × 3090 24 GB 与论文 8 × A100 80 GB 不同；即便名义全局 batch 一致，实际显存、吞吐和可运行的精度/累积设置须验证，任何变化都应作为与论文不同的设置记录。

本文件不把论文样本数当作实际样本数，也不把权重已下载当作模型已成功加载。实测列在真实运行和日志核对之前保持空白。
