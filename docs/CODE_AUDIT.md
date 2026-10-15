# 作者代码审计

审计版本：作者仓库提交 `0cda9876d8ce6ee07ed0c43eee5e9a6f5c24b177`。以下为静态代码检查结果，尚未执行训练。

## 指令微调入口

- `train_instruction.py` 的 `main()` 固定依次构建 TUAB、TUEV、SEED、HMC、Workload、TUSL 六个数据集；当前没有选择单个数据集的 CLI 参数。
- 程序从 `out_dir/text/train.bin` 读取文本 token；即使使用 `--eval_only`，也会先初始化文本批次。作者的 `text_dataset_maker/prepare.py` 使用 OpenWebText，说明文件估计下载缓存约 54 GB、输出训练 token 文件约 17 GB；该脚本把文件写到自身目录，需要另行放到入口期望的 `out_dir/text/`。
- `--NeuroLM_path` 默认指向 `checkpoints/NeuroLM-B.pt`，作者 README 将其用于指令微调的起始权重。仅评估该起始权重不能等同于论文表 2–4 的微调结果。
- `train_instruction.py`、`train_pretrain.py` 和 `train_vq.py` 均使用 `if compile:`，引用 Python 内置函数而不是 `args.compile`；按当前写法会尝试执行 `torch.compile(model)`。修改前需保留补丁并记录差异。
- `--tokenizer_path` 的默认值为 `checkpoints/VQ.py`，而作者提供的权重文件名是 `VQ.pt`。指令入口从 `NeuroLM-B.pt` 初始化时不会读取这个参数；它只出现在当前不可由 CLI 选择的 `init_from == 'gpt'` 分支。B 权重自身的 `model` 中已有 tokenizer 参数。补丁仍修正默认扩展名，避免以后启用该分支时读取错误文件。
- 论文环境的 PyTorch 2.2.2 不提供 `torch.amp.GradScaler`；三个训练入口均需改用该版本支持的 `torch.cuda.amp.GradScaler`。这些运行兼容性修改与 `--compile`、默认权重文件名修复一起保存在 [`0002-runtime-compat.patch`](../patches/0002-runtime-compat.patch)。补丁只应用于独立的作者源码副本。

## 数据目录接口

详见[数据清单](DATASETS.md)。当前服务器的专用数据目录尚无论文所需数据，需逐项核对可用来源和权限。
SEED 预处理脚本要求原始 `.cnt`，并含固定的绝对输入与输出路径；共享卷发现的 `.mat` 候选不能直接用于该脚本或训练入口。
Workload 预处理脚本的原始输入路径为作者机器上的固定绝对路径；[`0001-workload-paths.patch`](../patches/0001-workload-paths.patch)增加 `--raw_dir` 与 `--out_dir`，保留原有预处理及排序划分逻辑。

## 指标实现与论文口径

- 固定提交的 `evaluate()` 调用 `model.generate()`，解码后由 `get_pred()` 解析类别。`generate()` 默认 `top_k=1`，逐 token 保留最大 logit，与论文第 3.2 节的选取最大 logit 描述相符；答案解析仍需单独核查。
- TUAB、Workload 的 `pr_auc` 与 `roc_auc` 分别调用 sklearn 的 `average_precision_score`、`roc_auc_score`，但输入是解析后的 0/1 类别，因此仅有两个分数档位。与使用连续概率计算的曲线面积不应混用。
- 多分类任务把解析类别变成 one-hot 后计算 balanced accuracy、Cohen's kappa 和 weighted F1。若解析失败，`get_pred()` 返回 `-1`，`np.eye(num_classes)[-1]` 会把它当作最后一类；运行时应记录解析失败数。

## 后续审计

1. 核对作者权重的 checkpoint 字段与模型构造参数。
2. 核对预处理脚本、数据划分及指标函数与论文表 2–4 的一致性。
3. 对必要的代码修复制作独立补丁，并在实验记录中区分原版与修改版。
