# 作者代码审计

审计版本：作者仓库提交 `0cda9876d8ce6ee07ed0c43eee5e9a6f5c24b177`。以下为静态代码检查结果，尚未执行训练。

## 指令微调入口

- `train_instruction.py` 的 `main()` 固定依次构建 TUAB、TUEV、SEED、HMC、Workload、TUSL 六个数据集；当前没有选择单个数据集的 CLI 参数。
- 程序从 `out_dir/text/train.bin` 读取文本 token；即使使用 `--eval_only`，也会先初始化文本批次。作者的 `text_dataset_maker/prepare.py` 使用 OpenWebText，说明文件估计下载缓存约 54 GB、输出训练 token 文件约 17 GB；该脚本把文件写到自身目录，需要另行放到入口期望的 `out_dir/text/`。
- `--NeuroLM_path` 默认指向 `checkpoints/NeuroLM-B.pt`，作者 README 将其用于指令微调的起始权重。仅评估该起始权重不能等同于论文表 2–4 的微调结果。
- `train_instruction.py`、`train_pretrain.py` 和 `train_vq.py` 均使用 `if compile:`，引用 Python 内置函数而不是 `args.compile`；按当前写法会尝试执行 `torch.compile(model)`。修改前需保留补丁并记录差异。
- `--tokenizer_path` 的默认值为 `checkpoints/VQ.py`，而作者提供的权重文件名是 `VQ.pt`。运行时应显式传入正确文件名；还需确认该参数在哪条初始化路径实际使用。

## 数据目录接口

详见[数据清单](DATASETS.md)。当前服务器的专用数据目录尚无论文所需数据，需逐项核对可用来源和权限。

## 后续审计

1. 核对作者权重的 checkpoint 字段与模型构造参数。
2. 核对预处理脚本、数据划分及指标函数与论文表 2–4 的一致性。
3. 对必要的代码修复制作独立补丁，并在实验记录中区分原版与修改版。
