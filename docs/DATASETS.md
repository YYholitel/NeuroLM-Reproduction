# 下游数据清单

论文第 3.1 节使用六个数据集。下表目录是作者 `train_instruction.py` 和 `utils.py` 相对 `--dataset_dir` 读取的预处理后布局；各项原始数据与样本数的核对进度见下文。

| 数据集 | 代码期望的相对位置 | 论文划分 | 当前状态 |
| --- | --- | --- | --- |
| TUAB | `TUAB/processed/{train,val,test}` | 原测试集；训练患者再按 80%/20% 分训练/验证 | v3.0.1 原始副本已放在服务器数据盘，复制校验及预处理待完成 |
| TUEV | `TUEV/{processed_train,processed_eval,processed_test}` | 原测试集；训练患者再按 80%/20% 分训练/验证 | v2.0.1 原始副本已放在服务器数据盘，复制校验及预处理待完成 |
| SEED | `h5data/seed-3.hdf5` | 每个 session 的试次按时间顺序 9:3:3 | 未准备 |
| HMC | `HMC/{train,eval,test}` | 前 100 人训练、中间 25 人验证、后 26 人测试 | v1.1 原始 151 组已校验；预处理完成，输出 91,248/22,124/23,871 个窗口；尚无论文指标 |
| Workload | `EEGWorkload/{train,eval,test}` | 受试者 0–25 训练、26–30 验证、31–35 测试 | 原始文件校验、预处理、划分计数及起始权重烟测已完成；尚无论文指标 |
| TUSL | `TUSL/{train,eval,test}` | 60%/20%/20% | v2.0.1 原始副本已放在服务器数据盘，复制校验及预处理待完成 |

## 2026-10-15 服务器候选数据核验

- 共享卷发现 SEED 命名的归档临时目录，含 `.mat` 特征和预处理文件；它不在项目专用数据目录中，也不是训练入口需要的 `h5data/seed-3.hdf5`。
- 作者 `dataset_maker/prepare_SEED.py` 从原始 `.cnt` 读取 EEG，按 0.1–75 Hz 滤波、50 Hz 陷波、200 Hz 重采样后写入 HDF5。候选目录未发现 `.cnt`，不能直接用此脚本重建论文输入；文件名相似不能证明预处理版本一致。
- SEED 官方[下载说明](https://bcmi.sjtu.edu.cn/home/seed/downloads.html)要求提交签署的许可协议并经审核；[许可协议](https://bcmi.sjtu.edu.cn/~seed/resource/license/license.pdf)限制使用与分发。项目使用者表示机构已有授权，但授权由他人办理，尚无可核对的授权范围或候选副本来源记录。该目录暂不计为已准备数据，也不用于实验。
- 专用数据目录仍未找到六项任务所需的预处理文件。其余共享卷候选尚无足以确认论文版本的数据与许可记录。

## 2026-10-20 共享卷补充核验

- 共享数据集合中找到 TUAB v3.0.1、TUEV v2.0.1、TUSL v2.0.1 的 EDF 原始文件，计数分别为 2993、518、112。集合的迁移说明显示这些文件从旧共享目录搬迁而来；另一共享卷也有相同版本的目录。2026-10-21 已从该共享集合复制到服务器项目数据盘，对应 EDF 计数为 2993、518、112；逐文件内容比较仍在运行。申请人的使用范围、两处共享副本是否一致，以及它们与论文实际版本及划分是否一致，尚未核实；未用于训练。
- 作者 TUAB 预处理脚本指定 v3.0.0；共享卷主版本为 v3.0.1。其随附说明称 v3.0.1 修改 EDF 文件头、信号数据未变，但仍须记录此版本差异并实测预处理结果。
- 已制作 [`0006-tuh-preprocess.patch`](../patches/0006-tuh-preprocess.patch)，使三个脚本从项目数据盘读取原始文件、向独立预处理目录写出，并固定此前依赖集合或文件系统顺序的划分。TUEV 的 518 份 EDF 均有同名 `.rec`；TUSL 的 112 份 EDF 中，111 份有 `.tse_agg`，1 份只有同格式的 `.tse_ag`，补丁对后一种名称作回退。上述版本及划分差异尚待实际预处理结果核对。
- 共享卷的 SEED 系列集合含 SEED-V、SEED-IV 等变体；另有两个名为 SEED 的归档临时目录，主要包含 `.mat` 特征和预处理文件。对这两个目录的扫描未发现作者预处理所需的 `.cnt`。SEED-V 等变体的 `.cnt` 不能当作论文所用原版 SEED；许可范围也尚未核实。
- 另发现原版三分类 SEED 的 15 个按受试者分开的 HDF5 衍生文件。其说明记录 200 Hz、0.1–75 Hz、50 Hz 陷波及 2 秒无重叠片段；抽查文件为 `trial/segment/eeg` 层级，单段形状 62×400。逐文件元数据检查显示每名受试者有 3 个 session、共 39 个 trial；官方原版说明为每 session 15 个 trial，共 45 个。作者 `SEEDDataset` 则要求单个 `seed-3.hdf5`，每个受试者组含连续 `eeg`、`trialStart`、`trialEnd`、`label` 等属性。这批 HDF5 不能直接传给作者加载器；其原始文件来源、缺少的 trial 和许可范围尚待核对。

## Workload 原始数据

- 来源：[PhysioNet EEG During Mental Arithmetic Tasks v1.0.0](https://physionet.org/content/eegmat/1.0.0/)，开放访问，Open Data Commons Attribution License v1.0。固定版本含 36 名受试者、每人休息与心算各 1 份 EDF，共 72 份。
- 使用 [`scripts/download_eegmat.sh`](../scripts/download_eegmat.sh) 下载至服务器数据盘，并以来源 `SHA256SUMS.txt` 验证全部 72 份 EDF 及元数据文件；原始数据和元数据不进入 Git 仓库。
- 作者预处理脚本按文件名排序分成 52/10/10 份 EDF，即受试者 0–25/26–30/31–35；每份截取末 60 秒并产生 29 个 4 秒重叠窗口。实测输出为训练/验证/测试 1508/290/290 个样本，总数 2088，与论文表 1 一致；日志中的单样本形状为 19×800。运行日志未出现错误或跳过文件。实际成员及输出文件名哈希见[划分清单](../experiments/workload_split_manifest.json)；哈希只覆盖排序后的文件名，不代表 EEG 内容校验。

## 其余数据的官方入口

- [NEDC 的 TUH EEG 数据页面](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/)列有 TUAB、TUEV、TUSL 的当前发布版本和申请流程；页面要求提交访问表并取得专用凭据。当前发布版本不能自动视为论文使用的版本。
- [PhysioNet HMC sleep staging v1.1](https://physionet.org/content/hmc-sleep-staging/1.1/)为开放访问，含 151 名受试者、下载文件清单和 SHA-256 清单。原始文件已校验，预处理输出的受试者数和窗口计数已核对。

## HMC 原始数据核对（2026-10-20 更新）

- 官方 v1.1 `RECORDS` 列出 151 份 `recordings/SN*.edf`；同一目录下有对应 `_sleepscoring.txt`，来源 `SHA256SUMS.txt` 为两类文件提供校验值。许可文件为 CC BY 4.0。
- `SN001.edf` 文件头列出四个 `EEG F4-M1 / C4-M1 / O2-M1 / C3-M2` 通道，以及作者脚本要丢弃的 EMG、两路 EOG 和 ECG；其标注文本列名及 30 秒睡眠阶段行与作者脚本读取方式一致。
- [`scripts/download_hmc.sh`](../scripts/download_hmc.sh)按清单获取 151 组 EDF 与标注文本，逐文件对照官方 SHA-256。服务器下载日志于 2026-10-16 记录“Verified 151 HMC v1.1 EDF and sleep scoring text pairs against PhysioNet SHA256SUMS.txt”；2026-10-20 只读复核确认目录中有 151 份 EDF、151 份标注文件。2026-10-21 作者预处理脚本在独立源码副本中运行结束，输出训练/验证/测试窗口 91,248/22,124/23,871，总计 137,243；各划分分别含 100/25/26 名受试者，日志没有出现异常窗口提示。抽查样本字段为 `X`、`ch_names`、`y`，`X` 形状为 4×6000；作者 `HMCLoader` 已成功读取并形成 7 个训练输入张量。
- [`0003-hmc-paths.patch`](../patches/0003-hmc-paths.patch)只把作者 HMC 预处理脚本的输入、输出绝对路径改为参数，未改动滤波、标签或划分逻辑。[`verify_hmc_split.py`](../scripts/verify_hmc_split.py)已核对 151 名受试者按排序后的 100/25/26 划分、每人的窗口编号连续且无重复；[划分清单](../experiments/hmc_split_manifest.json)保存受试者名单、逐人窗口计数及文件名哈希。文件名哈希不代表 EEG 内容校验；逐样本标签尚未审计。

作者代码还读取 `--out_dir/text/train.bin`，这是语言数据的 token 文件，并非上述六个 EEG 数据集之一。作者 `text_dataset_maker/prepare.py` 使用 OpenWebText，预计先产生约 54 GB 下载缓存，再产生约 17 GB 的 `train.bin`。[`0005-text-output-revision.patch`](../patches/0005-text-output-revision.patch)允许固定数据集修订并直接写入训练入口期望目录，不改变分割、分词或二进制格式。2026-10-21 已核对固定修订 `79d93d786212f7344586290adb811d4ae6a1762c` 有 8,013,769 条文本；下载仍在运行，未生成 token 文件。

## 数据接收规则

- 先记录原始来源、授权、版本和文件校验，再运行预处理。
- 私有或需申请访问的数据只留在服务器数据盘，不上传 GitHub。
- 保存受试者级划分清单和样本数，避免训练、验证与测试泄漏。
