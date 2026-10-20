# 下游数据清单

论文第 3.1 节使用六个数据集。下表目录是作者 `train_instruction.py` 和 `utils.py` 相对 `--dataset_dir` 读取的预处理后布局；原始数据获取方式和实际样本数尚未核对。

| 数据集 | 代码期望的相对位置 | 论文划分 | 当前状态 |
| --- | --- | --- | --- |
| TUAB | `TUAB/processed/{train,val,test}` | 原测试集；训练患者再按 80%/20% 分训练/验证 | 未准备 |
| TUEV | `TUEV/{processed_train,processed_eval,processed_test}` | 原测试集；训练患者再按 80%/20% 分训练/验证 | 未准备 |
| SEED | `h5data/seed-3.hdf5` | 每个 session 的试次按时间顺序 9:3:3 | 未准备 |
| HMC | `HMC/{train,eval,test}` | 前 100 人训练、中间 25 人验证、后 26 人测试 | v1.1 原始 151 组 EDF 与标注已校验；尚未预处理 |
| Workload | `EEGWorkload/{train,eval,test}` | 受试者 0–25 训练、26–30 验证、31–35 测试 | 原始文件校验、预处理、划分计数及起始权重烟测已完成；尚无论文指标 |
| TUSL | `TUSL/{train,eval,test}` | 60%/20%/20% | 未准备 |

## 2026-10-15 服务器候选数据核验

- 共享卷发现 SEED 命名的归档临时目录，含 `.mat` 特征和预处理文件；它不在项目专用数据目录中，也不是训练入口需要的 `h5data/seed-3.hdf5`。
- 作者 `dataset_maker/prepare_SEED.py` 从原始 `.cnt` 读取 EEG，按 0.1–75 Hz 滤波、50 Hz 陷波、200 Hz 重采样后写入 HDF5。候选目录未发现 `.cnt`，不能直接用此脚本重建论文输入；文件名相似不能证明预处理版本一致。
- SEED 官方[下载说明](https://bcmi.sjtu.edu.cn/home/seed/downloads.html)要求提交签署的许可协议并经审核；[许可协议](https://bcmi.sjtu.edu.cn/~seed/resource/license/license.pdf)限制使用与分发。项目使用者表示机构已有授权，但授权由他人办理，尚无可核对的授权范围或候选副本来源记录。该目录暂不计为已准备数据，也不用于实验。
- 专用数据目录仍未找到六项任务所需的预处理文件。其余共享卷候选尚无足以确认论文版本的数据与许可记录。

## Workload 原始数据

- 来源：[PhysioNet EEG During Mental Arithmetic Tasks v1.0.0](https://physionet.org/content/eegmat/1.0.0/)，开放访问，Open Data Commons Attribution License v1.0。固定版本含 36 名受试者、每人休息与心算各 1 份 EDF，共 72 份。
- 使用 [`scripts/download_eegmat.sh`](../scripts/download_eegmat.sh) 下载至服务器数据盘，并以来源 `SHA256SUMS.txt` 验证全部 72 份 EDF 及元数据文件；原始数据和元数据不进入 Git 仓库。
- 作者预处理脚本按文件名排序分成 52/10/10 份 EDF，即受试者 0–25/26–30/31–35；每份截取末 60 秒并产生 29 个 4 秒重叠窗口。实测输出为训练/验证/测试 1508/290/290 个样本，总数 2088，与论文表 1 一致；日志中的单样本形状为 19×800。运行日志未出现错误或跳过文件。实际成员及输出文件名哈希见[划分清单](../experiments/workload_split_manifest.json)；哈希只覆盖排序后的文件名，不代表 EEG 内容校验。

## 其余数据的官方入口

- [NEDC 的 TUH EEG 数据页面](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/)列有 TUAB、TUEV、TUSL 的当前发布版本和申请流程；页面要求提交访问表并取得专用凭据。当前发布版本不能自动视为论文使用的版本。
- [PhysioNet HMC sleep staging v1.1](https://physionet.org/content/hmc-sleep-staging/1.1/)为开放访问，含 151 名受试者、下载文件清单和 SHA-256 清单。原始文件已校验；还需核对作者预处理脚本的实际输出与划分。

## HMC 原始数据核对（2026-10-20 更新）

- 官方 v1.1 `RECORDS` 列出 151 份 `recordings/SN*.edf`；同一目录下有对应 `_sleepscoring.txt`，来源 `SHA256SUMS.txt` 为两类文件提供校验值。许可文件为 CC BY 4.0。
- `SN001.edf` 文件头列出四个 `EEG F4-M1 / C4-M1 / O2-M1 / C3-M2` 通道，以及作者脚本要丢弃的 EMG、两路 EOG 和 ECG；其标注文本列名及 30 秒睡眠阶段行与作者脚本读取方式一致。
- [`scripts/download_hmc.sh`](../scripts/download_hmc.sh)按清单获取 151 组 EDF 与标注文本，逐文件对照官方 SHA-256。服务器下载日志于 2026-10-16 记录“Verified 151 HMC v1.1 EDF and sleep scoring text pairs against PhysioNet SHA256SUMS.txt”；2026-10-20 只读复核确认目录中有 151 份 EDF、151 份标注文件，下载进程已结束。尚未运行作者预处理，因此 HMC 仍不计作可用训练数据。
- [`0003-hmc-paths.patch`](../patches/0003-hmc-paths.patch)只把作者 HMC 预处理脚本的输入、输出绝对路径改为参数，未改动滤波、标签或划分逻辑。后续需保留实际 100/25/26 名单及窗口计数。

作者代码还读取 `--out_dir/text/train.bin`，这是语言数据的 token 文件，并非上述六个 EEG 数据集之一。作者 `text_dataset_maker/prepare.py` 使用 OpenWebText，预计先产生约 54 GB 下载缓存，再产生约 17 GB 的 `train.bin`；其输出目录与训练入口期望目录不同，须显式衔接并记录文件校验。

## 数据接收规则

- 先记录原始来源、授权、版本和文件校验，再运行预处理。
- 私有或需申请访问的数据只留在服务器数据盘，不上传 GitHub。
- 保存受试者级划分清单和样本数，避免训练、验证与测试泄漏。
