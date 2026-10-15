# 下游数据清单

论文第 3.1 节使用六个数据集。下表目录是作者 `train_instruction.py` 和 `utils.py` 相对 `--dataset_dir` 读取的预处理后布局；原始数据获取方式和实际样本数尚未核对。

| 数据集 | 代码期望的相对位置 | 论文划分 | 当前状态 |
| --- | --- | --- | --- |
| TUAB | `TUAB/processed/{train,val,test}` | 原测试集；训练患者再按 80%/20% 分训练/验证 | 未准备 |
| TUEV | `TUEV/{processed_train,processed_eval,processed_test}` | 原测试集；训练患者再按 80%/20% 分训练/验证 | 未准备 |
| SEED | `h5data/seed-3.hdf5` | 每个 session 的试次按时间顺序 9:3:3 | 未准备 |
| HMC | `HMC/{train,eval,test}` | 前 100 人训练、中间 25 人验证、后 26 人测试 | 未准备 |
| Workload | `EEGWorkload/{train,eval,test}` | 受试者 0–25 训练、26–30 验证、31–35 测试 | 未准备 |
| TUSL | `TUSL/{train,eval,test}` | 60%/20%/20% | 未准备 |

作者代码还读取 `--out_dir/text/train.bin`，这是语言数据的 token 文件，并非上述六个 EEG 数据集之一。作者 `text_dataset_maker/prepare.py` 使用 OpenWebText，预计先产生约 54 GB 下载缓存，再产生约 17 GB 的 `train.bin`；其输出目录与训练入口期望目录不同，须显式衔接并记录文件校验。

## 数据接收规则

- 先记录原始来源、授权、版本和文件校验，再运行预处理。
- 私有或需申请访问的数据只留在服务器数据盘，不上传 GitHub。
- 保存受试者级划分清单和样本数，避免训练、验证与测试泄漏。
