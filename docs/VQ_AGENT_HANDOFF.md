# VQ 训练连接与协作交接

本页用于其他 agent 同步现有运行。先读 [AGENTS.md](../AGENTS.md) 和 [STS 连接说明](TIONE_STS_CONNECTION.md)，优先做只读检查。已有后台进程负责迁移、校验和训练提交，协作者应先核对状态再决定是否需要操作，避免重复提交或同时修改运行目录。

## 连接与算力

| 用途 | 入口与目录 |
| --- | --- |
| 原始数据、CPU 预处理 | SSH `lihy`；`lihy@10.20.38.38:1022`；主机 `nccserv0`；项目 `/data/lihy/NeuroLM` |
| 已验证源端 Python | `/data/lihy/NeuroLM/envs/neurolm-py311/bin/python` |
| TI-ONE Notebook | SSH `tencent`；`lhy@43.137.68.228:11076`；项目 `/mnt/cfs-omni/lhy/neurolm-reproduction` |
| 训练资源 | `ap-beijing`，资源组 `rsg-b227f5pp`，申请 2 节点 × 8 H20 |
| 每节点申请规格 | 112 CPU、896 GiB 内存、8 GPU；实际分配以 API 和节点实测为准 |
| 共享存储 | CFS `cfs-ini8ib89`，挂载 `/mnt/cfs-omni` |
| 训练镜像 | `omni-cr.tencentcloudcr.com/kexin/pretrain:0.0.4` |

这两个 SSH 入口不是两个 H20 训练节点。TI-ONE 会为任务创建训练节点并注入 rendezvous 信息，不要使用 Notebook SSH 地址替代 `MASTER_ADDR`。

本机工作区为 `C:\Users\PC\Documents\ChatGPT\Neuro_LLM`。SSH 配置和登录凭证不随 Git 分发；其他环境需要已有的独立授权访问。临时令牌、STS 地址及密钥只在已授权远端使用，不能粘贴到聊天或复制到本机。没有现成权限时说明缺少哪一项条件，不索取其他人的私钥或长期 API Key。

## 可复用的只读命令

在工作区运行：

```powershell
.\scripts\Test-TioneSts.ps1
python scripts/check_vq_full.py status
python scripts/check_vq_full.py files
```

检查本次认证 JSON、退出码和临时内核清理结果；不要用历史成功记录替代新实测。`status` 查询指定任务名，`files` 返回远端迁移、loss、epoch 汇总和 GPU 采样。还支持 `logs`、`details`、`events`，这些命令不提交或停止任务。脚本依赖本次已部署的 CFS `control.py` 和 `inspect.py`；它不是从空环境部署整个系统的一键脚本。

## 正在执行的工作

- 任务名：`neurolm-vq-five-full16-e50-20261107-r1`。
- 腾讯运行根目录：`/mnt/cfs-omni/lhy/neurolm-reproduction/vq-five-full-20261107-r1`。
- 源端运行目录：`/data/lihy/NeuroLM/outputs/vq-five-full-20261107-r1`。
- 五数据集 train 划分共 459,656 个窗口；保留验证和测试划分，不是论文 25,000 小时语料复现。
- encoder 12、两个 decoder 各 3 层，码本 8192×128；每卡 EEG/text batch 32/8，全局 512/128；50 epochs，warmup 5，LR 5e-5 → 1e-5；W&B 关闭。
- TUAB/TUEV 从原始 EDF 重新做 60 Hz 陷波，其他三套保留既有信号。分片内为 `X + ch_names` pickle，微伏值在加载器中 `/100`。

完整方案、检查结果和补丁说明见 [运行记录](VQ_FIVE_FULL_20261107.md)。[状态 JSON](../results/vq_five_full_status_20261107.json) 在 Git 中是提交时的快照，不是实时监控；**必须检查时间戳并重新查询**。准备、提交、排队与实际训练是不同状态，数据迁移不能写成“模型正在训练”。

## 日志与监控

腾讯运行根目录下：

- `pipeline_state.json`、`pipeline.log`：阶段及校验/提交过程。
- `cos_ingest.log`：接收数据；源端看 `cos_upload.log` 和 `cos_upload.pid`。
- `submitted.json`：提交后才有实际任务 ID。
- `run/metrics.jsonl`：逐步全局平均 loss。
- `run/epoch_metrics.jsonl`、`run/review_checkpoints.jsonl`：每轮统计和阶段检查。
- `run/node-0/gpu_metrics.csv`、`run/node-1/gpu_metrics.csv`：每 5 秒采样实际训练 GPU。
- `run/checkpoints/VQ/ckpt.pt`：每轮更新的检查点。

关注第 1、5、20–30 epoch 的重建/码本/对抗 loss、LR 和排除 padding 的码本使用率。第 5 epoch 末核对 LR 为 5e-5。固定诊断评估用于比较本次运行内的变化；smoke 的 2.86 来自两个诊断样本，不是全量 loss 的早停阈值。NaN/Inf 会终止训练，其他检查标记需结合持续趋势分析。

协作者首次同步请报告：连接是否成功、数据进度、实际任务 ID/状态、是否已有优化器更新，以及本次任务的 GPU 利用率和显存。未分配节点时报告暂无指标，不能据此断言整个资源组的 GPU 利用率为 0%。
