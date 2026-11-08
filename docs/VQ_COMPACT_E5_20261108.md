# VQ 容量与死码策略的五 epoch 对照

按用户指定顺序，先完成修复后 8192×128 VQ 的阶段二 100 步最小实验，再执行本次实验。提交入口会检查阶段二完成标记、完整性核验及两个节点退出码；未满足时不能提交。

## 固定条件与改动

本次从头初始化 **1024×64** 码本，连续 **50 步**未用的码在每 **50 步**检查时重置。基线为 8192×128、patience 100、interval 50。检查间隔原本已经是 50；本次改变的是死码判定所需的连续未使用步数。三个设置同时改变，不能据此单独归因于容量、维度或重置策略。

其余沿用已完成的五 epoch 基线：五数据集 459656 训练窗口，两个 8×H20 节点，全局 EEG/text batch 512/128，encoder/decoder 12/3、hidden 768，EMA .99、β=1、seed 42、50 epoch / 5 warmup 调度，4485 步恰好五完整 epoch，峰值/最小 LR 5e-5/1e-5。W&B 关闭。两次模型结构不同，因此不能要求初始化权重逐字节相同；数据清单和全部 16 rank 的输入顺序必须完全一致。

补丁 `0020-configurable-vq-codebook-capacity.patch` 只将 `train_vq.py` 的码本容量、维度开放为 CLI，并将 decoder 输入维度与之关联。默认仍是 8192×128；checkpoint 的 `vq_options` 保存容量和维度，禁止不同配置隐式恢复。旧 8192×128 checkpoint 不加载到本次模型。脚本 `apply_vq_capacity.py` 验证原 runtime 和补丁，输出全新 runtime。

真实 CPU 优化器/严格 checkpoint 回载检查使用 1024×64、hidden 768、缩小为 1/1 层的测试模型，确认 padding 不进入码本计数、梯度有限、量化器评估不变、使用率分母为 1024。实际训练仍是 12/3 层。

## 观测协议

记录逐步三项 loss、每 batch 活跃码比例、perplexity、perplexity/K、最大码份额、重置数及 epoch 末 100 步指标。原基线中已有的 512/128 阈值只是历史诊断提示，不作为本实验自动通过标准。更小的容量自然更容易获得较高使用率，必须同时看绝对 perplexity、重建误差和集中度。

最终对同一组 32 个训练诊断样本、4944 个有效 token 做冻结 checkpoint 评估，与基线最终权重比较。该组不是独立验证集，不据此宣称下游准确率提升。两轮数据顺序的 SHA-256 完全一致才标记对照核验完成。

另外复用阶段二已建立的 **160 个独立验证窗口**（五数据集各 32 个），通过 `audit_vq_heldout_probe.py` 分别评估两轮最终 checkpoint，记录有效 token 加权的频域/时域重建误差、commitment 和码本分配，并保存逐样本结果。`compare_vq_heldout_probe.py` 先评估已完成的基线，再等待紧凑模型五 epoch 完成后评估；整个过程冻结模型、确认量化器状态不变。这个均衡小探针不等同于完整验证总体或下游任务指标。

## 运行与产物

隔离根目录：`/mnt/cfs-omni/lhy/neurolm-reproduction/vq-compact-e5-20261108-r1`。提交和状态入口 `scripts/check_vq_compact_e5.py`；TI-ONE 配置见 `experiments/vq_compact_e5_tione_20261108.json`。

任务已提交：`train-1713086037633018624`，名称 `neurolm-vq-compact16-e5-20261108-r1`。提交时阶段二 100 步任务已 SUCCEED，冻结状态、权重更新、数据和产物检查全部通过。tmux 会话为 `vq-compact`，socket `/tmp/clawdbot-tmux-sockets/neurolm-vq-compact.sock`。

`scripts/observe_vq_compact_e5.py` 在 tmux 中监控、绘制真实日志对比，完成后核验 4485 步、运行冻结审计、导出哈希清单。训练由 TI-ONE 工作节点执行。失败即报告，不自动提交完整阶段二或额外实验。
