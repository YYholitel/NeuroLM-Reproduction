# 决策记录

## 2026-10-15：固定作者源码

- 论文：[NeuroLM](https://arxiv.org/pdf/2409.00101)。
- 作者实现：[935963004/NeuroLM](https://github.com/935963004/NeuroLM)。
- 固定提交：`0cda9876d8ce6ee07ed0c43eee5e9a6f5c24b177`。
- 原因：避免上游更新改变实验代码或配置。

## 2026-10-15：以 NeuroLM-B 建立首条复现链路

- 论文将未另行注明的 NeuroLM 结果对应 NeuroLM-B，参数量为 254M。
- 先验证数据、权重、推理和指标，再扩展到六项任务与三次随机种子。
- 任何与论文环境不同的设置都在复现记录中披露。

## 2026-10-15：建立独立公开仓库

- 仓库：[YYholitel/NeuroLM-Reproduction](https://github.com/YYholitel/NeuroLM-Reproduction)。
- 仅公开脚本、配置、日志摘要与可分享结果；原始 EEG、权重、凭据和内部服务器路径不进入 Git 历史。
