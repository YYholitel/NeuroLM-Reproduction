# 作者权重清单

来源：[Weibang/NeuroLM](https://huggingface.co/Weibang/NeuroLM/tree/main/checkpoints)，固定修订 `eddfff5c64a4139442f826d6c67c8369fd00f45a`。模型库页面标注 CC BY 4.0；具体使用仍以来源条款为准。

| 文件 | Hugging Face 标注大小 | SHA-256 | 状态 |
| --- | ---: | --- | --- |
| `NeuroLM-B.pt` | 2.38 GB | `ffe098bc138b89f8817d3710a3604498d8ecd15135080e2ca27735d05c6d29ab` | 本机与服务器文件已完整校验 |
| `VQ.pt` | 1.90 GB | `e792c39a6a9e6d1bf4604cf63090730424f1d37f942597883d0c0a1375a2663a` | 本机与服务器文件已完整校验 |

`NeuroLM-L.pt` 与 `NeuroLM-XL.pt` 暂未下载。PyTorch `.pt` 文件可能包含 pickle 序列化对象；加载前须核对来源和哈希，优先检查能否使用限制反序列化的加载方式。权重不进入 Git 仓库。

服务器的 `NeuroLM-B.pt` 与 `VQ.pt` 均通过固定修订的镜像地址续传；续传前以实际 GET Range 请求确认返回 `206`，传输后核对完整 SHA-256。服务器文件长度分别为 `2,377,399,148` 与 `1,904,671,888` 字节。

PyTorch 2.2.2 的 `weights_only=True` 无法读取 `NeuroLM-B.pt` 中的 NumPy scalar 元数据，且该版本不提供可添加安全类型的 `add_safe_globals`。[`scripts/inspect_checkpoint.py`](../scripts/inspect_checkpoint.py)先校验完整 SHA-256，只有显式传入 `--trusted-pickle` 才回退到 pickle 反序列化。B 权重通过该方式读到 `model_args`（12 层、12 头、768 维、1024 长度）和 240 个模型张量；这只是权重结构检查，并非结果评估。
