# 固定实验定义：NeuroLM-B 六项任务

机器可读配置为 [`experiments/neurolm_b_six_tasks.yaml`](../experiments/neurolm_b_six_tasks.yaml)。它固定作者源码提交、NeuroLM-B 与起始权重、六个数据集、论文划分、每项指标及本项目三个随机种子 `42 / 43 / 44`。论文只说明使用三个种子汇报均值与标准差，未列出种子值；这三个数字是本项目选择，不代表作者原始种子。[论文第 3.1–3.3 节及表 2–4](https://arxiv.org/pdf/2409.00101)为划分和指标依据。

每个种子运行一次六任务联合指令微调，并用该次最终模型分别评估六个测试集。`NeuroLM-B.pt` 是起始权重，不能直接作为论文表 2–4 的指令微调结果。论文称评估时使用最大 logit；固定提交的 `generate()` 默认 `top_k=1`，逐 token 取最大 logit 后解码并解析类别。解析失败如何处理仍需核对。

作者代码的二分类 `pr_auc` 对应 `sklearn.metrics.average_precision_score`，`roc_auc` 对应 `roc_auc_score`，但传入的是解析出的 0/1 类别，而非连续分数。配置中的 `auc_pr`、`auroc` 按这一作者实现映射；若改用概率分数，应另记为不同评估协议。其余指标使用 sklearn 的 balanced accuracy、Cohen's kappa 和 weighted F1。

论文数值单独保存在 [`results/paper_neurolm_b.csv`](../results/paper_neurolm_b.csv)，每行对应一个数据集和指标的论文均值、标准差。实测值只写入 [`results/measured_neurolm_b.csv`](../results/measured_neurolm_b.csv)：每个种子、数据集、指标一行，合计应有 54 行；当前只有表头，因为尚未运行。`value` 必须来自实际测试日志，另填测试样本数、划分清单和最终权重的 SHA-256、源码提交及日志路径。三个种子的汇总值从实测行计算，不能复制论文均值。

运行前仍需生成并冻结实际训练、验证、测试成员清单及哈希。TUAB/TUEV 按原始训练患者做 80%/20% 划分，并保留原始测试集；SEED 按每个 session 的试次时间顺序划分；HMC 和 Workload 按论文指定的受试者顺序或编号划分。TUSL 作者代码按未排序的 EDF 文件遍历顺序切 60% 训练、接着 20% 测试、最后 20% 验证，不能仅凭比例声称划分可重现。原始数据来源、预处理版本及样本数也需在运行前核对。

论文和作者代码逐项核查见[实验细节审计](EXPERIMENT_DETAILS.md)。
