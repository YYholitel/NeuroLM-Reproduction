# Agent 工作入口

## TI-ONE 连接

涉及腾讯云 TI-ONE、服务器连接或 STS 临时认证时，先阅读
[docs/TIONE_STS_CONNECTION.md](docs/TIONE_STS_CONNECTION.md)。

本仓库已验证的入口是 `scripts/Test-TioneSts.ps1`，它通过 `tencent` SSH
在远端 Notebook 环境中调用用户指定的 `tione-platform-ops` skill，执行一次
TI-ONE 只读查询，并清理本次临时内核。具体目标、前提、授权范围与故障处理见文档。

- 复用现有脚本；不要把 SSH 登录成功当作 TI-ONE API 认证成功。
- 以本次执行的 JSON 和退出码报告状态；文档中的历史结果不能冒充新实测。
- 令牌、临时密钥和 STS 地址只在已授权服务器内使用，不输出、不保存或复制到本机。
- 连接检查的授权不扩展为训练提交、停止任务、权限修改或其他 root 操作。
- 这条路径使用 Notebook 的临时身份；`lhy` SSH 会话缺少 STS 入口时，
  不要反复在该会话直接运行原 skill CLI，也不要据此要求提供长期 API 密钥。
