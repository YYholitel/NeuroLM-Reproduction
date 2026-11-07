# TI-ONE 连接与 STS 临时身份验证（Agent 调用说明）

本仓库已验证的连接入口是 [Test-TioneSts.ps1](../scripts/Test-TioneSts.ps1)。它通过现有 SSH 连接，在服务器的 Notebook 环境中调用用户指定的 skill，取得 STS 临时身份，并执行一次 TI-ONE 只读查询。

最近实测：**2026-11-06，Asia/Shanghai，两次成功**。这份记录只证明认证和查询连通，不代表训练已经执行或 NeuroLM 实验已经复现。

## 1. 直接调用

在本仓库根目录的 **Windows PowerShell** 中运行：

```powershell
& .\scripts\Test-TioneSts.ps1
```

当前电脑的绝对路径调用：

```powershell
& 'C:\Users\PC\Documents\ChatGPT\Neuro_LLM\scripts\Test-TioneSts.ps1'
```

入口文件会把 [Python 探针](../scripts/tione_notebook_sts_probe.py)通过标准输入送到服务器执行，实际命令等价于：

```powershell
Get-Content -LiteralPath .\scripts\tione_notebook_sts_probe.py -Raw |
    ssh -o BatchMode=yes -o ConnectTimeout=10 tencent 'python -'
```

使用 Linux/macOS 的 agent，可在已配置同一授权目标的 SSH 别名后运行：

```bash
ssh -o BatchMode=yes -o ConnectTimeout=10 tencent 'python -' < scripts/tione_notebook_sts_probe.py
```

Python 探针必须在远端 Linux 开发机执行；在 Windows 本机直接运行它会找不到 `/proc` 和远端 skill。

## 2. 当前目标与运行前提

| 项目 | 当前已验证配置 |
| --- | --- |
| 本机 SSH 别名 | `tencent` |
| SSH 目标 | `43.137.68.228:11076`，用户 `lhy` |
| Notebook 主机名 | `nb-1618225634846853120-cz56w4reusxs` |
| TI-ONE 区域 | `ap-beijing` |
| Notebook 接口 | 服务器内部 `http://127.0.0.1:8888` |
| Notebook 服务用户 | `root` |
| 用户指定的 skill | `/mnt/cfs-omni/chr/video_reasoning/cvpr2027/skills/tione-platform-ops/SKILL.md` |
| skill 的 Python helper | 同目录下的 `tione_ops.py` |

调用前提：

- 本机已经有可用的 `tencent` SSH 配置和登录凭证，支持 `BatchMode=yes` 非交互连接。SSH 私钥保留在本机，不写入仓库。
- 远端 SSH 使用的 `python` 可以导入 `requests` 和 `websocket`；后者的安装包名是 `websocket-client`。
- 目标开发机和现有 Jupyter 服务正在运行，授权范围内的 Jupyter 启动参数可以读取。
- Notebook 的 Python 内核可以导入 `tikit`、读取上述 skill，并具有平台注入的 `QCLOUD_CONTAINER_INSTANCE_CREDENTIALS_URL`。
- 当前探针按这台开发机的 Jupyter 配置实现，端口固定为 `8888`；`base_url` 从启动参数读取。换服务器或出现多个 Jupyter 服务时，应先核对目标配置。

## 3. 连接流程

1. 本地 agent 使用 `tencent` 别名，以 `lhy` 登录开发机。
2. 探针仅在服务器内读取已获授权的现有 `root` Jupyter 启动令牌，用它访问本机 Jupyter API。
3. 创建本次检查专用的临时 Python 内核，优先选择 `python3`。
4. 在该内核中导入指定的 `tione_ops.py`，调用 `_clear_proxy()` 和 `_fetch_sts_creds()`，取得平台提供的 `TmpSecretId + TmpSecretKey + Token`。
5. 显式携带三项临时凭证创建 `tikit.client.Client`，执行：
   ```python
   client.describe_training_tasks(limit=1, offset=0)
   ```
6. 返回不含凭证的结果 JSON，关闭连接并删除本次创建的临时内核。

这里复用了原 skill 的认证函数，**没有直接运行原 skill 的 `status` CLI**，也不会回退到长期 SecretId/SecretKey。

`lhy` 的 SSH 会话没有继承 STS 入口，而 Notebook 环境中存在该入口。因此，单独运行 `ssh tencent` 后直接执行原 skill CLI，仍可能报告：

```text
auth failed: no STS credentials and no explicit keys. Not proceeding.
```

应优先使用本仓库入口。它提供可重复的连接验证方式，没有修改 `lhy` 的环境变量继承。

## 4. 成功判据与实测证据

下面是 **2026-11-06 保存入口后的真实运行结果**：

```json
{
  "ok": true,
  "auth": "STS",
  "region": "ap-beijing",
  "notebook_user": "root",
  "hostname": "nb-1618225634846853120-cz56w4reusxs",
  "returned_tasks": 1,
  "request_id": "11f4ab3f-8fb8-4f28-8709-2a636be8294d",
  "temporary_probe_kernel_removed": true
}
```

另一次成功查询的 RequestId：`822e39f1-6f63-4f84-b24b-6831047fcc5a`。

判断本次调用成功，应同时满足：

- JSON 的 `ok` 为 `true`，`auth` 为 `STS`。
- `temporary_probe_kernel_removed` 为 `true`。
- 命令退出码为 `0`。

`returned_tasks` 为 `0` 也代表查询成功。它最多返回一条任务，`1` 不是账号内的任务总数。历史 RequestId 只用于追溯，后续运行会返回新值；其他 agent 应以自己的实际输出报告本次状态。

## 5. 常见失败与处理

| 输出或现象 | 含义与处理 |
| --- | --- |
| SSH 超时或 `Permission denied` | 检查原有 `tencent` SSH 配置、网络和已授权登录凭证；尚未到达 TI-ONE 认证。 |
| 工具沙箱阻止 SSH 网络访问 | 使用工具的正常网络审批流程，说明目标及只读用途；不要通过其他程序绕过审批。 |
| `LAUNCH_TOKEN_NOT_FOUND` | 当前没有读到已授权的 Jupyter 启动令牌；核对开发机是否运行、服务配置是否变化，不扫描其他账号的凭证文件。 |
| `NOTEBOOK_STS_ENTRY_MISSING` | Notebook 内核本身缺少平台 STS 入口，需要核对实例的临时身份配置；不能靠编造环境变量地址解决。 |
| `STS_FETCH_FAILED` | skill 没有成功取得 STS；检查平台凭证入口及服务端网络。探针会在自己的进程/内核中清除代理环境变量，不需要修改本机代理设置。 |
| `stage: notebook_interface` | Jupyter HTTP/WebSocket 或内核启动失败；按 `error_type` 检查服务可用性和已授权认证配置。 |
| 其他 `error_type` | 可能是 SDK、导入或 TI-ONE API 查询失败。排查时仅记录脱敏错误，不打印完整凭证、认证 URL 或请求头。 |
| `EXECUTION_TIMEOUT` | 本次执行未在等待期限内完成；先检查服务状态和临时内核清理结果。 |
| `ok: true`，但清理字段为 `false` | 查询成功，但本次临时内核清理失败；退出码仍为 `1`。只处理能确认属于该次探针的内核，不能批量停止其他会话。 |
| Python 导入错误且没有 JSON | 外层 `requests` / `websocket` 导入发生在结果捕获前，先检查远端 SSH Python 的依赖。 |

不要把“SSH 成功”“拿到 Jupyter 会话”或历史成功输出当作本次 TI-ONE API 验证成功。

## 6. 授权范围与复用边界

2026-11-06，用户明确授权：在 `43.137.68.228:11076` 上读取 `root` Jupyter 进程启动参数中的现有登录令牌，**仅在该服务器本机使用**，进入 Notebook 并运行指定 skill 做 TI-ONE 只读认证验证。用户随后要求将方法写入仓库供其他 agent 复用。

这项授权覆盖上述连接检查；不包含任意 root 操作、其他服务器、训练提交或停止、权限修改、跨用户分发凭证。继承同一任务及上述授权的 agent 可以直接复用入口；其他上下文仍应遵循当前用户授权和工具审批要求。

令牌、STS 地址和临时密钥只在服务器进程内存中使用，不输出、不保存、不复制到本机。脚本仅打印状态、主机名、任务数量和 RequestId。TI-ONE 操作是只读的；Jupyter 侧会创建并删除本次检查专用的临时内核。

如果自动审批再次拒绝操作，应说明具体拒绝理由和已有授权范围，按正常审批机制处理，不改用间接手段绕过。此文档记录连接方法与授权事实，不替代审批。

## 7. 后续 agent 的最短工作流程

1. 阅读本文，核对目标仍是上述开发机，确认任务继承的授权范围。
2. 运行 `scripts/Test-TioneSts.ps1`。
3. 检查退出码、`ok`、`auth` 和清理字段。
4. 用本次输出报告连接状态；如要继续执行训练等操作，先按实际任务授权及远端 skill 的对应流程处理。

相关文件：[PowerShell 入口](../scripts/Test-TioneSts.ps1) · [远端 Python 探针](../scripts/tione_notebook_sts_probe.py) · [项目进度](STATUS.md)
