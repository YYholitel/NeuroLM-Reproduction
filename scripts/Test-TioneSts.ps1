[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
$probePath = Join-Path $PSScriptRoot 'tione_notebook_sts_probe.py'
Get-Content -LiteralPath $probePath -Raw | & ssh -o BatchMode=yes -o ConnectTimeout=10 tencent 'python -'
exit $LASTEXITCODE
