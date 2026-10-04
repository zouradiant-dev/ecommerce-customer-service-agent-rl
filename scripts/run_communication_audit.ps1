param(
    [switch]$Execute,
    [decimal]$BudgetYuan = 5,
    [int]$Limit = 80,
    [string]$ApiBase = $env:BAILIAN_API_BASE,
    [decimal]$InputPrice = 2.5,
    [decimal]$OutputPrice = 10
)
$ErrorActionPreference = 'Stop'
$taskRepo = Split-Path -Parent $PSScriptRoot
$taskWorkspace = Split-Path -Parent $taskRepo
$taskPython = Join-Path $taskWorkspace '.venv\Scripts\python.exe'
$taskJob = Join-Path $taskRepo 'outputs\communication-audit-v1\prepared'
$taskOutput = Join-Path $taskRepo ('outputs\communication-audit-v1\api-' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
$taskArguments = @(
    (Join-Path $taskRepo 'scripts\audit_communication.py'), 'run',
    '--job', $taskJob, '--output', $taskOutput,
    '--limit', $Limit, '--budget-yuan', $BudgetYuan.ToString([Globalization.CultureInfo]::InvariantCulture),
    '--input-price', $InputPrice.ToString([Globalization.CultureInfo]::InvariantCulture),
    '--output-price', $OutputPrice.ToString([Globalization.CultureInfo]::InvariantCulture)
)
if ($Execute) {
    if (-not $ApiBase) { throw 'Set BAILIAN_API_BASE or pass -ApiBase before enabling paid calls.' }
    if ($ApiBase.TrimEnd('/') -ne 'https://dashscope.aliyuncs.com/compatible-mode/v1' -and
        (-not $PSBoundParameters.ContainsKey('InputPrice') -or -not $PSBoundParameters.ContainsKey('OutputPrice'))) {
        throw 'Default prices are for Beijing only. Confirm and explicitly pass both prices for another region.'
    }
    $taskArguments += @('--execute', '--api-base', $ApiBase)
}
& $taskPython @taskArguments
if ($LASTEXITCODE -ne 0) { throw "Communication audit failed with exit code $LASTEXITCODE" }
