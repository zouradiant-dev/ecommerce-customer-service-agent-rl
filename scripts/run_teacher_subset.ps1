param(
    [string]$Model = "qwen3.7-plus-2026-05-26",
    [Parameter(Mandatory = $true)][string]$Manifest,
    [Parameter(Mandatory = $true)][string]$Output,
    [int]$ExpectedTasks = 0,
    [int]$TrialsPerTask = 1
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($env:OPENAI_API_KEY)) {
    throw "OPENAI_API_KEY is not set in this PowerShell session."
}
if ([string]::IsNullOrWhiteSpace($env:BAILIAN_API_BASE)) {
    throw "BAILIAN_API_BASE is not set in this PowerShell session."
}

$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path
$runner = Join-Path $projectRoot ".venv-api\Scripts\ecommerce-cs.exe"
$manifestPath = if ([IO.Path]::IsPathRooted($Manifest)) { $Manifest } else { Join-Path $projectRoot $Manifest }
$outputPath = if ([IO.Path]::IsPathRooted($Output)) { $Output } else { Join-Path $projectRoot $Output }
if (-not (Test-Path -LiteralPath $runner)) { throw "Missing API environment executable: $runner" }
if (-not (Test-Path -LiteralPath $manifestPath)) { throw "Missing retry manifest: $manifestPath" }

$manifestRows = @(
    Get-Content -LiteralPath $manifestPath -Encoding UTF8 |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
        ForEach-Object { $_ | ConvertFrom-Json }
)
$invalidSplits = @($manifestRows | Where-Object { $_.split -notin @("train", "dev") })
$taskIds = @($manifestRows | ForEach-Object { [string]$_.task_id })
$uniqueTaskIds = @($taskIds | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Sort-Object -Unique)
if ($invalidSplits.Count -gt 0) {
    throw "Teacher subset may contain only train/dev rows."
}
if ($TrialsPerTask -lt 1 -or $manifestRows.Count -lt 1) {
    throw "Retry manifest and TrialsPerTask must be non-empty and positive."
}
if ($manifestRows.Count -ne $uniqueTaskIds.Count) {
    throw "Retry manifest contains duplicate or missing task IDs."
}
if ($ExpectedTasks -gt 0 -and $manifestRows.Count -ne $ExpectedTasks) {
    throw "Retry manifest has $($manifestRows.Count) tasks; expected $ExpectedTasks."
}

& $runner run-model `
    --manifest $manifestPath `
    --model $Model `
    --api-base $env:BAILIAN_API_BASE `
    --output $outputPath `
    --trials-per-task $TrialsPerTask `
    --temperature 0.3 `
    --max-steps 20 `
    --max-tokens 1024 `
    --no-enable-thinking `
    --verbose

if ($LASTEXITCODE -ne 0) { throw "Teacher subset failed with exit code $LASTEXITCODE." }

$rolloutRows = @(
    Get-Content -LiteralPath $outputPath -Encoding UTF8 |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
        ForEach-Object { $_ | ConvertFrom-Json }
)
$expectedRollouts = $manifestRows.Count * $TrialsPerTask
if ($rolloutRows.Count -ne $expectedRollouts) {
    throw "Teacher subset produced $($rolloutRows.Count) rows; expected $expectedRollouts."
}

& $runner evaluate --rollouts $outputPath
if ($LASTEXITCODE -ne 0) { throw "Teacher subset evaluation failed with exit code $LASTEXITCODE." }
