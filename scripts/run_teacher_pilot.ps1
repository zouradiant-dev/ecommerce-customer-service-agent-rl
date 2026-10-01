param(
    [string]$Model = "qwen3.7-plus-2026-05-26",
    [string]$Manifest = "data\qwen15b-v2\pilot-train-dev.jsonl",
    [string]$Output = "outputs\teacher-qwen37\pilot-one-per-template.jsonl",
    [int]$ExpectedTasks = 22,
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
if (-not (Test-Path -LiteralPath $runner)) {
    throw "Missing API environment executable: $runner"
}
if (-not (Test-Path -LiteralPath $manifestPath)) {
    throw "Missing pilot manifest: $manifestPath"
}

$manifestRows = @(
    Get-Content -LiteralPath $manifestPath -Encoding UTF8 |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
        ForEach-Object { $_ | ConvertFrom-Json }
)
$invalidSplits = @($manifestRows | Where-Object { $_.split -notin @("train", "dev") })
$templateIds = @($manifestRows | ForEach-Object { $_.metadata.template_id })
$uniqueTemplateIds = @($templateIds | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Sort-Object -Unique)
if ($invalidSplits.Count -gt 0) {
    $foundSplits = ($invalidSplits.split | Sort-Object -Unique) -join ", "
    throw "Teacher pilot may contain only train/dev rows; found: $foundSplits"
}
if ($ExpectedTasks -lt 1 -or $TrialsPerTask -lt 1) {
    throw "ExpectedTasks and TrialsPerTask must be positive."
}
if ($manifestRows.Count -ne $ExpectedTasks -or $uniqueTemplateIds.Count -ne $ExpectedTasks) {
    throw "Teacher pilot must contain exactly $ExpectedTasks rows from $ExpectedTasks unique templates."
}

& $runner run-model `
    --manifest $manifestPath `
    --model $Model `
    --api-base $env:BAILIAN_API_BASE `
    --output $outputPath `
    --trials-per-task $TrialsPerTask `
    --temperature 0.3 `
    --max-tokens 1024 `
    --no-enable-thinking `
    --verbose

if ($LASTEXITCODE -ne 0) {
    throw "Teacher pilot failed with exit code $LASTEXITCODE."
}

if (-not (Test-Path -LiteralPath $outputPath)) {
    throw "Teacher pilot did not create an output file: $outputPath"
}
$rolloutRows = @(
    Get-Content -LiteralPath $outputPath -Encoding UTF8 |
        Where-Object { -not [string]::IsNullOrWhiteSpace($_) } |
        ForEach-Object { $_ | ConvertFrom-Json }
)
$workerErrors = @($rolloutRows | Where-Object { $_.stopped_reason -eq "worker_error" -or $_.error })
$expectedRollouts = $ExpectedTasks * $TrialsPerTask
if ($rolloutRows.Count -ne $expectedRollouts) {
    throw "Teacher pilot produced $($rolloutRows.Count) rows; expected $expectedRollouts."
}
if ($workerErrors.Count -gt 0) {
    $examples = ($workerErrors | Select-Object -First 3 | ForEach-Object { "$($_.task_id): $($_.error)" }) -join "; "
    throw "Teacher pilot contains $($workerErrors.Count) worker errors. Examples: $examples"
}

& $runner evaluate --rollouts $outputPath
if ($LASTEXITCODE -ne 0) {
    throw "Teacher pilot evaluation failed with exit code $LASTEXITCODE."
}
