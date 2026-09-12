param(
    [int]$RunStart = 11000,
    [int]$RunLimit = 10000,
    [int]$BatchPages = 500,
    [int]$RequestBudget = 40000,
    [int]$RequiredHealthyChecks = 3,
    [int]$HealthyCheckIntervalSeconds = 20,
    [int]$OutageCheckIntervalSeconds = 60,
    [double]$MinimumKeyHeadroomUsd = 12.0,
    [string]$PythonExe = 'python',
    [string]$KeyPath = $env:OPENROUTER_KEY_FILE,
    [string]$InputDir = $env:PIPELINE_IMAGE_DIR
)

$ErrorActionPreference = 'Stop'
$PipelineRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$KeyPath = if ($KeyPath) { $KeyPath } else { Join-Path $PipelineRoot 'openrouter_key.txt' }
$InputDir = if ($InputDir) { $InputDir } else { Join-Path $PipelineRoot 'data\images' }
$StatusPath = Join-Path $PipelineRoot 'output\production\full_pages_joined_v1\run_status.json'
$RunArguments = @(
    'run.py', 'run-all',
    '--input-dir', $InputDir,
    '--run-name', 'full_pages_joined_v1',
    '--processing-order', (Join-Path $PipelineRoot 'data\full_pages_main_evaluation_600_priority_after_11000.json'),
    '--start', $RunStart.ToString(),
    '--limit', $RunLimit.ToString(),
    '--batch-pages', $BatchPages.ToString(),
    '--page-workers', '24',
    '--entity-workers', '40',
    '--request-budget', $RequestBudget.ToString(),
    '--target-tokens-per-minute', '850000',
    '--recovery-passes', '1',
    '--variant', 's2',
    '--style', 'direct',
    '--gaze', 'yes',
    '--crop-cache', 'none'
)

function Write-SupervisorMessage([string]$Message) {
    $stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss K'
    Write-Host "[$stamp] SUPERVISOR: $Message"
}

function Wait-ForStableOpenRouter {
    $healthy = 0
    while ($healthy -lt $RequiredHealthyChecks) {
        try {
            $apiKey = (Get-Content -LiteralPath $KeyPath -Raw).Trim()
            $headers = @{ Authorization = "Bearer $apiKey" }
            $response = Invoke-RestMethod -Uri 'https://openrouter.ai/api/v1/key' -Method Get -Headers $headers -TimeoutSec 15
            $remaining = $response.data.limit_remaining
            $hasHeadroom = ($null -eq $response.data.limit) -or ($null -ne $remaining -and [double]$remaining -ge $MinimumKeyHeadroomUsd)
            if ($hasHeadroom) {
                $healthy++
                $headroomText = if ($null -eq $response.data.limit) { 'unlimited' } else { ('$' + ([double]$remaining).ToString('0.00')) }
                Write-SupervisorMessage "authenticated readiness check $healthy/$RequiredHealthyChecks succeeded; key headroom $headroomText"
                if ($healthy -lt $RequiredHealthyChecks) {
                    Start-Sleep -Seconds $HealthyCheckIntervalSeconds
                }
                continue
            }
            $remainingText = if ($null -eq $remaining) { 'unknown' } else { ('$' + ([double]$remaining).ToString('0.00')) }
            Write-SupervisorMessage "key headroom $remainingText is below required `$$($MinimumKeyHeadroomUsd.ToString('0.00')); retrying in $OutageCheckIntervalSeconds seconds"
        }
        catch {
            Write-SupervisorMessage "OpenRouter/key check unavailable; retrying in $OutageCheckIntervalSeconds seconds ($($_.Exception.Message))"
        }
        $healthy = 0
        Start-Sleep -Seconds $OutageCheckIntervalSeconds
    }
}

function Test-TrancheComplete {
    if (-not (Test-Path -LiteralPath $StatusPath)) {
        return $false
    }
    try {
        $status = Get-Content -LiteralPath $StatusPath -Raw | ConvertFrom-Json
        return (
            $status.selected_range.start -eq $RunStart -and
            $status.selected_range.count -eq $RunLimit -and
            ($status.all_selected_complete -eq $true -or $status.supervisor_terminal_state -eq $true)
        )
    }
    catch {
        return $false
    }
}

Set-Location -LiteralPath $PipelineRoot
while (-not (Test-TrancheComplete)) {
    Wait-ForStableOpenRouter
    Write-SupervisorMessage "stable connectivity detected; starting/resuming configured tranche $RunStart..$($RunStart + $RunLimit - 1)"
    & $PythonExe @RunArguments
    $runExitCode = $LASTEXITCODE
    if (Test-TrancheComplete) {
        $status = Get-Content -LiteralPath $StatusPath -Raw | ConvertFrom-Json
        if ($status.all_selected_complete -eq $true) {
            Write-SupervisorMessage 'configured tranche is fully complete'
        }
        else {
            Write-SupervisorMessage "configured tranche reached a documented terminal state; incomplete pages=$($status.terminal_unrecovered.page_count)"
        }
        exit 0
    }
    Write-SupervisorMessage "runner exited with code $runExitCode before full coverage; returning to connectivity gate"
}

Write-SupervisorMessage 'configured tranche was already complete or terminally audited'
exit 0
