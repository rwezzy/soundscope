param(
    [switch]$UseMeta,
    [switch]$Record,
    [switch]$Fullscreen,
    [string]$EnvFile = '',
    [int]$Seconds = 0
)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
# Keep each recording take in its own history. Old experiments are never deleted.
$take = Get-Date -Format 'yyyyMMdd_HHmmss'
$env:SOUNDSCOPE_MEMORY = Join-Path $PSScriptRoot "recordings\takes\$take\memory"
$env:META_SESSION_LIMIT_USD = '0'
$env:META_DAILY_LIMIT_USD = '0'
$launchArgs = @('-B', 'scope.py', '--mount', 'carrier_v3', '--palette', 'amber', '--heatmap-style', 'classic', '--heatmap-opacity', '0.44')
if ($UseMeta) {
    $env:META_SESSION_LIMIT_USD = '0.10'
    $env:META_DAILY_LIMIT_USD = '1.00'
    $launchArgs += '--meta'
}
if ($EnvFile) { $launchArgs += @('--env-file', $EnvFile) }
if ($Fullscreen) { $launchArgs += '--fullscreen' }
if ($Seconds -gt 0) { $launchArgs += @('--seconds', "$Seconds") }
if ($Record) {
    $videoDir = Join-Path $PSScriptRoot 'recordings\demo'
    New-Item -ItemType Directory -Path $videoDir -Force | Out-Null
    $outputVideo = Join-Path $videoDir "soundscope_$take.mp4"
    $launchArgs += @('--record', $outputVideo)
    Write-Host "Silent screen video: $outputVideo"
}
Write-Host 'Standalone only. No arm connection.'
Write-Host 'Use the on-screen icon buttons for Voice, Analyze, Search, History, Calibrate, Heatmap and Finish.'
Write-Host 'Keyboard alternatives are in Help. Use -UseMeta to enable automatic transcripts and sound labels.'
Write-Host "This take's sound history: $env:SOUNDSCOPE_MEMORY"
$python = (Get-Command python -ErrorAction Stop).Source
& $python @launchArgs
exit $LASTEXITCODE
