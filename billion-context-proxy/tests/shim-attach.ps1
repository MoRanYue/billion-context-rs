param(
    [int]$Port = 18116,
    [string]$ShimRoot = $env:BILI_SHIM_ROOT
)
$ErrorActionPreference = 'Stop'
if (-not $ShimRoot -or -not (Test-Path $ShimRoot)) { throw 'set BILI_SHIM_ROOT to the installed billion-context package directory' }
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$exe = Join-Path $root 'target\debug\billion-context-proxy.exe'
if (-not (Test-Path $exe)) { throw "build first: cargo build ($exe missing)" }
$proc = Start-Process -FilePath $exe -ArgumentList 'start','--port',"$Port" -WorkingDirectory $root -WindowStyle Hidden -PassThru
try {
    Start-Sleep -Seconds 2
    $env:BILI_SHIM_ROOT = $ShimRoot
    node (Join-Path $PSScriptRoot 'shim-attach.mjs') "http://127.0.0.1:$Port"
    if ($LASTEXITCODE -ne 0) { throw "shim attach probe failed ($LASTEXITCODE)" }
} finally {
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
}
