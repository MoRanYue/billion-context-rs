param(
    [int]$Port = 18120,
    [string]$ShimRoot = $env:BILI_SHIM_ROOT
)
# Acceptance probe for the shim's spawn path. The installed DSH native shim
# resolves BILLION_CONTEXT_NODE without probing and spawns:
#   <nodeRuntime> <proxy-script> start --host 127.0.0.1 --port <n>
# Pointing BILLION_CONTEXT_NODE at billion-context-proxy.exe therefore makes the shim spawn
# the Rust host. This script executes that exact argv shape.
$ErrorActionPreference = 'Stop'
if (-not $ShimRoot -or -not (Test-Path $ShimRoot)) { throw 'set BILI_SHIM_ROOT to the installed billion-context package directory' }
$root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
$exe = Join-Path $root 'target\debug\billion-context-proxy.exe'
if (-not (Test-Path $exe)) { throw "build first: cargo build ($exe missing)" }
$proxyScript = Join-Path $ShimRoot 'dist\index.js'
if (-not (Test-Path $proxyScript)) { throw "installed package has no dist/index.js: $proxyScript" }
$err = Join-Path $env:TEMP "bili-shim-spawn-$Port.err"
$out = Join-Path $env:TEMP "bili-shim-spawn-$Port.out"
Remove-Item $err, $out -ErrorAction SilentlyContinue
$proc = Start-Process -FilePath $exe -ArgumentList @(
    $proxyScript, 'start', '--host', '127.0.0.1', '--port', "$Port"
) -WorkingDirectory $root -WindowStyle Hidden -PassThru `
  -RedirectStandardError $err -RedirectStandardOutput $out
try {
    Start-Sleep -Seconds 2
    $info = Invoke-RestMethod "http://127.0.0.1:$Port/__bili/plugin/runtime-info"
    if ($info.runtime -ne 'rust') { throw "spawned host is not the Rust runtime: $($info.runtime)" }
    $ignored = Select-String -Path $err -Pattern 'ignoring unrecognized argument' -Quiet
    if (-not $ignored) { throw 'Rust host did not tolerate the shim script argument' }
    "ok runtime=$($info.runtime) version=$($info.version) port=$Port"
} finally {
    Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
    Remove-Item $err, $out -ErrorAction SilentlyContinue
}
