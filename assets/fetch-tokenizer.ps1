<#
.SYNOPSIS
    Download the DeepSeek tokenizer that billion-context（hf-tokenizer feature） expects.

.PARAMETER Variant
    v4 (DeepSeek-V4-Pro) or v41 (DeepSeek-V4-Flash-Vision-Exp, the default).

.PARAMETER Force
    Overwrite an existing assets/tokenizer.json.
#>
[CmdletBinding()]
param(
    [ValidateSet('v4', 'v41')]
    [string]$Variant = 'v41',
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$dest = Join-Path $root 'tokenizer.json'

if ((Test-Path $dest) -and -not $Force) {
    Write-Host "already present: $dest ($([math]::Round((Get-Item $dest).Length / 1MB, 2)) MB)"
    Write-Host 'pass -Force to replace it'
    return
}

$url = "https://raw.githubusercontent.com/deepseek-ai/deepseek-recipe/main/static/tokenizers/$Variant/tokenizer.json"
Write-Host "downloading $Variant -> $dest"

$sw = [Diagnostics.Stopwatch]::StartNew()
Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing -TimeoutSec 300
$sw.Stop()

$size = (Get-Item $dest).Length
$sha = (Get-FileHash $dest -Algorithm SHA256).Hash
Write-Host ("saved {0:N2} MB in {1:N1}s" -f ($size / 1MB), $sw.Elapsed.TotalSeconds)
Write-Host "sha256 $sha"
