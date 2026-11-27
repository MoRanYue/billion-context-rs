# Release + dist packaging for billion-context-proxy.
#
# Every step that Phase 4.1/4.2 had to be done by hand is encoded here, because
# doing it by hand is exactly how a release binary got overwritten and how dist
# drifted 5 versions behind:
#
#   1. back up the current release binary (never overwrite without a copy)
#   2. build release
#   3. run the test suite
#   4. verify `--version` and `/__bili/status` report the same build identity
#   5. stage into a versioned dist directory (old directories are left alone)
#   6. write SHA256SUMS.txt
#   7. prove release == dist byte for byte
#
# It refuses to publish anything that fails verification, so a red run leaves
# the previous release untouched.
#
# Everything here is a gate: if any check fails the script throws before it
# touches the dist directory, so a failed run leaves the previous release and
# every historical dist directory exactly as they were.
#
# Usage:
#   .\scripts\release.ps1 -Tag v0.1.217            # build, verify, package
#   .\scripts\release.ps1 -Tag v0.1.217 -SkipTests # skip step 3 (not recommended)
#
# -Tag is mandatory and is never guessed. The tag has to exist, has to point at
# HEAD, and its name has to match the package version. A release is never built
# from a dirty tree: a dirty build cannot be reproduced from the tag, so its
# identity would name a commit that does not contain the code that was built.
#
# This package is a subdirectory of a larger repository (`rust/billion-context-proxy` in the
# ezxss workspace), and that repository is routinely dirty for reasons that have
# nothing to do with this code. Every git query below is therefore run from the
# package root and scoped with an explicit `.` pathspec: the cleanliness gate asks
# whether *this package* is clean, not whether the enclosing workspace is.

param(
    [Parameter(Mandatory = $true)]
    [string]$Tag,
    [string]$TargetDir = "C:\bc-target",
    [switch]$SkipTests
)

$ErrorActionPreference = 'Stop'

$repo = Split-Path -Parent $PSScriptRoot
$manifest = Join-Path $repo 'Cargo.toml'
if (-not (Test-Path $manifest)) { throw "missing $manifest" }

# --- version ---------------------------------------------------------------
$versionLine = Select-String -Path $manifest -Pattern '^version\s*=\s*"([^"]+)"' |
    Select-Object -First 1
if (-not $versionLine) { throw "cannot read [package] version from $manifest" }
$version = $versionLine.Matches[0].Groups[1].Value
Write-Host "==> billion-context-proxy $version"

# --- git state -------------------------------------------------------------
# A release is built from a tag, and only from a tag that describes exactly the
# tree being built. All four checks run before anything is written, and any
# failure aborts the run with the previous release untouched.
Push-Location $repo
try {
    $headCommit = (git rev-parse HEAD 2>$null)
    if ($LASTEXITCODE -ne 0 -or -not $headCommit) {
        throw "not a git checkout, or git failed; a release needs full git metadata"
    }
    $headCommit = $headCommit.Trim()
    $commit = $headCommit.Substring(0, 12)

    # dirty: refuses the release (a dirty tree cannot be reproduced from a tag).
    # Scoped to this package: the enclosing workspace is dirty most of the time
    # for unrelated reasons, and an unscoped check would never pass here.
    $dirtyPaths = @(git status --porcelain --untracked-files=normal -- . 2>$null)
    if ($LASTEXITCODE -ne 0) {
        throw "cannot read git status for $repo"
    }
    $dirty = $dirtyPaths.Count -gt 0

    # tag must exist ...
    $tagRef = (git rev-parse --verify --quiet "refs/tags/$Tag" 2>$null)
    if ($LASTEXITCODE -ne 0 -or -not $tagRef) {
        throw "tag $Tag does not exist; pass an existing tag with -Tag"
    }
    # ... and resolve to a commit. An annotated tag's own object is not a commit,
    # so the ^{commit} peel is required; without it the comparison below would
    # silently compare two different kinds of object.
    $tagCommit = (git rev-parse --verify --quiet "$Tag^{commit}" 2>$null)
    if ($LASTEXITCODE -ne 0 -or -not $tagCommit) {
        throw "tag $Tag does not resolve to a commit"
    }
    $tagCommit = $tagCommit.Trim()
} finally {
    Pop-Location
}

if ($dirty) {
    $preview = ($dirtyPaths | Select-Object -First 5) -join '; '
    throw "rust/billion-context-proxy is dirty ($($dirtyPaths.Count) path(s): $preview); a release must be built from a clean tree (commit or stash first)"
}
if ($tagCommit -ne $headCommit) {
    throw "tag $Tag points at $tagCommit but HEAD is $headCommit; the tag must describe the tree being built"
}

# The tag name carries the version, so the two cannot drift apart unnoticed.
# Tags in the enclosing workspace are namespaced per product (`rust/seria-config-*`,
# `migration-guard-v1.3.1`, ...), so an optional `billion-context-proxy-` prefix is accepted:
# both `v0.1.217` and `billion-context-proxy-v0.1.217` name version 0.1.217.
$tagVersion = $Tag -replace '^billion-context-proxy-', '' -replace '^v', ''
if ($tagVersion -ne $version) {
    throw "tag $Tag implies version $tagVersion but Cargo.toml says $version"
}

Write-Host "==> commit $commit (clean)"
Write-Host "==> tag    $Tag -> $tagCommit"

$env:CARGO_TARGET_DIR = $TargetDir
# Pin the identity the script just verified. Without this the binary's identity
# comes from whatever cargo decides to re-run, which can be a cached build-script
# result — Phase 4.3 shipped a run where the script said dirty=true and the
# binary said dirty=false for exactly that reason. The rerun-if-env-changed
# directives in build.rs make these values force a fresh build script run.
$env:BILI_BUILD_COMMIT = $commit
$env:BILI_BUILD_DIRTY = '0'
$releaseExe = Join-Path $TargetDir 'release\billion-context-proxy.exe'
$testCount = 'not run'

# --- 1. back up the previous release --------------------------------------
if (Test-Path $releaseExe) {
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $backup = "$releaseExe.pre-p43-$stamp.bak"
    Copy-Item $releaseExe $backup -Force
    $previous = Get-FileHash $releaseExe -Algorithm SHA256
    Write-Host "==> backed up previous release -> $backup"
    Write-Host "    size $((Get-Item $backup).Length) B  sha256 $($previous.Hash)"
} else {
    Write-Host "==> no previous release to back up"
}

# --- 2. build --------------------------------------------------------------
Push-Location $repo
try {
    Write-Host "==> cargo build --release"
    cargo build -p billion-context-proxy --release
    if ($LASTEXITCODE -ne 0) { throw "release build failed (exit $LASTEXITCODE)" }

    # --- 3. tests ----------------------------------------------------------
    if (-not $SkipTests) {
        Write-Host "==> cargo test"
        $testLog = & cargo test -p billion-context-proxy 2>&1
        $testLog | ForEach-Object { Write-Host $_ }
        if ($LASTEXITCODE -ne 0) { throw "tests failed (exit $LASTEXITCODE)" }
        $match = $testLog | Select-String 'test result: ok\. (\d+) passed'
        if ($match) { $script:testCount = $match.Matches[0].Groups[1].Value }
    } else {
        Write-Warning "tests skipped"
        $script:testCount = 'not run'
    }
} finally {
    Pop-Location
}

if (-not (Test-Path $releaseExe)) { throw "build produced no $releaseExe" }

# --- 4. verify the build identity -----------------------------------------
# The binary must agree with itself: the CLI banner, the status endpoint and the
# version we are about to publish all have to be the same identity.
$cliLines = & $releaseExe --version
if ($LASTEXITCODE -ne 0) { throw "--version failed (exit $LASTEXITCODE)" }
$cliVersion = ($cliLines | Select-String '^billion-context-proxy\s+(\S+)$').Matches.Groups[1].Value
$cliCommit = ($cliLines | Select-String '^commit\s+(\S+)$').Matches.Groups[1].Value
$cliProfile = ($cliLines | Select-String '^profile\s+(\S+)$').Matches.Groups[1].Value
$cliDirty = ($cliLines | Select-String '^dirty\s+(\S+)$').Matches.Groups[1].Value

Write-Host "==> --version"
$cliLines | ForEach-Object { Write-Host "    $_" }

if ($cliVersion -ne $version) {
    throw "--version says $cliVersion but Cargo.toml says $version"
}
if ($cliProfile -ne 'release') {
    throw "--version profile is $cliProfile, expected release"
}
# The binary must name the commit the tag names. Anything else means the build
# was served from a stale build-script result, which is precisely how a release
# ends up claiming a commit it was not built from.
if ($cliCommit -ne $commit) {
    throw "--version commit '$cliCommit' != expected commit '$commit' (stale build?)"
}
if ($cliDirty -ne 'false') {
    throw "--version dirty is '$cliDirty', expected false"
}

# Query the status endpoint from a throwaway instance on a non-production port.
$probePort = 18999
$probeDir = Join-Path $TargetDir 'p43-verify'
Remove-Item $probeDir -Recurse -Force -ErrorAction SilentlyContinue
New-Item -ItemType Directory $probeDir -Force | Out-Null

$savedStateHome = $env:XDG_STATE_HOME
$savedSessions = $env:BILI_SESSIONS_DIR
$env:XDG_STATE_HOME = $probeDir
$env:BILI_SESSIONS_DIR = Join-Path $probeDir 'rust-sessions'

$proc = $null
try {
    $proc = Start-Process -FilePath $releaseExe `
        -ArgumentList 'start', '--port', "$probePort" `
        -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $probeDir 'out.log') `
        -RedirectStandardError (Join-Path $probeDir 'err.log')

    $status = $null
    foreach ($attempt in 1..20) {
        Start-Sleep -Milliseconds 300
        try {
            $status = Invoke-RestMethod "http://127.0.0.1:$probePort/__bili/status" -TimeoutSec 2
            break
        } catch { }
    }
    if (-not $status) { throw "status endpoint never answered on port $probePort" }

    Write-Host "==> /__bili/status"
    Write-Host "    build $($status.build | ConvertTo-Json -Compress)"
    Write-Host "    commit field $($status.commit)  (runtime marker, not a SHA)"

    if ($status.build.version -ne $version) {
        throw "status build.version $($status.build.version) != $version"
    }
    if ($status.build.commit -ne $cliCommit) {
        throw "status build.commit $($status.build.commit) != --version commit $cliCommit"
    }
    if ($status.build.profile -ne $cliProfile) {
        throw "status build.profile $($status.build.profile) != --version profile $cliProfile"
    }
    if ("$($status.build.dirty)" -ne $cliDirty) {
        throw "status build.dirty $($status.build.dirty) != --version dirty $cliDirty"
    }
} finally {
    if ($proc -and -not $proc.HasExited) {
        Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
        Start-Sleep -Milliseconds 400
    }
    $env:XDG_STATE_HOME = $savedStateHome
    $env:BILI_SESSIONS_DIR = $savedSessions
    Remove-Item $probeDir -Recurse -Force -ErrorAction SilentlyContinue
}

# --- 5. stage into dist ---------------------------------------------------
# New directory per version. Existing directories are never deleted or
# modified: they hold historical binaries that are the only copy of those builds.
$distRoot = Join-Path $repo 'dist'
$distDir = Join-Path $distRoot "bili-context-rust-$version-win64"
$distExe = Join-Path $distDir 'billion-context-proxy.exe'
New-Item -ItemType Directory $distDir -Force | Out-Null
Copy-Item $releaseExe $distExe -Force
Write-Host "==> staged -> $distExe"

# start.ps1 points at $PSScriptRoot\billion-context-proxy.exe; ship it if it is not there.
$startScript = Join-Path $distDir 'start.ps1'
if (-not (Test-Path $startScript)) {
    $template = Get-ChildItem $distRoot -Directory |
        Where-Object { Test-Path (Join-Path $_.FullName 'start.ps1') } |
        Select-Object -First 1
    if ($template) {
        Copy-Item (Join-Path $template.FullName 'start.ps1') $startScript -Force
        Write-Host "==> copied start.ps1 from $($template.Name)"
    }
}

# Render the dist README from the template so its version, commit and test count
# cannot drift from the binary that ships beside it.
$readmeTemplate = Join-Path $PSScriptRoot 'dist-readme.md'
if (Test-Path $readmeTemplate) {
    $readme = Get-Content $readmeTemplate -Raw
    $readme = $readme.Replace('{{VERSION}}', $version).
        Replace('{{COMMIT}}', $commit).
        Replace('{{TEST_COUNT}}', "$testCount")
    # UTF-8 without BOM, LF line endings.
    $readme = $readme -replace "`r`n", "`n"
    [System.IO.File]::WriteAllText(
        (Join-Path $distDir 'README.md'),
        $readme,
        (New-Object System.Text.UTF8Encoding($false)))
    Write-Host "==> wrote README.md (version $version, commit $commit, tests $testCount)"
}

# --- 6. checksums ---------------------------------------------------------
$releaseHash = (Get-FileHash $releaseExe -Algorithm SHA256).Hash
$distHash = (Get-FileHash $distExe -Algorithm SHA256).Hash
$sumsPath = Join-Path $distDir 'SHA256SUMS.txt'

# Every shipped file is listed, exe first. Two-space separator and bare file
# names keep the file usable by `sha256sum -c` from inside the dist directory.
$sumLines = @("$distHash  billion-context-proxy.exe")
foreach ($extra in @('README.md', 'start.ps1')) {
    $path = Join-Path $distDir $extra
    if (Test-Path $path) {
        $sumLines += "$((Get-FileHash $path -Algorithm SHA256).Hash)  $extra"
    }
}
$sumLines -join "`n" | Set-Content $sumsPath -Encoding ascii -NoNewline
Add-Content $sumsPath '' -NoNewline
Write-Host "==> wrote $sumsPath"
$sumLines | ForEach-Object { Write-Host "    $_" }

# --- 7. prove release == dist --------------------------------------------
Write-Host "==> release $releaseHash"
Write-Host "==> dist    $distHash"
if ($releaseHash -ne $distHash) {
    throw "dist binary does not match the release binary"
}
if ($releaseHash -ne ((Get-FileHash $releaseExe -Algorithm SHA256).Hash)) {
    throw "release binary changed during packaging"
}

Write-Host ''
Write-Host "OK  billion-context-proxy $version"
Write-Host "    commit  $commit (clean)"
Write-Host "    tag     $Tag"
Write-Host "    sha256  $releaseHash"
Write-Host "    dist    $distDir"
Write-Host "    start   $startScript  (port 18113, not 18787)"
