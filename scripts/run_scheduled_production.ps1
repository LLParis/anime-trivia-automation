[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$runtimeDir = Join-Path $repoRoot "runtime"
$worker = Join-Path $repoRoot ".venv\Scripts\anime-trivia.exe"
$config = Join-Path $repoRoot "config.json"

if (-not (Test-Path -LiteralPath $worker -PathType Leaf)) {
    throw "Production worker executable is missing."
}
if (-not (Test-Path -LiteralPath $config -PathType Leaf)) {
    throw "Production configuration is missing."
}

New-Item -ItemType Directory -Path $runtimeDir -Force | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$stdoutPath = Join-Path $runtimeDir "scheduled-$stamp.out.log"
$stderrPath = Join-Path $runtimeDir "scheduled-$stamp.err.log"

Push-Location $repoRoot
try {
    & $worker --config $config 1>> $stdoutPath 2>> $stderrPath
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
