param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]] $UvArgs
)

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$UvExe = Join-Path $ProjectRoot "tools\uv\uv.exe"

if (-not (Test-Path -LiteralPath $UvExe)) {
    throw "Project-local uv was not found at $UvExe"
}

$env:UV_PYTHON_INSTALL_DIR = Join-Path $ProjectRoot "tools\python"
$env:UV_PYTHON_BIN_DIR = Join-Path $ProjectRoot "tools\python-bin"
$env:UV_CACHE_DIR = Join-Path $ProjectRoot "tools\cache"
$env:UV_LINK_MODE = "copy"

& $UvExe @UvArgs
exit $LASTEXITCODE

