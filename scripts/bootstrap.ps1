$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot

Push-Location $ProjectRoot
try {
    $ManagedPython = Get-ChildItem -LiteralPath "$ProjectRoot\tools\python" -Directory -ErrorAction SilentlyContinue |
        Where-Object Name -Like "cpython-3.12*-windows-x86_64-none" |
        Sort-Object Name -Descending |
        ForEach-Object { Join-Path $_.FullName "python.exe" } |
        Where-Object { Test-Path -LiteralPath $_ } |
        Select-Object -First 1

    if (-not $ManagedPython) {
        & "$PSScriptRoot\uv.ps1" python install 3.12 --no-bin --no-registry
        $ManagedPython = Get-ChildItem -LiteralPath "$ProjectRoot\tools\python" -Directory -ErrorAction SilentlyContinue |
            Where-Object Name -Like "cpython-3.12*-windows-x86_64-none" |
            Sort-Object Name -Descending |
            ForEach-Object { Join-Path $_.FullName "python.exe" } |
            Where-Object { Test-Path -LiteralPath $_ } |
            Select-Object -First 1
    }

    if (-not $ManagedPython) {
        throw "Python 3.12 was not found in the project-local tools directory."
    }

    & "$PSScriptRoot\uv.ps1" sync --python $ManagedPython
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

    & "$PSScriptRoot\uv.ps1" run fremont doctor
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}

