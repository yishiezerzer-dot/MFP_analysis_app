param(
    [switch]$OneFile
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$specPath = Join-Path $root "MainMFP.spec"
if (-not (Test-Path $specPath)) {
    throw "Spec file not found: $specPath"
}

$args = @("-m", "PyInstaller", "--noconfirm")
if ($OneFile) {
    $args += "--clean"
}
$args += $specPath

& python @args

Write-Host "Build finished. Output is in dist/." -ForegroundColor Green