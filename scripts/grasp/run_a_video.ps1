param(
    [string]$IsaacLabPath = "C:\IsaacLab"
)

$ErrorActionPreference = "Stop"
$launcher = Join-Path $IsaacLabPath "isaaclab.bat"
if (-not (Test-Path -LiteralPath $launcher)) {
    throw "Isaac Lab launcher not found: $launcher"
}
& $launcher -p (Join-Path $PSScriptRoot "grasp_stability_demo.py") --video --picker_backend attachment --viz kit presets=isaacsim_physx
if ($LASTEXITCODE -ne 0) {
    throw "Simulation or video recording failed (exit $LASTEXITCODE)."
}
