param(
    [string]$IsaacLabPath = "C:\IsaacLab",
    [double]$ClothX = 0.50,
    [double]$ClothY = -0.05,
    [double]$ClothYawDeg = 0
)
$ErrorActionPreference = "Stop"
& (Join-Path $IsaacLabPath "isaaclab.bat") -p (Join-Path $PSScriptRoot "rectangular_fold_demo.py") --cloth_x $ClothX --cloth_y $ClothY --cloth_yaw_deg $ClothYawDeg --viz kit presets=isaacsim_physx
if ($LASTEXITCODE -ne 0) { throw "Simulation failed (exit $LASTEXITCODE)." }
