param(
    [string]$IsaacLabPath = "C:\IsaacLab",
    [double]$Speed = 0.06,
    [ValidateSet("virtual", "physical")][string]$GraspMode = "virtual",
    [string]$OutputDir = "",
    [switch]$SmokeTest
)
$ErrorActionPreference = "Stop"
$bundledPython = Join-Path $IsaacLabPath "_isaac_sim\python.bat"
$scriptPath = Join-Path $PSScriptRoot "teleop_cloth.py"
if (Test-Path -LiteralPath $bundledPython) {
    $launcher = $bundledPython
    $launchArgs = @("-X", "utf8", "-u", $scriptPath)
} else {
    $launcher = Join-Path $IsaacLabPath "isaaclab.bat"
    $launchArgs = @("-p", $scriptPath)
}
$launchArgs += @("--speed", $Speed, "--grasp_mode", $GraspMode)
if ($OutputDir) { $launchArgs += @("--output_dir", $OutputDir) }
if ($SmokeTest) {
    $launchArgs += @("--smoke_test", "--viz", "none")
} else {
    $launchArgs += @("--viz", "kit")
}
$launchArgs += "presets=isaacsim_physx"
& $launcher @launchArgs
if ($LASTEXITCODE -ne 0) { throw "Teleoperation failed (exit $LASTEXITCODE)." }
