# download_models.ps1 — fetch MediaPipe BlazeFace + Face Mesh ONNX weights
$ErrorActionPreference = 'Stop'
$ProgressPreference    = 'SilentlyContinue'   # progress bar makes Invoke-WebRequest very slow on PS 5.1
$base = 'https://github.com/yakhyo/mediapipe-face-mesh-onnx/releases/download/weights'
$dir  = Join-Path $PSScriptRoot 'assets/models'
New-Item -ItemType Directory -Force -Path $dir | Out-Null

$sw = [Diagnostics.Stopwatch]::StartNew()
foreach ($f in @('face_detection_short_range.onnx', 'face_mesh_Nx3x192x192.onnx')) {
    $dest = Join-Path $dir $f
    if ((Test-Path $dest) -and ((Get-Item $dest).Length -gt 100KB)) { Write-Host "$f already present."; continue }
    Write-Host "Downloading $f ..."
    Invoke-WebRequest -Uri "$base/$f" -OutFile $dest -UseBasicParsing -TimeoutSec 90
    if ((Get-Item $dest).Length -lt 100KB) { Remove-Item $dest; throw "$f looks truncated" }
}
Write-Host ("Done in {0:n1}s" -f $sw.Elapsed.TotalSeconds)
if ($sw.Elapsed.TotalSeconds -gt 120) { Write-Warning 'Took longer than 2 minutes — check your connection.' }
