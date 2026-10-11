param(
  [string]$Python='C:/Users/Li/Desktop/aic/TempSamp-R1/.venv-yolo/Scripts/python.exe',
  [string]$VideoDir='C:/Users/Li/Desktop/复赛-基于视频大模型的通用视频高光剪辑/video',
  [string]$Device='cuda'
)
$ErrorActionPreference='Stop'
Set-Location (Split-Path $PSScriptRoot -Parent)
& $Python scripts/download_transnet.py
if($LASTEXITCODE -ne 0){throw 'Weight download failed'}
& $Python -m unittest discover -s tests -v
if($LASTEXITCODE -ne 0){throw 'Tests failed'}
& $Python src/shots.py --video-dir $VideoDir --device $Device
if($LASTEXITCODE -ne 0){throw 'Shot detection failed'}
& $Python src/stage1.py
if($LASTEXITCODE -ne 0){throw 'Stage1 failed'}
