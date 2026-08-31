[CmdletBinding()]
param(
    [ValidateSet('3.11', '3.12', '3.13')]
    [string]$PythonVersion = '3.12'
)

$ErrorActionPreference = 'Stop'
$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$venvPath = Join-Path $repoRoot '.venv'
$venvPython = Join-Path $venvPath 'Scripts\python.exe'

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw 'The Python launcher (py.exe) is required.'
}

if (-not (Test-Path -LiteralPath $venvPython)) {
    & py "-$PythonVersion" -m venv $venvPath
    if ($LASTEXITCODE -ne 0) { throw "Could not create a Python $PythonVersion environment." }
}

$actualPython = & $venvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($actualPython.Trim() -ne $PythonVersion) {
    throw ".venv uses Python $actualPython but Python $PythonVersion was requested. Remove or rename that exact .venv, then rerun."
}

& $venvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) { throw 'pip bootstrap failed.' }

# Remove the legacy second CUDA stack if repairing an earlier partial install.
# PaddleOCR's Transformers engine shares one PyTorch/CUDA runtime with Qwen.
$legacyCudaPackages = @(
    'paddlepaddle-gpu',
    'nvidia-cuda-runtime-cu12',
    'nvidia-cudnn-cu12',
    'nvidia-cublas-cu12',
    'nvidia-cufft-cu12',
    'nvidia-curand-cu12',
    'nvidia-cusolver-cu12',
    'nvidia-cusparse-cu12',
    'nvidia-nvjitlink-cu12'
)
& $venvPython -m pip uninstall --yes @legacyCudaPackages | Out-Host
if ($LASTEXITCODE -ne 0) { throw 'Legacy CUDA runtime cleanup failed.' }

# Current stable CUDA 13.0 PyTorch includes Blackwell/sm_120 kernels.
& $venvPython -m pip install torch==2.13.0+cu130 torchvision==0.28.0+cu130 --index-url https://download.pytorch.org/whl/cu130
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }

& $venvPython -m pip install --editable $repoRoot
if ($LASTEXITCODE -ne 0) { throw 'Application dependency installation failed.' }

& $venvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Installed dependency set is inconsistent.' }

# Execute a real Blackwell CUDA kernel before model initialization.
& $venvPython -c "import torch; assert torch.cuda.is_available(); t=torch.ones(1024,device='cuda').sum(); torch.cuda.synchronize(); print('torch',torch.__version__,torch.cuda.get_device_name(0),t.item())"
if ($LASTEXITCODE -ne 0) { throw 'GPU runtime verification failed.' }

# Mandatory known-text inference catches Windows/Blackwell builds that load
# CUDA successfully but silently return no OCR boxes.
& $venvPython -c "from anime_trivia_automation.config import load_config; from anime_trivia_automation.ocr import PaddleOCREngine; c=load_config(r'$repoRoot\config.example.json'); PaddleOCREngine(c.ocr); print('PaddleOCR smoke passed')"
if ($LASTEXITCODE -ne 0) { throw 'PaddleOCR known-text smoke failed.' }

Write-Host "Installed successfully. Next: copy config.example.json to config.json, calibrate the region, then run scripts\warm_models.py."
