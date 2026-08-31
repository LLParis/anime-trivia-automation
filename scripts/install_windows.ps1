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

# Current stable CUDA 13.0 PyTorch includes Blackwell/sm_120 kernels.
& $venvPython -m pip install torch==2.13.0+cu130 torchvision==0.28.0+cu130 --index-url https://download.pytorch.org/whl/cu130
if ($LASTEXITCODE -ne 0) { throw 'CUDA PyTorch installation failed.' }

# Paddle's official CUDA 12.9 index currently supplies the Windows GPU wheel.
& $venvPython -m pip install paddlepaddle-gpu==3.3.1 --index-url https://www.paddlepaddle.org.cn/packages/stable/cu129/
if ($LASTEXITCODE -ne 0) { throw 'PaddlePaddle GPU installation failed.' }

& $venvPython -m pip install --editable $repoRoot
if ($LASTEXITCODE -ne 0) { throw 'Application dependency installation failed.' }

& $venvPython -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Installed dependency set is inconsistent.' }

# Mirror application import order and execute real kernels in both runtimes.
& $venvPython -c "import torch; assert torch.cuda.is_available(); t=torch.ones(1024,device='cuda').sum(); torch.cuda.synchronize(); print('torch',torch.__version__,torch.cuda.get_device_name(0),t.item()); import paddle; assert paddle.is_compiled_with_cuda() and paddle.device.cuda.device_count()>0; paddle.set_device('gpu:0'); p=paddle.ones([1024]).sum(); paddle.device.cuda.synchronize(); print('paddle',paddle.__version__,'gpus=',paddle.device.cuda.device_count(),'sum=',float(p)); paddle.utils.run_check()"
if ($LASTEXITCODE -ne 0) { throw 'GPU runtime verification failed.' }

# Mandatory known-text inference catches Windows/Blackwell builds that load
# CUDA successfully but silently return no OCR boxes.
& $venvPython -c "from anime_trivia_automation.config import load_config; from anime_trivia_automation.ocr import PaddleOCREngine; c=load_config(r'$repoRoot\config.example.json'); PaddleOCREngine(c.ocr); print('PaddleOCR smoke passed')"
if ($LASTEXITCODE -ne 0) { throw 'PaddleOCR known-text smoke failed.' }

Write-Host "Installed successfully. Next: copy config.example.json to config.json, calibrate the region, then run scripts\warm_models.py."
