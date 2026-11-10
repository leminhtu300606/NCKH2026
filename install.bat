@echo off
chcp 65001 >nul
setlocal
echo ================================================================
echo  Cai dat dependencies cho he thong bao ve rieng tu video UAV (NCKH2026)
echo ================================================================

echo.
echo Chon loai cai dat:
echo   [1] CPU (mac dinh) - dung requirements.txt
echo   [2] CUDA (GPU NVIDIA) - dung requirements-cuda.txt
set /p CHON="Nhap 1 hoac 2 (mac dinh 1): "
if "%CHON%"=="2" goto :cuda
set REQ=requirements.txt
echo Da chon CPU.
goto :install_pip

:cuda
set REQ=requirements-cuda.txt
echo Da chon CUDA.

echo.
echo [1/4] Cap nhat pip...
python -m pip install --upgrade pip
if errorlevel 1 goto :error

echo.
echo [2/4] Go ban CPU/khac de tranh xung dot...
echo    - torch/torchvision ban CPU lam torch.cuda.is_available luon bang False
echo    - onnxruntime ban CPU dung chung thu muc voi onnxruntime-gpu
python -m pip uninstall -y torch torchvision onnxruntime onnxruntime-gpu

echo.
echo [3/4] Cai PyTorch CUDA (cu124)...
echo    Neu driver cu hon, doi cu124 thanh cu121 trong ca URL va ten goi.
python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.15.0+cu124 torchvision==0.29.2+cu124
if errorlevel 1 goto :error

echo.
echo [4/4] Cai dat cac thu vien tu %REQ%...
python -m pip install -r "%REQ%"
if errorlevel 1 goto :error
goto :verify_cuda

:install_pip
echo.
echo [1/3] Cap nhat pip...
python -m pip install --upgrade pip
if errorlevel 1 goto :error

echo.
echo [2/3] Cai dat cac thu vien tu %REQ%...
python -m pip install -r "%REQ%"
if errorlevel 1 goto :error

echo.
echo [3/3] Kiem tra cac thu vien can thiet...
python -c "import torch, cv2, numpy, pandas, ultralytics, onnxruntime, xgboost, psutil, sklearn; print('Tat ca thu vien da san sang.'); print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
if errorlevel 1 goto :error
goto :done

:verify_cuda
echo.
echo ================================================================
echo  Kiem tra CUDA
echo ================================================================
python -c "import torch, cv2, numpy, pandas, ultralytics, onnxruntime, xgboost, psutil, sklearn; print('Tat ca thu vien da san sang.'); print('torch:', torch.__version__, '| torch.version.cuda:', torch.version.cuda, '| CUDA available:', torch.cuda.is_available())"
if errorlevel 1 goto :error
python -c "import sys, torch; ok = torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0) if ok else 'N/A'); sys.exit(0 if ok else 1)"
if errorlevel 1 goto :cuda_error

:done
echo.
echo ================================================================
echo  Hoan tat. Co the chay: python step5_integrated_system.py
echo ================================================================
pause
exit /b 0

:cuda_error
echo.
echo ################################################################
echo  LOI: Da chon CUDA nhung PyTorch KHONG thay GPU.
echo  Kiem tra driver: nvidia-smi
echo  Kiem tra torch  : python -c "import torch; print(torch.version.cuda)"
echo  Neu ket qua la None, cai lai dung index CUDA khop driver, vi du:
echo    python -m pip install --index-url https://download.pytorch.org/whl/cu124 torch==2.15.0+cu124 torchvision==0.29.2+cu124
echo ################################################################
pause
exit /b 1

:error
echo.
echo ################################################################
echo  LOI: mot buoc cai dat that bai (xem thong bao phia tren).
echo ################################################################
pause
exit /b 1
