@echo off
chcp 65001 >nul
echo ================================================================
echo  Cai dat dependencies cho he thong bao ve rip tu video UAV (NCKH2026)
echo ================================================================

echo.
echo Chon loai cai dat:
echo   [1] CPU (mac dinh) - dung requirements.txt
echo   [2] CUDA (GPU NVIDIA) - dung requirements-cuda.txt
set /p CHON="Nhap 1 hoac 2 (mac dinh 1): "
if "%CHON%"=="2" (
    set REQ=requirements-cuda.txt
    echo Da chon CUDA.
) else (
    set REQ=requirements.txt
    echo Da chon CPU.
)

echo.
echo [1/3] Cap nhat pip...
python -m pip install --upgrade pip

echo.
echo [2/3] Cai dat cac thu vien tu %REQ%...
pip install -r "%REQ%"

echo.
echo [3/3] Kiem tra cac thu vien can thiet...
python -c "import torch, cv2, numpy, pandas, ultralytics, onnxruntime, xgboost, psutil, sklearn; print('Tat ca thu vien da san sang.')"

echo.
echo ================================================================
echo  Hoan tat. Co chay: python step5_integrated_system.py
echo ================================================================
pause
