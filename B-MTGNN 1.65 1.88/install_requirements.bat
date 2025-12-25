@echo off
REM 필요한 Python 패키지 설치 스크립트

echo ============================================
echo B-MTGNN 환경 설정 시작
echo ============================================
echo.

echo [1/6] pip 업그레이드 중...
python -m pip install --upgrade pip
echo.

echo [2/6] NumPy 설치 중...
pip install numpy>=1.21.0
echo.

echo [3/6] PyTorch 설치 중...
echo PyTorch 설치 (CPU 버전)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
echo.

echo [4/6] Pandas 설치 중...
pip install pandas>=1.3.0
echo.

echo [5/6] Matplotlib 설치 중...
pip install matplotlib>=3.5.0
echo.

echo [6/6] SciPy, Scikit-learn 설치 중...
pip install scipy>=1.7.0 scikit-learn>=1.0.0
echo.

echo ============================================
echo 설치 완료!
echo ============================================
echo.

echo [확인] 설치된 패키지:
pip list | findstr -E "torch|numpy|pandas|matplotlib|scipy"
echo.

pause
