# B-MTGNN 환경 설정 스크립트 (PowerShell)

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "B-MTGNN 환경 설정 시작" -ForegroundColor Cyan
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[1/6] pip 업그레이드 중..." -ForegroundColor Yellow
python -m pip install --upgrade pip
Write-Host ""

Write-Host "[2/6] NumPy 설치 중..." -ForegroundColor Yellow
pip install numpy>=1.21.0
Write-Host ""

Write-Host "[3/6] PyTorch 설치 중..." -ForegroundColor Yellow
Write-Host "PyTorch 설치 (CPU 버전)" -ForegroundColor Green
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
Write-Host ""

Write-Host "[4/6] Pandas 설치 중..." -ForegroundColor Yellow
pip install pandas>=1.3.0
Write-Host ""

Write-Host "[5/6] Matplotlib 설치 중..." -ForegroundColor Yellow
pip install matplotlib>=3.5.0
Write-Host ""

Write-Host "[6/6] SciPy, Scikit-learn 설치 중..." -ForegroundColor Yellow
pip install scipy>=1.7.0 scikit-learn>=1.0.0
Write-Host ""

Write-Host "============================================" -ForegroundColor Cyan
Write-Host "설치 완료!" -ForegroundColor Green
Write-Host "============================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "[확인] 설치된 패키지:" -ForegroundColor Green
pip list | Select-String -Pattern "torch|numpy|pandas|matplotlib|scipy"
Write-Host ""

Write-Host "Press any key to continue..." -ForegroundColor Cyan
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
