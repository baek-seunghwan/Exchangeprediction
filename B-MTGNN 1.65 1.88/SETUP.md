# B-MTGNN 환경 설정 가이드

## 필수 요구사항

- **Python**: 3.8 이상 (현재: Python 3.12.10 ✓)
- **Operating System**: Windows 10/11

## 필요한 패키지

| 패키지 | 버전 | 용도 |
|--------|------|------|
| torch | ≥2.0.0 | 딥러닝 프레임워크 (모델 구현) |
| numpy | ≥1.21.0 | 수치 계산 |
| pandas | ≥1.3.0 | 데이터 전처리, CSV 로드 |
| matplotlib | ≥3.5.0 | 그래프 시각화 |
| scipy | ≥1.7.0 | 과학 계산 (상관계수 등) |
| scikit-learn | ≥1.0.0 | 머신러닝 유틸리티 |

## 설치 방법

### 방법 1: 자동 설치 스크립트 (권장)

**CMD에서:**
```bash
cd C:\A\lab\exchange\B-MTGNN
install_requirements.bat
```

**PowerShell에서:**
```powershell
cd C:\A\lab\exchange\B-MTGNN
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\install_requirements.ps1
```

### 방법 2: 수동 설치 (한 줄 명령어)

```bash
pip install -r requirements.txt
```

### 방법 3: 개별 설치

```bash
# pip 업그레이드
python -m pip install --upgrade pip

# 패키지 설치 (CPU 버전 PyTorch)
pip install numpy pandas matplotlib scipy scikit-learn
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 또는 GPU 버전 PyTorch (CUDA 12.1)
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## 설치 확인

```bash
# Python 버전 확인
python --version

# 설치된 패키지 확인
pip list | findstr "torch numpy pandas matplotlib scipy"

# 또는 각 패키지 import 테스트
python -c "import torch; import numpy; import pandas; import matplotlib; print('모든 패키지 설치 완료!')"
```

## GPU 사용 (선택사항)

만약 NVIDIA GPU가 있다면:

```bash
# CUDA 12.1 버전
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 또는 CUDA 11.8 버전
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

설치 후 CUDA 확인:
```python
python -c "import torch; print('GPU available:', torch.cuda.is_available()); print('GPU device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 문제 해결

### pip 설치 실패
```bash
# pip 업그레이드
python -m pip install --upgrade pip

# 캐시 제거하고 재설치
pip install --no-cache-dir -r requirements.txt
```

### PyTorch 설치 느림
- 네트워크 속도에 따라 5~15분 소요
- 기다리거나 CPU 버전으로 임시 설치 후 나중에 GPU 버전으로 업그레이드

### matplotlib 에러
```bash
# 충돌하는 패키지 제거 후 재설치
pip uninstall matplotlib -y
pip install matplotlib>=3.5.0
```

## 다음 단계: 훈련 실행

설치 완료 후 다음 명령어로 모델 훈련:

```powershell
cd C:\A\lab\exchange\B-MTGNN\scripts
python -u .\train_test.py --layers 2 --dilation_exponential 2 --epochs 50 --dropout 0.1 2>&1 | Tee-Object -FilePath .\run.log
```

다른 터미널에서 실시간 로그 모니터링:
```powershell
cd C:\A\lab\exchange\B-MTGNN\scripts
Get-Content .\run.log -Tail 50 -Wait
```

---

**완료 후 메시지:**
```
모든 패키지 설치 완료!
이제 train_test.py를 실행할 준비가 되었습니다.
```
