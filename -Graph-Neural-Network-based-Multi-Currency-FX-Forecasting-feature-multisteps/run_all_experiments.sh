#!/bin/bash

# 새 그래프 구조로 모든 버전의 실험 자동 실행 스크립트
# graph2-fx_Sheet.csv 사용 (us_Trade Weighted Dollar Index 중심)

cd /Users/samrobert/Documents/GitHub/Exchangeprediction/-Graph-Neural-Network-based-Multi-Currency-FX-Forecasting-feature-multisteps

source .venv/bin/activate

echo "=========================================="
echo "그래프 구조 변경 실험 시작"
echo "기존: us_fx 중심"
echo "변경: us_Trade Weighted Dollar Index 중심"
echo "=========================================="
echo ""

# 1. Multi-step 버전 (train_test.py) - 이미 완료됨
echo "[1/3] Multi-step forecast 버전 (train_test.py) - 이미 완료"
echo "결과: AXIS/model/Bayesian/model.pt, hp.txt 저장됨"
echo ""

# 2. Single step 버전 (train.py) 실행 중 확인
echo "[2/3] Single step forecast 버전 (train.py) 실행 상태 확인..."
if pgrep -f "python B-MTGNN/train.py" > /dev/null; then
    echo "현재 실행 중... 완료될 때까지 기다리는 중..."
    wait
    echo "✓ Single step 버전 완료"
else
    echo "실행되지 않음. train.py 다시 실행..."
    python B-MTGNN/train.py
    echo "✓ Single step 버전 완료"
fi
echo ""

# 3. 변화량 버전 (pt_plots.py) 실행
echo "[3/3] 변화량/차분 버전 (pt_plots.py) 실행..."
python B-MTGNN/pt_plots.py
echo "✓ 변화량 버전 완료"
echo ""

echo "=========================================="
echo "✓ 모든 실험 완료!"
echo "=========================================="
echo ""
echo "결과 위치:"
echo "  - Multi-step: AXIS/model/Bayesian/"
echo "  - Single-step: AXIS/model/Bayesian/o_model.pt"
echo "  - 변화량: AXIS/model/Bayesian/forecast/"
echo ""
