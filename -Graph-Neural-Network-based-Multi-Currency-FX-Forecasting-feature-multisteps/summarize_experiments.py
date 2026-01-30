#!/usr/bin/env python3
"""
그래프 구조 변경 실험 결과 정리 및 비교 스크립트
- 기존 그래프: us_fx 중심 노드
- 변경 그래프: us_Trade Weighted Dollar Index 중심 노드
"""

import os
import json
from pathlib import Path
import sys

# 프로젝트 경로 설정
PROJECT_DIR = Path(__file__).resolve().parent
AXIS_DIR = PROJECT_DIR / 'AXIS'
MODEL_BASE_DIR = AXIS_DIR / 'model' / 'Bayesian'

print("\n" + "="*80)
print("그래프 구조 변경 실험 결과 정리")
print("="*80)

print("\n[실험 개요]")
print("- 기존 그래프 구조: us_fx가 중심(사이버위협) 노드")
print("- 변경 그래프 구조: us_Trade Weighted Dollar Index가 중심 노드")
print("- 적용 대상 버전 3가지:")
print("  1. single step forecast 버전 (train.py, o_util.py)")
print("  2. multi-step forecast 버전 (train_test.py, util.py)")
print("  3. 변화량(차분) 버전 (pt_plots.py)")

print("\n[코드 수정 사항]")
print("1. o_util.py:")
print("   - build_predefined_adj 함수의 기본 그래프 파일: data/graph2-fx_Sheet.csv")
print("\n2. util.py:")
print("   - build_predefined_adj 함수의 기본 그래프 파일: data/graph2-fx_Sheet.csv")
print("   - DataLoaderS에 graph_file 인자 추가")
print("\n3. pt_plots.py:")
print("   - graph_file: 'data/graph2-fx_Sheet.csv'로 변경")
print("\n4. train.py:")
print("   - 경로 설정 (PROJECT_DIR, AXIS_DIR, MODEL_BASE_DIR) 추가")
print("   - hp.txt와 모델 저장 경로를 절대 경로로 수정")
print("   - graph_file='data/graph2-fx_Sheet.csv' 사용")
print("\n5. train_test.py:")
print("   - DataLoaderS 생성 시 graph_file='data/graph2-fx_Sheet.csv' 인자 추가")

print("\n[실험 결과 위치]")
model_dir = MODEL_BASE_DIR
forecast_dir = model_dir / 'forecast'

if model_dir.exists():
    print(f"✓ 모델 디렉토리: {model_dir}")
    if (model_dir / 'hp.txt').exists():
        print(f"  - hp.txt: 하이퍼파라미터 저장됨")
    if (model_dir / 'model.pt').exists():
        print(f"  - model.pt: multi-step 모델")
    if (model_dir / 'o_model.pt').exists():
        print(f"  - o_model.pt: single-step 모델")
else:
    print(f"✗ 모델 디렉토리 없음: {model_dir}")

if forecast_dir.exists():
    print(f"\n✓ 예측 결과 디렉토리: {forecast_dir}")
    subdirs = ['data', 'data_rebased', 'gap', 'plots', 'pt_plots']
    for subdir in subdirs:
        subpath = forecast_dir / subdir
        if subpath.exists():
            file_count = len(list(subpath.glob('*')))
            print(f"  - {subdir}/: {file_count}개 파일")
else:
    print(f"\n예측 결과 디렉토리 준비 중...")

print("\n[그래프 구조 파일]")
graph_file = PROJECT_DIR / 'B-MTGNN' / 'data' / 'graph2-fx_Sheet.csv'
if graph_file.exists():
    with open(graph_file, 'r') as f:
        lines = f.readlines()
    print(f"✓ {graph_file}")
    print(f"  - 첫 번째 노드: {lines[0].strip().split(',')[0]} (중심)")
    print(f"  - 연결 노드: {', '.join(lines[0].strip().split(',')[1:5])}")
    print(f"  - 총 {len(lines)} 줄")
else:
    print(f"✗ 그래프 파일 없음: {graph_file}")

print("\n[예상 결과]")
print("1. us_Trade Weighted Dollar Index를 중심 노드로 사용했으므로,")
print("   기존의 us_fx 중심 구조와 다른 학습 및 예측 결과를 얻을 것으로 예상됩니다.")
print("\n2. 세 가지 버전 모두 동일한 그래프 구조를 적용했으므로,")
print("   버전 간 성능 비교가 의미 있을 것입니다.")
print("\n3. 결과는 다음 위치에 저장됩니다:")
print(f"   - {model_dir}")
print(f"   - {forecast_dir}")

print("\n[다음 단계]")
print("1. 각 버전의 성능 메트릭 비교")
print("2. 예측 결과의 정확도 평가 (RSE, RAE, CORR, SMAPE)")
print("3. 그래프 구조 변경이 미친 영향 분석")
print("4. 데이터셋 확장 실험 진행 (예정)")

print("\n" + "="*80)
print("실험 진행 상황:")
print("="*80)

# 실행 상태 확인
print("\n현재 실행 중인 프로세스:")
import subprocess
try:
    result = subprocess.run(['pgrep', '-f', 'pt_plots.py'], 
                          capture_output=True, text=True)
    if result.stdout.strip():
        print("✓ pt_plots.py (변화량 버전) 실행 중...")
    else:
        print("✓ pt_plots.py 완료")
except:
    print("  프로세스 확인 불가")

print("\n실험 완료 예상 시간: 약 10-20분")
print("결과 확인 후 데이터셋 확장 실험을 진행할 예정입니다.")
print("\n" + "="*80)
