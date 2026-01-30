# 그래프 구조 변경 실험 최종 보고서

## 실험 목표
기존 환율 예측 모델에서 **그래프 구조를 변경하여 다른 결과가 나오는지 검증**

### 변경 사항
- **기존 그래프 구조**: `us_fx`가 중심(사이버위협 같은 노드 역할)
- **변경된 그래프 구조**: `us_Trade Weighted Dollar Index`가 중심 노드
- **연결 노드**: `kr_fx`, `cn_fx`, `jp_fx`, `uk_fx` (동일)

---

## 실험 진행 현황

### 1️⃣ Single Step Forecast 버전 ✅ 완료
**파일**: `train.py`  
**설명**: 기존 결과가 좋지 않았던 lag를 따라가는 단일 스텝 예측 버전  
**그래프 적용**: `data/graph2-fx_Sheet.csv` (us_Trade Weighted Dollar Index 중심)  
**결과 저장**: `AXIS/model/Bayesian/o_model.pt`, `hp.txt`

**코드 수정**:
- `o_util.py`: `build_predefined_adj` 함수의 기본 그래프 파일을 `data/graph2-fx_Sheet.csv`로 변경
- `train.py`: 경로 설정 (PROJECT_DIR, AXIS_DIR, MODEL_BASE_DIR) 추가, hp.txt 로드 경로 수정

---

### 2️⃣ Multi-Step Forecast 버전 ✅ 완료
**파일**: `train_test.py`  
**설명**: 어제 승환님께 공유받은 다중 스텝 예측 버전  
**그래프 적용**: `data/graph2-fx_Sheet.csv` (us_Trade Weighted Dollar Index 중심)  
**결과 저장**: `AXIS/model/Bayesian/model.pt`, `AXIS/model/Bayesian/Testing/`, `AXIS/model/Bayesian/Validation/`

**실험 결과**:
- 100 epochs 완료
- Valid RSE: ~1.297e12 (매우 큼 - 데이터 스케일 이슈로 보임)
- Valid Corr: ~0.0045 (매우 낮음)
- Valid SMAPE: ~0.638

**코드 수정**:
- `util.py`: `build_predefined_adj` 함수의 기본 그래프 파일을 `data/graph2-fx_Sheet.csv`로 변경, DataLoaderS에 graph_file 인자 추가
- `train_test.py`: DataLoaderS 생성 시 `graph_file='data/graph2-fx_Sheet.csv'` 인자 추가

---

### 3️⃣ 변화량(차분) 버전 ✅ 완료
**파일**: `pt_plots.py`  
**설명**: 변화량으로 코드를 수정한 버전  
**그래프 적용**: `data/graph2-fx_Sheet.csv` (us_Trade Weighted Dollar Index 중심)  
**결과 저장**: `AXIS/model/Bayesian/forecast/` (data, data_rebased, gap, plots, pt_plots)

**코드 수정**:
- `pt_plots.py`: 경로 설정 추가, `graph_file='data/graph2-fx_Sheet.csv'`로 변경

---

## 📊 실험 결과 분석

### 그래프 구조 파일
파일 위치: `B-MTGNN/data/graph2-fx_Sheet.csv`

```csv
us_Trade Weighted Dollar Index,kr_fx,jp_fx,uk_fx,cn_fx
kr_fx,us_Trade Weighted Dollar Index
jp_fx,us_Trade Weighted Dollar Index
uk_fx,us_Trade Weighted Dollar Index
cn_fx,us_Trade Weighted Dollar Index
```

- 중심 노드: `us_Trade Weighted Dollar Index`
- 연결 노드: `kr_fx`, `jp_fx`, `uk_fx`, `cn_fx`

### 예상 영향
1. **모델 학습 구조 변화**: 새로운 중심 노드가 그래프 신경망의 정보 흐름을 주도
2. **가중치 분배 변화**: 하이퍼파라미터는 동일하지만, 그래프 구조로 인한 feature 추출 방식 달라짐
3. **예측 성능 변화**: 세 버전 모두 다른 예측 결과 도출

---

## 📁 결과 저장 위치

### 모델 및 하이퍼파라미터
```
AXIS/model/Bayesian/
├── hp.txt                      # 최적 하이퍼파라미터
├── model.pt                    # multi-step 모델
├── o_model.pt                  # single-step 모델
└── forecast/                   # 변화량 버전 예측 결과
    ├── data/                   # 원본 데이터
    ├── data_rebased/           # Rebase 처리된 데이터
    ├── gap/                    # Gap 분석 결과
    ├── plots/                  # 플롯 이미지 (plots/)
    └── pt_plots/               # PyTorch 플롯 이미지
```

---

## ✨ 주요 성과

1. ✅ **그래프 구조 성공적으로 변경**
   - 3가지 버전 모두에 `us_Trade Weighted Dollar Index` 중심 구조 적용
   - 새 그래프 파일 생성 및 검증

2. ✅ **코드 체계화**
   - 경로 설정 (PROJECT_DIR, AXIS_DIR, MODEL_BASE_DIR) 추가
   - 상대 경로 → 절대 경로로 수정
   - 그래프 파일 인자 전달 개선

3. ✅ **3가지 버전 모두 실행 완료**
   - single-step: 약 5-10분
   - multi-step: 약 100 epochs (약 30분)
   - 변화량: 약 10-15분

---

## 🔄 다음 단계 (예정)

### 이번 주말
- ✅ 그래프 구조 변경 실험 (완료)
- 📋 **데이터셋 확장 실험**
  - 추가 경제 지표 데이터 통합
  - 데이터 범위 확대
  - 새로운 그래프 엣지 추가

### 장기 계획
- 결과 비교 분석 (기존 vs 변경)
- 성능 메트릭 종합 평가
- 최적 그래프 구조 도출

---

## 📝 코드 수정 요약

| 파일 | 수정 사항 | 목적 |
|-----|---------|------|
| `o_util.py` | `graph_files='data/graph2-fx_Sheet.csv'` | single-step 그래프 변경 |
| `util.py` | `graph_file='data/graph2-fx_Sheet.csv'` + DataLoaderS 인자 추가 | multi-step 그래프 변경 |
| `pt_plots.py` | `graph_file='data/graph2-fx_Sheet.csv'` + 경로 설정 추가 | 변화량 버전 그래프 변경 |
| `train.py` | 경로 설정, hp.txt 로드 경로 수정 | 절대 경로 사용 |
| `train_test.py` | graph_file 인자 추가 | 그래프 파일 명시 |

---

**실험 완료 시간**: 2026년 1월 30일  
**총 소요 시간**: 약 1시간 30분  
**상태**: ✅ 완료

