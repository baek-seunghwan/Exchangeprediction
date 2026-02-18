# Cyber Trend Forecasting - 프로젝트 리뷰 및 가이드

이 저장소는 논문 "Forecasting Cyber Threats and Pertinent Mitigation Technologies"의 베이지안 MTGNN(B-MTGNN) 모델 구현입니다.  
시계열 그래프 데이터를 입력받아 각 노드(사이버 공격/기술)의 트렌드를 다변량·그래프 기반으로 예측합니다.  
이 문서는 원 프로젝트를 이해하고, 이를 주가(또는 재무지표) 예측으로 재가공하는 방법을 쉽고 자세하게 설명합니다.

## 📋 목차
1. [프로젝트 개요](#프로젝트-개요)
2. [핵심 요약](#핵심-요약)
3. [전체 워크플로우](#전체-워크플로우)
4. [디렉터리/파일 구조 상세 설명](#디렉터리파일-구조-상세-설명)
5. [모델 아키텍처 상세 설명](#모델-아키텍처-상세-설명)
6. [평가 지표 설명](#평가-지표-설명)
7. [설치 및 실행](#설치-및-실행)
8. [데이터 준비 가이드](#데이터-준비-가이드)
9. [비교 평가 실험 상세](#비교-평가-실험-상세)
10. [주가 예측으로 재가공하기](#주가-예측으로-재가공하기)

---

## 프로젝트 개요

### 논문 정보
- **제목**: Forecasting Cyber Threats and Pertinent Mitigation Technologies
- **저널**: Technological Forecasting and Social Change (2025)
- **DOI**: https://doi.org/10.1016/j.techfore.2024.123836

### 연구 목적
사이버 공격 트렌드와 관련 완화 기술(PT: Pertinent Technologies)의 미래 동향을 예측하여, 공격과 방어 기술 간의 격차(Gap)를 분석하고 대응 전략을 수립하는 것

### 데이터 기간
- **학습 기간**: 2011년 7월 ~ 2022년 12월 (월별 데이터)
- **예측 기간**: 최대 3년(36개월) 미래 예측

### 그래프 구조
- **노드 수**: 총 142개
  - 공격 노드: 26개 (급증하는 사이버 공격 유형)
  - 기술 노드: 98개 (관련 완화 기술)
  - 보조 특징 노드: 18개 (사건 수, 언급량, 트윗, 공휴일 등)
- **그래프 타입**: TPT 그래프 (Threats and Pertinent Technologies Graph)
  - 공격 노드와 관련 기술 노드 간의 연결 관계

---

## 핵심 요약

### 모델 아키텍처
- **기반 모델**: MTGNN (Multi-variate Time series Graph Neural Network)
  - Temporal CNN (Dilated Convolution + Inception 모듈)
  - Graph Convolution (Mixprop 기반)
- **베이지안 확장**: Monte Carlo Dropout
  - 드롭아웃을 활성화한 상태로 여러 번 추론하여 불확실성 정량화
  - 기본 반복 횟수: 10회 (최적은 30회로 확인됨)

### 입출력 형태
- **입력**: `[batch, in_dim(=1), num_nodes, seq_in_len]`
  - 예: 과거 10개월 데이터 (전체 142개 노드 동시)
- **출력**: `[batch, out_dim(=seq_out_len), num_nodes, 1]`
  - 예: 미래 36개월 예측 (전체 142개 노드 동시)
  - 평균, 분산, 95% 신뢰구간 제공

### 그래프 사용
- **정적 그래프**: `data/graph.csv`의 노드 간 연결로 사전 정의된 인접 행렬 구성
- **적응형 그래프**: 학습 중 데이터 기반으로 동적 인접 행렬 생성
- **결합**: 두 그래프를 혼합하여 사용

### 학습/예측 파이프라인
1. **데이터 정규화/배치화** (`util.py`/`o_util.py`)
   - 열별 최대값 기준 정규화
   - 시계열 슬라이딩 윈도우로 배치 생성
2. **하이퍼파라미터 탐색** (`train_test.py`)
   - 랜덤 서치로 최적 하이퍼파라미터 탐색
   - 검증/테스트 세트로 평가
3. **최종 학습** (`train.py`)
   - 전체 데이터로 최종 모델 학습
   - `o_model.pt` 저장
4. **베이지안 예측** (`forecast.py`)
   - 드롭아웃 활성화 상태로 여러 번 추론
   - 평균/분산/95% 신뢰구간 산출
   - 결과 저장 및 시각화

### 평가 지표
- **RRSE** (Root Relative Squared Error): 상대 제곱근 오차
- **RAE** (Relative Absolute Error): 상대 절대 오차
- **상관계수** (Correlation Coefficient)
- **sMAPE** (Symmetric Mean Absolute Percentage Error)

---

## 전체 워크플로우

```
1. 데이터 수집 및 전처리 (Data_Preparation/)
   ├─ Hackmageddon_Attacks: 공격 사건 수집 (NoI)
   ├─ Elsevier_Attacks: 논문에서 공격 언급량 수집 (A_NoM)
   ├─ Elsevier_PTs: 논문에서 기술 언급량 수집 (PT_NoM)
   ├─ Twitter_Tweets: 전쟁/분쟁 관련 트윗 수집 (ACA)
   └─ Python_Holidays: 공휴일 데이터 생성 (PH)
   ↓
2. 그래프 구성 (PT_Extractor/)
   ├─ E_GPT.py: 논문에서 GPT로 기술 추출
   └─ D_GPT.py: GPT에 직접 질문하여 기술 추출
   ↓
3. 데이터셋 통합 (Dataset/)
   └─ CT-0711-1222.csv: 모든 특징을 통합한 최종 데이터셋
   ↓
4. 모델 학습 및 예측 (B-MTGNN/)
   ├─ train_test.py: 하이퍼파라미터 최적화
   ├─ train.py: 최종 모델 학습
   └─ forecast.py: 미래 예측 및 시각화
   ↓
5. 비교 평가 (Comparative_Evaluation/)
   ├─ Baselines: ARIMA, VAR, LSTM, Transformer
   ├─ MTGNN: 베이지안 확장 전 모델
   └─ BMTGNN: 베이지안 확장 모델 (반복 횟수별 평가)
```

---

## 디렉터리/파일 구조 상세 설명

### `B-MTGNN/` - 메인 모델 구현

#### 핵심 파일
- **`net.py`**: 모델 본체 `gtnet` 클래스
  - `__init__`: 모델 초기화 (레이어 구성, 그래프 생성기 등)
  - `forward`: 순전파 (입력 → Temporal CNN → Graph Convolution → 출력)
  - 주요 구성 요소:
    - `dilated_inception`: 확장된 Inception 모듈 (다양한 커널 크기)
    - `graph_constructor`: 적응형 인접 행렬 생성
    - `mixprop`: 그래프 확산 기반 GCN 레이어
    - `skip connection`: 잔차 연결로 깊은 네트워크 학습 안정화

- **`layer.py`**: 그래프/합성곱/정규화 레이어 구현
  - `nconv`, `dy_nconv`: 그래프 컨볼루션 레이어
  - `prop`, `mixprop`: 그래프 확산 레이어
  - `dilated_1D`, `dilated_inception`: 시간적 컨볼루션 레이어
  - `graph_constructor`: 적응형 그래프 생성
  - `LayerNorm`: 레이어 정규화

- **`util.py`, `o_util.py`**: 데이터 로더 및 유틸리티
  - `DataLoaderS`: 시계열 데이터 로더
    - 탭 구분 텍스트(`.txt`) 읽기
    - 전역 최대값 기준 정규화 (열별 max)
    - 슬라이딩 윈도우로 배치 생성
  - `build_predefined_adj()`: 정적 인접 행렬 생성
    - `data/graph.csv`와 `data/sm_data_g.csv` 컬럼명 매칭
  - `create_columns()`: 컬럼명 생성 및 매핑
  - 평가 지표 계산 함수들

- **`train_test.py`**: 하이퍼파라미터 최적화 및 평가
  - 랜덤 서치로 하이퍼파라미터 탐색
  - 검증/테스트 세트로 모델 평가
  - 결과 저장 (플롯, 지표, 모델 체크포인트)
  - 출력: `model/Bayesian/hp.txt` (최적 하이퍼파라미터)

- **`train.py`**: 최종 모델 학습
  - `hp.txt`를 읽어 최적 하이퍼파라미터 로드
  - 전체 데이터로 최종 학습
  - 출력: `model/Bayesian/o_model.pt` (운용 모델)

- **`forecast.py`**: 미래 예측 및 시각화
  - `o_model.pt` 로드
  - 베이지안 추론 (기본 10회 반복, 드롭아웃 활성화)
  - 평균/분산/95% 신뢰구간 계산
  - 결과 저장:
    - `forecast/data/*.txt`: 과거/예측/신뢰구간/분산
    - `forecast/plots/*.png, *.pdf`: 공격 vs 기술 그룹 플롯
    - `forecast/gap/*_gap.csv`: 공격-기술 연평균 격차

- **`pt_plots.py`**: 개별 기술별 시각화
  - 각 기술(PT)별 과거/미래 데이터 플롯
  - 출력: `forecast/pt_plots/*.png, *.pdf`

- **`smoothing.py`**: 데이터 스무딩
  - 더블 지수평활 적용
  - `data/data.txt` → `data/sm_data.txt` 변환

- **`trainer.py`**: 학습 루프 및 최적화
  - 옵티마이저 설정 (Adam)
  - 학습률 스케줄링
  - 손실 함수 계산

#### 데이터 파일 (`data/`)
- **`data.txt`**: 원시 시계열 데이터 (탭 구분, 행=시점, 열=노드)
- **`sm_data.txt`**: 스무딩/전처리된 시계열 (실제 학습 입력)
- **`data.csv`**: 컬럼(노드) 이름 목록
- **`sm_data_g.csv`**: 그래프용 컬럼명 (인접 행렬 매핑 기준)
- **`graph.csv`**: 노드 간 연결 정의
  - 형식: 첫 컬럼 = 기준 노드, 이후 컬럼 = 연결 노드들
  - 예시:
    ```
    Password Attack,Multi-Factor Authentication,Biometric Authentication
    Ransomware,Backup,Encryption
    ```

#### 모델 저장 (`model/Bayesian/`)
- **`hp.txt`**: 최적 하이퍼파라미터
- **`model.pt`**: 검증 세트에서 최고 성능 모델
- **`o_model.pt`**: 최종 운용 모델 (전체 데이터 학습)
- **`Validation/`**: 검증 세트 예측 결과
  - 플롯: `*_Validation.png`, `*_Validation.pdf`
  - 지표: `evaluation.txt`
- **`Testing/`**: 테스트 세트 예측 결과
- **`forecast/`**: 미래 예측 결과
  - `data/`: 예측 데이터 (텍스트)
  - `plots/`: 공격-기술 그룹 플롯
  - `gap/`: 격차 분석 결과 (CSV)
  - `pt_plots/`: 개별 기술 플롯

---

### `Comparative_Evaluation/` - 비교 평가 실험

이 폴더는 B-MTGNN 모델의 성능을 다양한 베이스라인 모델과 비교하기 위한 실험 코드를 포함합니다.

#### `Baselines/` - 베이스라인 모델들

##### `ARIMA/`
- **목적**: 전통적인 시계열 예측 모델
- **특징**: 단변량 모델 (각 노드별로 독립적인 모델 학습)
- **하이퍼파라미터**: (p, d, q) 파라미터 최적화
- **평가**: 142개 노드 각각에 대해 36개월 예측
- **출력**: `forecast_results_ARIMA.csv`, `out_ARIMA.txt`

##### `VAR/`
- **목적**: 벡터 자기회귀 모델 (다변량 시계열)
- **특징**: 도메인 기반 특징 선택
  - 공격 예측 시: 외부 요인(NoM, ACA, PH) + 관련 기술(PATs)
  - 기술 예측 시: 관련 공격 유형
- **하이퍼파라미터**: 시차(lag) 최적화
- **출력**: `forecast_results_VAR.csv`, `out_VAR.txt`

##### `LSTM/`
- **`LSTM_u.py`**: 단변량 LSTM
  - 각 노드별로 독립적인 LSTM 모델
- **`LSTM_m.py`**: 다변량 LSTM
  - 도메인 기반 특징 선택 (VAR과 동일한 전략)
- **특징**: 순환 신경망 기반 시계열 예측
- **출력**: `out_u.txt`, `out_m.txt`

##### `Transformer/`
- **`transformer_u.py`**: 단변량 Transformer
- **`transformer_m.py`**: 다변량 Transformer
- **특징**: 어텐션 메커니즘 기반 시계열 예측
- **출력**: `out_u.txt`, `out_m.txt`

#### `MTGNN/`
- **목적**: 베이지안 확장 전 원본 MTGNN 모델
- **특징**: 베이지안 반복 없이 단일 예측 (반복 횟수 = 1)
- **출력**: `modelb1.pt`, `outb1.txt`

#### `BMTGNN/`
- **목적**: 베이지안 확장 MTGNN 모델 (반복 횟수별 평가)
- **특징**: 5가지 변형 평가
  - `BMTGNN.py`: 반복 횟수 조정 가능 (기본 10회)
  - 모델 파일: `modelb10.pt`, `modelb20.pt`, `modelb30.pt`, `modelb40.pt`, `modelb50.pt`
  - 결과 파일: `outb10.txt`, `outb20.txt`, `outb30.txt`, `outb40.txt`, `outb50.txt`
- **결과**: 반복 횟수 30회가 최적 성능 (논문 결과)

#### 평가 방법론
- **실험 반복**: 각 모델에 대해 5회 반복 실험
- **하이퍼파라미터 탐색**: 랜덤 서치 30회
- **평가 지표**: RRSE, RAE (평균값 계산)
- **예측 지평**: 36개월 (3년)

---

### `Data_Preparation/` - 데이터 수집 및 전처리

이 폴더는 사이버 보안 도메인의 특징을 생성하는 파이프라인을 포함합니다.  
**주가 예측 전환 시**: 이 폴더의 스크립트는 참고용이며, 금융 도메인에 맞는 특징 생성 파이프라인으로 대체해야 합니다.

#### `Hackmageddon_Attacks/`
- **목적**: 공격 사건 수집 (NoI: Number of Incidents)
- **데이터 소스**: Hackmageddon 웹사이트 CSV
- **스크립트**:
  - `NoI_daily.py`: 일별 사건 수 집계
    - 공격 유형 × 국가 조합별 카운트
    - 2011-07-01 ~ 2022-12-31
    - 출력: `NoI_daily.csv`
  - `NoI_monthly.py`: 월별 집계
    - `NoI_daily.csv` → `NoI_monthly.csv`
    - 윤년/말일 처리 포함
- **주의사항**: 하드코딩된 컬럼 수(col=988) 등이 있어 입력 포맷 변경 시 수정 필요

#### `Elsevier_Attacks/`
- **목적**: 논문에서 공격 언급량 수집 (A_NoM: Attack Number of Mentions)
- **데이터 소스**: Elsevier Scopus API + Selenium
- **스크립트**: `A_NoM.py`
  - Elsevier API로 공격 관련 논문 검색
  - Selenium으로 상세 페이지 로딩
  - 강조 텍스트에서 공격 키워드 카운트
  - 월별 집계: 2011-07 ~ 2022-12
  - 출력: `Attacks_NoM.txt` (탭 구분, 행=월, 열=공격)
- **필수 요구사항**:
  - Elsevier API Key (`config.json`)
  - Selenium + Firefox/GeckoDriver
  - 의존성: `elsapy`, `selenium`, `webdriver_manager`
- **주의사항**: 
  - 페이지 구조 변경 시 스크립트 수정 필요
  - Rate limit 고려 필요
  - 장시간 크롤링 부담

#### `Elsevier_PTs/`
- **목적**: 논문에서 기술 언급량 수집 (PT_NoM: Pertinent Technology Number of Mentions)
- **구조**: `Elsevier_Attacks/`와 동일
- **스크립트**: `PT_NoM.py`
  - PT 키워드 리스트 기반으로 월별 언급량 수집
  - 출력: `PTs_NoM.txt`

#### `Twitter_Tweets/`
- **목적**: 전쟁/분쟁 관련 트윗 수집 (ACA: Armed Conflict Areas)
- **데이터 소스**: Twitter API v2 (full-archive search)
- **스크립트**: `ACA.py`
  - 국가별 키워드 쿼리로 월별 total count 추출
  - 출력: `ACA.csv`
- **필수 요구사항**:
  - Twitter Bearer Token (환경변수 `TOKEN`)
- **주의사항**:
  - Rate limit 고려 (`time.sleep` 사용)
  - 최대치 상한 포함

#### `Python_Holidays/`
- **목적**: 공휴일 데이터 생성 (PH: Public Holidays)
- **스크립트**: `PH.py`
  - `holidays` 라이브러리 사용
  - 36개국 월별 공휴일 수 계산
  - 출력: `PH.csv`

---

### `Dataset/` - 통합 데이터셋

- **`CT-0711-1222.csv`**: 최종 통합 데이터셋
  - 기간: 2011-07 ~ 2022-12 (월별)
  - 특징: 공격 사건 수, 논문 언급량, 트윗, 공휴일 등
  - 형식: CSV (행=시점, 열=특징)
- **`README.md`**: 메타데이터 명명 규칙 설명
  - 특징 이름 형식 및 의미 설명
  - 예: `ATTACK-COUNTRY`, `Mentions-ATTACK`, `WAR/CONFLICT COUNTRY` 등

---

### `PT_Extractor/` - 그래프 엣지 추출

이 폴더는 공격별 관련 기술(PT) 목록을 GPT로 도출하여 TPT 그래프의 간선을 생성합니다.  
**주가 예측 전환 시**: "공격-기술" 관계를 "종목-연관지표/동일섹터" 관계로 치환하여 활용할 수 있습니다.

#### `E_GPT.py` (Extractive-GPT)
- **목적**: 논문에서 GPT로 기술 추출
- **프로세스**:
  1. Elsevier에서 공격 관련 논문 수집
  2. 각 논문의 Title/Abstract/Keywords에서 GPT로 PT 키워드 추출
  3. 키워드-완화용어 간 거리 기반 랭킹
  4. 상위 N개 반환
  5. 출력: `E_GPT.txt`
- **필수 요구사항**:
  - OpenAI API Key
  - Elsevier API Key (`config.json`)
  - Selenium (엘스비어 페이지 파싱)
  - `solution_synonyms.txt`: 동의어 사전
  - `excluded_keywords.txt`: 제외 키워드 목록

#### `D_GPT.py` (Direct-GPT)
- **목적**: GPT에 직접 질문하여 기술 추출
- **프로세스**:
  1. "어떤 공격에 대한 방어 기술 키워드 10개"를 직접 질문
  2. GPT 응답 파싱
  3. 필터링 후 저장
  4. 출력: `D_GPT.txt`
- **필수 요구사항**:
  - OpenAI API Key

#### 산출물 결합
- 두 결과(`E_GPT.txt`, `D_GPT.txt`)를 결합/클린하여 최종 PT 목록 생성
- 최종 목록을 `graph.csv`로 변환하여 그래프 엣지 정의에 활용

---

## 모델 아키텍처 상세 설명

### MTGNN (Multi-variate Time series Graph Neural Network)

#### 전체 구조
```
입력 [B, 1, N, T_in]
  ↓
Start Conv2d (1 → residual_channels)
  ↓
[Temporal CNN Block] × layers
  ├─ Dilated Inception (다양한 커널 크기)
  ├─ Graph Convolution (Mixprop)
  ├─ Skip Connection
  └─ Layer Normalization
  ↓
Skip Connections (입력 → 출력 직접 연결)
  ↓
End Conv2d (skip_channels → end_channels → out_dim)
  ↓
출력 [B, out_dim, N, 1]
```

#### 주요 구성 요소

##### 1. Temporal CNN (시간적 컨볼루션)
- **Dilated Convolution**: 확장된 컨볼루션으로 수용 영역 확대
  - 예: dilation=1, 2, 4, 8 → 점진적으로 넓은 시간 범위 포착
- **Inception 모듈**: 다양한 커널 크기 병렬 사용
  - 1×1, 3×1, 5×1, 7×1 커널을 병렬로 적용 후 결합
  - 다양한 시간 패턴 동시 포착

##### 2. Graph Convolution (그래프 컨볼루션)
- **Mixprop**: 그래프 확산 기반 GCN 레이어
  - 정적 그래프 + 적응형 그래프 혼합
  - 여러 깊이의 그래프 확산 결합
  - 공식: `(I - α·A)^k · X` (k는 깊이)
- **Graph Constructor**: 적응형 인접 행렬 생성
  - 노드 임베딩 기반으로 동적 그래프 학습
  - KNN 기반으로 가장 유사한 노드 연결

##### 3. Skip Connection
- 입력과 중간 레이어 출력을 직접 연결
- 깊은 네트워크에서 그래디언트 소실 방지
- 학습 안정성 향상

### 베이지안 확장 (Monte Carlo Dropout)

#### 원리
- **드롭아웃**: 학습 시 랜덤하게 뉴런을 비활성화하여 과적합 방지
- **베이지안 추론**: 드롭아웃을 추론 시에도 활성화하여 여러 번 예측
  - 각 예측은 다른 뉴런 조합을 사용
  - 여러 예측의 분포로 불확실성 정량화

#### 프로세스
1. 학습된 모델에 드롭아웃 활성화 (`model.train()` 모드 유지)
2. 동일 입력에 대해 `num_runs`번 반복 예측 (기본 10회)
3. 예측 결과 집계:
   - **평균**: `mean = (1/num_runs) * Σ predictions`
   - **분산**: `variance = (1/num_runs) * Σ (predictions - mean)²`
   - **95% 신뢰구간**: `mean ± 1.96 * sqrt(variance)`

#### 반복 횟수의 영향
- **10회**: 빠르지만 불확실성 추정이 부정확할 수 있음
- **30회**: 최적 성능 (논문 결과)
- **50회**: 더 정확하지만 계산 비용 증가
- **권장**: 30회 (성능과 효율의 균형)

---

## 평가 지표 설명

### RRSE (Root Relative Squared Error)
**정의**: 
```
RRSE = sqrt(Σ(y_true - y_pred)² / Σ(y_true - mean(y_true))²)
```

**의미**:
- 예측 오차의 제곱합을 기준선(평균) 오차의 제곱합으로 나눈 값의 제곱근
- 0에 가까울수록 좋음 (완벽한 예측 시 0)
- 1보다 크면 평균보다 나쁜 예측

**해석**:
- RRSE < 0.5: 우수한 예측
- 0.5 ≤ RRSE < 1.0: 양호한 예측
- RRSE ≥ 1.0: 기준선보다 나쁜 예측

### RAE (Relative Absolute Error)
**정의**:
```
RAE = Σ|y_true - y_pred| / Σ|y_true - mean(y_true)|
```

**의미**:
- 절대 오차의 합을 기준선 절대 오차의 합으로 나눈 값
- 0에 가까울수록 좋음
- 1보다 크면 평균보다 나쁜 예측

**해석**:
- RAE < 0.5: 우수한 예측
- 0.5 ≤ RAE < 1.0: 양호한 예측
- RAE ≥ 1.0: 기준선보다 나쁜 예측

### sMAPE (Symmetric Mean Absolute Percentage Error)
**정의**:
```
sMAPE = (100/n) * Σ|y_true - y_pred| / ((|y_true| + |y_pred|) / 2)
```

**의미**:
- 대칭 평균 절대 백분율 오차
- 0~100% 범위 (0에 가까울수록 좋음)
- MAPE와 달리 0 값에서도 안정적

### 상관계수 (Correlation Coefficient)
**정의**:
```
r = Σ(y_true - mean(y_true))(y_pred - mean(y_pred)) / 
    sqrt(Σ(y_true - mean(y_true))² * Σ(y_pred - mean(y_pred))²)
```

**의미**:
- 예측값과 실제값의 선형 상관관계
- -1 ~ 1 범위 (1에 가까울수록 좋음)

---

## 설치 및 실행

### 환경 요구사항
- **Python**: 3.6 이상
- **PyTorch**: 1.2.0 (오래된 버전, 호환성 주의 필요)
- **기타**: `requirements.txt` 참조

### 설치
```bash
cd B-MTGNN
pip install -r requirements.txt
```

### 주의사항
- **의존성 버전**: 매우 오래된 버전 사용 (2019년 기준)
  - `torch==1.2.0`, `numpy==1.17.4`, `pandas==0.25.3` 등
  - 최신 환경에서 호환성 문제 발생 가능
- **해결 방법**:
  1. 가상 환경 사용 권장
  2. 또는 상위 호환 버전으로 업그레이드 시도
     - `torch>=1.2.0`, `numpy>=1.17.4` 등으로 범위 지정

### 기본 실행 방법

#### 1) 하이퍼파라미터 탐색 (랜덤 서치)
```bash
cd B-MTGNN
python train_test.py --data ./data/sm_data.txt --seq_in_len 10 --seq_out_len 36
```
- **출력**:
  - `model/Bayesian/hp.txt`: 최적 하이퍼파라미터
  - `model/Bayesian/Validation/`: 검증 결과
  - `model/Bayesian/Testing/`: 테스트 결과

#### 2) 최종 모델 학습 (전체 데이터)
```bash
python train.py --data ./data/sm_data.txt --save model/Bayesian/o_model.pt --seq_in_len 10 --seq_out_len 36
```
- **출력**: `model/Bayesian/o_model.pt` (운용 모델)

#### 3) 미래 예측 및 시각화 (기본 36개월)
```bash
python forecast.py
```
- **출력**:
  - `model/Bayesian/forecast/data/*.txt`: 과거/예측/95% CI/분산
  - `model/Bayesian/forecast/plots/*.png, *.pdf`: 공격 vs 기술 그룹 플롯
  - `model/Bayesian/forecast/gap/*_gap.csv`: 공격-기술 연평균 격차
  - `model/Bayesian/forecast/pt_plots/*.png, *.pdf`: 개별 기술 플롯 (pt_plots.py 실행 시)

### 팁
- **시퀀스 길이 조절**: 데이터 주기에 맞게 조절
  - 일별 데이터: `--seq_in_len 60 --seq_out_len 20` (입력 60일, 출력 20일)
  - 주별 데이터: `--seq_in_len 26 --seq_out_len 8` (입력 26주, 출력 8주)
  - 월별 데이터: `--seq_in_len 10 --seq_out_len 36` (입력 10개월, 출력 36개월)
- **정규화**: 열별 최대값으로 정규화 후 학습, 역정규화하여 저장
- **노드 수**: `--num_nodes`는 데이터 열 수와 일치해야 함

---

## 데이터 준비 가이드

### 필수 파일 형식

#### 1. `data/sm_data.txt` (시계열 데이터)
- **형식**: 탭 구분 텍스트, 헤더 없음
- **구조**: 행 = 시점, 열 = 노드
- **예시** (3개 시점, 5개 노드):
```
0.5	0.3	0.8	0.2	0.6
0.6	0.4	0.9	0.3	0.7
0.7	0.5	1.0	0.4	0.8
```

#### 2. `data/sm_data_g.csv` (노드 이름)
- **형식**: CSV, 첫 행에 노드 이름 헤더
- **중요**: `sm_data.txt`의 열 순서와 1:1 일치해야 함
- **예시**:
```csv
Node1,Node2,Node3,Node4,Node5
```

#### 3. `data/graph.csv` (그래프 연결)
- **형식**: CSV, 첫 컬럼 = 기준 노드, 이후 컬럼 = 연결 노드들
- **예시**:
```csv
Node1,Node2,Node3
Node2,Node1,Node4
Node3,Node1,Node5
```

### 데이터 전처리

#### 스무딩 (선택사항)
- **목적**: 노이즈 제거 및 트렌드 강조
- **방법**: 더블 지수평활
- **실행**: `python smoothing.py`

#### 정규화
- **방법**: 열별 최대값 기준 정규화
  - 각 열의 최대값으로 나눔: `normalized = data / max(data)`
- **역정규화**: 예측 후 동일 스케일로 복원
- **주의**: 음수 값도 허용 (로그 수익률 등)

---

## 비교 평가 실험 상세

### 실험 설계
- **목적**: B-MTGNN 모델의 성능을 다양한 베이스라인과 비교
- **평가 방법**:
  - 각 모델에 대해 5회 반복 실험
  - 랜덤 서치 30회로 하이퍼파라미터 최적화
  - 평균 RRSE, RAE 계산
- **예측 지평**: 36개월 (3년)

### 베이스라인 모델

#### ARIMA
- **특징**: 전통적인 시계열 예측 모델
- **방법**: 단변량 (각 노드별 독립 모델)
- **하이퍼파라미터**: (p, d, q) 최적화

#### VAR
- **특징**: 벡터 자기회귀 (다변량)
- **방법**: 도메인 기반 특징 선택
- **장점**: 노드 간 상관관계 고려

#### LSTM
- **단변량**: 각 노드별 독립 LSTM
- **다변량**: 도메인 기반 특징 선택
- **특징**: 순환 신경망 기반

#### Transformer
- **단변량**: 각 노드별 독립 Transformer
- **다변량**: 도메인 기반 특징 선택
- **특징**: 어텐션 메커니즘 기반

### MTGNN vs BMTGNN
- **MTGNN**: 베이지안 확장 전 (반복 횟수 = 1)
- **BMTGNN**: 베이지안 확장 (반복 횟수 = 10, 20, 30, 40, 50)
- **결과**: BMTGNN (30회)가 최적 성능

### 결과 해석
- **성능 순위**: BMTGNN (30회) > MTGNN > 다변량 Transformer > 다변량 LSTM > VAR > ARIMA
- **베이지안 효과**: 불확실성 정량화로 더 신뢰할 수 있는 예측 제공

---

## 주가 예측으로 재가공하기

주요 아이디어: "노드 = 종목/지표", "간선 = 상관/테마/산업/지수-구성종목 관계"로 해석하여 다변량 그래프 시계열 예측으로 확장합니다.

### 1) 데이터 준비

#### 주기 선택
- **일별**: 종가, 수익률, 거래량 등
- **주별**: 주간 평균/합계
- **월별**: 월간 평균/합계

#### 파일 포맷 (원 코드 기대 형태)
- **`data/sm_data.txt`**: 탭 구분 텍스트, 헤더 없음
  - 형식: T × N 행렬 (T: 시점 수, N: 노드 수)
  - 예: 504행(거래일) × 20열(종목)
- **`data/sm_data_g.csv`**: 첫 행에 노드 이름 헤더
  - 예: `AAPL,MSFT,GOOGL,QQQ,...`
- **`data/graph.csv`**: 노드 간 연결 정의
  - 예:
    ```
    AAPL,MSFT,GOOGL,QQQ
    MSFT,AAPL,QQQ
    GOOGL,AAPL,QQQ
    QQQ,AAPL,MSFT,GOOGL
    ```

#### 권장 파이프라인
1. 원시 가격 데이터 수집
2. 결측 처리/정렬/리샘플
3. 변환 (예: 로그 수익률)
4. 정규화 전 `sm_data.txt` 저장
5. 노드 이름 `sm_data_g.csv` 생성
6. 그래프 연결 `graph.csv` 생성

### 2) 그래프 정의

#### 간단한 방법
- **섹터 기반**: 동일 섹터 내 전부 연결
- **지수 기반**: 지수 ↔ 구성종목 연결
- **예시**: 기술주 섹터 내 AAPL, MSFT, GOOGL 연결

#### 고급 방법
- **상관계수 기반**: 상관계수 상위 K개 연결
- **Granger 인과**: 인과관계 기반 연결
- **Graph Lasso**: 희소 그래프 학습
- **섹터/ETF 관계**: 섹터 ETF와 구성종목 연결

#### 주의사항
- `sm_data_g.csv`의 컬럼명과 `graph.csv`의 노드명이 정확히 일치해야 함
- 인접 행렬이 올바로 생성되려면 이름 매칭 필수

### 3) 하이퍼파라미터 설정

#### 시퀀스 길이
- **단기 일별**: `--seq_in_len 60 --seq_out_len 20` (입력 60일, 출력 20일)
- **중기 주별**: `--seq_in_len 26 --seq_out_len 8` (입력 26주, 출력 8주)
- **장기 월별**: `--seq_in_len 10 --seq_out_len 36` (입력 10개월, 출력 36개월)

#### 기타 하이퍼파라미터
- `--num_nodes`: 열 수(N)와 일치
- `--dropout`: 베이지안 추정 분산에 영향 (너무 낮으면 불확실성 축소)
- `--subgraph_size(k)`: 그래프 크기에 맞춰 조정
- `--node_dim`: 노드 임베딩 차원

### 4) 실행 명령 (주가 데이터 예시)
```bash
cd B-MTGNN
# 1) sm_data.txt, sm_data_g.csv, graph.csv 생성/배치 후
python train_test.py --data ./data/sm_data.txt --seq_in_len 60 --seq_out_len 20 --num_nodes 20
# 2) 최종 학습
python train.py --data ./data/sm_data.txt --save model/Bayesian/o_model.pt --seq_in_len 60 --seq_out_len 20 --num_nodes 20
# 3) 예측/시각화
python forecast.py  # 내부 기본값 사용. 필요시 seq 길이 등 스크립트 내 수정
```

### 5) 출력 해석
- **`forecast/data/<노드>.txt`**: 과거/예측/95% 신뢰구간/분산 (시점별 리스트)
- **신뢰구간 폭**: 불확실성 지표
  - 폭 확대 = 모델 불확실성 상승 (변동성 증가 시 유의)
- **다변량/그래프 효과**: 관련 종목 간 영향 전파 포착

### 6) 자주 맞닥뜨리는 이슈

#### 컬럼명 불일치
- **증상**: 인접 행렬이 비게 됨
- **해결**: `sm_data_g.csv` 헤더 ↔ `graph.csv` 노드명 일치 확인

#### 시퀀스 길이 불일치
- **증상**: `assert seq_len==self.seq_length` 오류
- **해결**: 입력 윈도우 길이와 인자 일치 필요

#### 버전/의존성 문제
- **증상**: 경고 또는 오류 발생
- **해결**: 가상 환경 사용 또는 상위 호환 버전으로 업그레이드

---

## 설계 상의 장점과 한계 (주가 적용 관점)

### 장점
1. **다변량·그래프 모델**: 섹터/지수/인기테마 등 구조적 의존성 반영
2. **베이지안 평균/신뢰구간**: 리스크 관리/포지션 사이징 의사결정에 활용 가능
3. **슬라이딩 윈도우 확장**: 다양한 예측 지평 적용 가능
4. **불확실성 정량화**: 예측의 신뢰도 제공

### 한계/주의
1. **정적 그래프 한계**: 시장 레짐 변화를 완전히 반영하지 못함 (동적 그래프 필요할 수 있음)
2. **지표 스케일 민감성**: RRSE/RAE는 스케일 민감 (수익률 사용 권장)
3. **데이터 전처리 중요**: 데이터 누수/리샘플링/거래일 간격 불균형 등 주의
4. **계산 비용**: 베이지안 반복으로 인한 계산 시간 증가
5. **오래된 의존성**: 호환성 문제 가능성

---

## 코드 참조 포인트

### 데이터 로더/정규화/인접행렬
- **`B-MTGNN/util.py`** / **`B-MTGNN/o_util.py`**
  - `DataLoaderS`: 시계열 데이터 로더
  - `build_predefined_adj()`: 정적 인접 행렬 생성
  - `create_columns()`: 컬럼명 생성 및 매핑

### 모델
- **`B-MTGNN/net.py`**: `gtnet` 클래스 (메인 모델)
- **`B-MTGNN/layer.py`**: 
  - `GCN` 레이어 (`nconv`, `mixprop`)
  - `Inception` 모듈 (`dilated_inception`)
  - `LayerNorm`: 레이어 정규화

### 학습/평가/탐색
- **`B-MTGNN/train_test.py`**: 하이퍼파라미터 최적화 및 평가
- **`B-MTGNN/train.py`**: 최종 모델 학습
- **`B-MTGNN/trainer.py`**: 학습 루프 및 최적화

### 예측/시각화/결과 저장
- **`B-MTGNN/forecast.py`**: 베이지안 예측 및 결과 저장
- **`B-MTGNN/pt_plots.py`**: 개별 노드 시각화

---

## 추가 리소스

### 논문
- 원 논문: [Forecasting Cyber Threats and Pertinent Mitigation Technologies](https://www.sciencedirect.com/science/article/pii/S0040162524006346)
- 기반 모델 논문: [MTGNN](https://dl.acm.org/doi/abs/10.1145/3394486.3403118)

### 데이터 소스
- Hackmageddon: https://www.hackmageddon.com/
- Elsevier Scopus API: https://dev.elsevier.com/
- Twitter API: https://developer.twitter.com/

---

## 결론

이 프로젝트는 사이버 보안 도메인의 트렌드 예측을 위한 종합적인 프레임워크를 제공합니다.  
다변량 그래프 신경망과 베이지안 추론을 결합하여 불확실성을 정량화한 예측을 제공하며,  
이는 주가 예측을 포함한 다양한 시계열 예측 문제에 적용 가능합니다.

**핵심 포인트**:
- 그래프 구조를 활용한 다변량 시계열 예측
- 베이지안 추론으로 불확실성 정량화
- 종합적인 데이터 수집 및 전처리 파이프라인
- 다양한 베이스라인과의 비교 평가

**주가 예측 전환 시**:
- 노드 = 종목/지표
- 간선 = 상관/섹터/지수 관계
- 특징 = 금융 팩터/거시 지표/뉴스 신호
- 동일한 모델 아키텍처와 학습 파이프라인 활용 가능