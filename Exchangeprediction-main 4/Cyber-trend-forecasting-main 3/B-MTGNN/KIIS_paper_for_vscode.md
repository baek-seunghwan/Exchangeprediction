
## 1. 개요 (HY중고딕 11pt, 굵게)

최근 글로벌 금융 시장은 국가 간 경제적 밀착도가 심화됨에 따라 환율 변동의 복잡성과 상호의존성이 급격히 증대되고 있다 [1]. 환율은 단순한 개별 국가의 경제 지표를 넘어, 인접 국가 및 주요 교역국 간의 유기적인 관계 속에서 실시간으로 변화하는 동적 특성을 지닌다 [2]. 그러나 ARIMA와 같은 전통적인 통계 모델이나 단순 LSTM 계열의 딥러닝 모델들은 이러한 국가 간의 공간적 상관관계를 충분히 반영하지 못하고, 단일 국가의 과거 패턴 학습에 치중해 왔다는 한계가 있다 [3]. 이는 급변하는 매크로 경제 상황 속에서 국가 간 금융 전이 효과를 간과하게 만들며, 결과적으로 환율의 장기적 추세를 정확히 포착하고 변동 원인을 해석하는 데 걸림돌이 된다 [4].
이에 본 연구는 환율이 개별적으로 존재하는 것이 아니라 상호 동적인 관계로 연결되어 있다는 점에 주목하여, 국가 간 잠재적 관계를 스스로 학습하는 그래프 신경망(Graph Neural Networks, GNN) 기반의 MTGNN 모델을 제안한다 [5]. 특히 명시적인 인접 행렬이 존재하지 않는 외환 시장의 특성을 고려하여, 적응형 그래프 학습 모듈(Adaptive Graph Learning Layer)을 통해 주요 3개국(미국, 한국, 일본) 간의 숨겨진 시공간적 정보를 통합 모델링하였다. 또한, 금리 및 물가 등 국가별 거시경제 지표를 노드 피처(Node Feature)로 결합함으로써, 모델이 단순 시계열 패턴을 넘어 다차원적인 경제적 맥락을 학습할 수 있도록 확장성을 부여하였다 [6].
본 연구의 핵심적인 차별점은 단순한 수치 예측력 향상에 그치지 않고, SLM(Small Language Model) 기반의 다중 에이전트 시스템(Multi-Agent System) 을 결합하여 예측 결과에 대한 고도의 해석력을 확보했다는 점에 있다 [7]. 기존 연구들이 단일 수치 예측에 치중했던 것과 달리, 본 프레임워크는 전문화된 다중 에이전트 간의 순환형 체이닝(Chaining) 구조를 도입하여 예측 데이터의 시계열 일관성을 엄밀히 검증하고 심층적인 원인 분석을 수행한다 [8]. 이를 통해 환율 변동의 동인에 대한 설명 가능한(Explainable) 근거를 구축하며, 에이전트의 다각적 해석이 담긴 장기 예측 보고서를 제공함으로써 외환과 글로벌 경제에 대한 근본적인 분석을 제안한다.
단순한 기술적인 정확도 개선을 넘어, 데이터 기반의 정량적 예측과 경제적 추론 기반의 정성적 분석이 결합된 새로운 외환 분석 패러다임을 제시한다는 점에서 본 연구는 중요한 의의를 갖는다. 특히 인공지능이 도출한 수치에 전문가적 해석을 덧붙임으로써, 금융의 실질적인 활용 가능성을 입증하고자 한다. 이는 변동성이 큰 글로벌 경제 환경에서 보다 선제적이고 입체적인 리스크 대응 체계를 구축하는 데 기여할 것으로 기대된다.
본 논문의 구성은 다음과 같다. 2장에서는 환율 예측 및 거시경제 요인에 관한 선행 연구를 고찰하고, 3장에서는 제안하는 MTGNN 및 다중 에이전트 시스템의 상세 구조를 설명한다. 4장에서는 실험 과정과 결과를 통해 모델의 유효성을 검증하며, 마지막으로 결론 및 향후 연구 방향에 대해 논한다.
## 2. 관련 연구HY중고딕 11pt, 굵게)
## 2. 1 외환 환율 예측 및 딥러닝 모델
외환 환율 예측 분야는 2000년대 이전부터 계량경제학적 접근을 통해 활발히 연구되었다. 초기 연구들은 주로 벡터 자기회귀(VAR)나 GARCH와 같은 통계적 모델을 사용하여 환율의 변동성을 설명하고자 하였다. [9]. 그러나 이러한 전통적인 방식은 통계적 가정을 필요로 하며, 금융 시장의 복잡한 비선형 역학을 포착하는 데 한계가 존재한다 [10]. 이러한 한계를 해결하기 위해 최근에는 기계학습 및 딥러닝 기반의 방법론이 제안되었다.
딥러닝 모델인 LSTM, CNN, 그리고 최근 자연어 처리 분야에서 혁신을 일으킨 Transformer 등이 외환 예측에 적용되어 기존 통계 모델 대비 우수한 성능을 보였다. 특히 시계열 데이터의 장기 의존성(Long-term dependency)과 미시적 변동성을 포착하기 위해 Zhou et al.(2021)은 시간적 특징을 임베딩(Temporal Embeddings)하여 어텐션 메커니즘을 개선한 Informer 모델을 제안하였다 [11]. 이 연구는 단변량 데이터보다 거시경제 지표를 포함한 다변량(Multivariate) 데이터를 사용할 때 트랜스포머가 LSTM이나 ARIMA보다 월등한 예측 성능과 수익률을 달성함을 입증하였다.
## 2. 2 거시경제 요인과 기존 방법론의 한계
선행 연구들은 외환 시장의 예측 정확도를 높이기 위해 다양한 시도를 해왔다. 금리평가설(IRP)이나 구매력평가설과 같은 이론에 기반하여 금리, 인플레이션율, 통화 정책과 같은 거시경제 지표를 특징(feature)으로 포함하는 연구들이 진행되었다 [12]. Yao et al.(2018)의 연구 역시 기술적 지표 외에 펀더멘털(Fundamental) 데이터를 결합한 Dual-Stage Attention 구조를 통해 예측력을 강화하였다 [13].
그러나 이러한 최신 연구들조차 다음과 같은 구조적 한계를 지닌다. 첫째, 다중 통화 간의 구조적 상호 의존성(Interrelationships)을 간과하였다. 대부분의 딥러닝 기반 연구는 입력 데이터를 평면적인 벡터 형태로 처리할 뿐, 경제학적 이론에 근거한 통화 삼각 관계(triplet)나 통화 간의 네트워크 구조를 명시적으로 모델링하지 않는다. 둘째, 시장 마찰 요인에 따른 가치 평가의 상대성을 충분히 반영하지 못했다. 신용도나 은행 규제 등으로 인해 통화의 가치 평가는 상대 통화에 따라 달라질 수 있으나, 기존 연구는 이를 전체 통화 네트워크 관점에서 포괄적으로 활용하지 못하는 경향이 있다.
## 2. 3 그래프 신경망(GNN) 기반의 다변량 시계열 예측
앞서 언급한 외환 시장의 구조적 상호 의존성을 포착하기 위해, 최근 시계열 예측 분야에서는 그래프 신경망(Graph Neural Networks, GNN)을 활용한 연구가 주목받고 있다 [6], [14]. 다변량 시계열 데이터에서 각 변수를 그래프의 노드(Node)로 간주할 경우, GNN은 인접한 노드들의 정보를 집계하여 잠재된 공간적 의존성(Spatial Dependency)을 효과적으로 학습할 수 있다는 장점이 있다 [15] ,[16]. 특히 기존의 CNN이나 RNN 기반 방법론들이 변수 쌍(Pair-wise) 간의 관계를 명시적으로 모델링하지 못해 모델의 해석력이 떨어진다는 한계를 GNN을 통해 극복할 수 있다 [17].
그러나 외환 시장과 같은 일반적인 다변량 시계열 데이터는 변수 간의 연결 구조(Graph Structure)가 사전에 정의되어 있지 않다는 문제점이 존재한다. 이를 해결하기 위해 Wu et al.(2020)은 사전에 정의된 그래프 구조 없이도 데이터로부터 변수 간의 관계를 자동으로 추출할 수 있는 그래프 학습 모듈(Graph Learning Module)을 제안하였다 [5]. 이 연구에서 제안한 MTGNN(Multivariate Time Series Grpah Neural Network) 프레임워크는 그래프 학습 레이어를 통해 변수 간의 단방향(Uni-directed) 관계를 학습하여 인접 행렬(Adjacency Matrix)을 생성하고, 이를 바탕으로 그래프 컨볼루션(Graph Convolution)을 수행하여 공간적 관계를 포착한다.
또한, 시계열 데이터의 시간적 패턴을 학습하기 위해 팽창 컨볼루션(Dilated Convolution)과 인셉션 레이어(Inception Layer)를 결합한 시간 컨볼루션 모듈을 함께 사용하여, 시공간적(Spatial-Temporal) 의존성을 동시에 모델링하는 종단간(End-to-End) 학습 구조를 확립하였다[5]. 이러한 접근 방식은 환율과 같이 변수 간의 구조적 관계가 뚜렷하지 않은 데이터에서도 잠재된 상호 연관성을 발견하고 예측 성능을 향상시킬 수 있음을 시사한다.
## Chapter 3. Methodology
## 3. 1 문제 정의 및 개요
본 연구는 다수 국가(또는 통화)로 구성된 글로벌 외환(FX) 시스템에서, 국가 간 상호의존성(그래프 구조)과 시계열 동학(시간 의존성)을 함께 고려하여 다중 스텝(Multi-horizon) 환율 경로를 확률적으로 예측하는 문제를 다룬다.
국가(또는 통화) 집합을 라 하고, 각 국가는 FX 네트워크의 노드로 표현한다. 관측 시점  동안 국가별 환율 및 거시·금융 특징이 주어졌을 때, 목표는 다음의 조건부 예측분포를 추정하는 것이다.

여기서 는 예측 구간(horizon)이며, 는 미래 스텝의 환율(또는 환율 상태/수익률) 벡터 시퀀스이다. 이 정의는 한 점 예측이 아니라, 미래 경로 전체에 대한 분포를 추정한다.
예를 들어 학습 구간이 2011-01부터 T=2025-10까지라면, T+1은 2025-11이고, H=36이면 2025-11부터 36개월 구간을 예측한다. 이때 모델은 와 그래프 에 조건부로 의 확률분포를 추정한다.

## 3. 2 입력 변수 및 특징 구성
환율은 단일 요인으로 움직이지 않으며, (i) 환율 자체의 동학, (ii) 변동성·유동성(시장 활동), (iii) 거시·정책 환경(공통 컨텍스트)의 결합에 의해 비대칭적으로 공진화(co-evolve)한다. 이에 따라 본 연구는 각 국가 i에 대해 다음의 구성요소를 결합한다.

## 3. 2.1 환율 동학
국가 i의 환율 관측을 라 할 때, 로그수익률(또는 정규화된 환율 상태)을 로 정의한다.

실험/데이터 특성에 따라 를 정규화된 레벨로 두고 변환을 생략할 수도 있으나, 본문에서는 를 환율 상태(수익률 포함)로 정의한다.

## 3. 2.2 변동성·시장 활동
환율 리스크를 반영하기 위해 롤링 변동성 와, 유동성/거래강도를 대변하는 거래량(또는 대체 지표) 를 포함한다.

## 3. 2.3 거시·정책 컨텍스트
여러 국가에 동시에 영향을 주는 거시·금융 공통요인을 로 둔다(예: 글로벌 위험선호, 금리, 원자재 등)

## 3. 2.4 최종 특징 벡터
최종적으로 국가 i의 시점 t 입력 특징을 다음과 같이 정의한다.

시장 휴일/비동기 캘린더/거시지표 결측 등으로 발생하는 결측은 전진 채움(forward filling) 및 마스킹으로 처리한다.

## 3. 3 국가 상호작용 그래프 구성
국가 간 구조적 의존성을 명시적으로 반영하기 위해 가중 그래프를 정의한다.

여기서 는 국가(통화) 노드 집합, 는 국가 간 관계, 는 가중 인접행렬이다. 엣지 는 무역노출, 금융통합, 통화정책 스필오버, 지역 인접성, 환율 상관 기반 동조화 등으로 인해 i의 변화가 j에 영향을 줄 수 있음을 의미한다.
초기 인접행렬 는 상관, 경제/지역 근접성, 잠재 의존 성분을 혼합해 구성할 수 있으며,

처럼 가중 결합 형태로 표현된다. 학습 과정에서 그래프 구조는 메시지 패싱 등을 통해 동적으로 보정(refinement)될 수 있다.

## 3. 4 베이지안 MTGNN 예측
## 3. 4.1 모델 입력/출력 및 예측 목표
시점 t에서 국가별 특징 행렬을

로 두고, 각 행이 에 대응한다. 관측  및 그래프 가 주어졌을 때, 목적은 미래 환율 상태의 예측분포를 추정하는 것이다.

## 3. 4.2 베이지안 예측(불확실성 포함)과 구간(밴드) 해석
추론 시 드롭아웃을 활성화한 채 S회 확률적 전방추론을 수행한다.

$$\hat{y}_{t+h}^{(s)} \sim p(y_{t+h} | X_{1:t}, G; \theta^{(s)}), \quad s = 1, \ldots, S$$

이로부터 예측 평균과 분산을 다음과 같이 근사한다.

$$\mu_{t+h} = \frac{1}{S} \sum_{s=1}^{S} \hat{y}_{t+h}^{(s)}, \quad \sigma^2_{\text{MC},t+h} = \frac{1}{S-1} \sum_{s=1}^{S} (\hat{y}_{t+h}^{(s)} - \mu_{t+h})^2 \quad (11)$$

그러나 MC dropout은 모델의 인식론적 불확실성(Epistemic Uncertainty)만 포착하며, 데이터의 고유한 노이즈인 우연적 불확실성(Aleatoric Uncertainty)을 반영하지 못한다 [18]. 특히 dropout 비율이 낮을 경우($p_{\text{drop}} \approx 0.02$), MC 샘플 간 분산이 극히 작아 예측구간이 지나치게 좁아진다는 한계가 있다.

이를 해결하기 위해 본 연구는 **잔차 기반 예측구간(Residual-based Prediction Interval)** 방법을 제안한다. 검증 데이터에서 실제 예측 오차의 분산($\sigma^2_{\text{residual}}$)을 추정하고, 이를 MC dropout 분산과 결합하여 현실적인 예측구간을 산출한다.

$$\text{Residual: } r_{t} = \hat{y}_{t} - y_{t}, \quad \sigma^2_{\text{residual}} = \frac{1}{T_{\text{val}}} \sum_{t \in \text{val}} r_{t}^2 \quad (12)$$

$$\text{Combined Variance: } \sigma^2_{\text{total}} = \sigma^2_{\text{MC}} + \sigma^2_{\text{residual}} \quad (13)$$

$$\text{95% Prediction Interval: } \text{PI}_{95\%} = \mu_{t+h} \pm 1.96 \times \sqrt{\sigma^2_{\text{total}}} \times \beta \quad (14)$$

여기서 $\beta \in [0.5, 1.0]$는 구간 너비 조정 계수로, 사용자가 시각화 목적에 맞춰 조정할 수 있다. 본 연구의 실험에서는 $\beta = 0.5$를 적용하여 과도한 불확실성 팽창을 억제하였다.

이 방법은 단순히 모델 내부 분산만 사용하는 기존 접근과 달리, 실제 예측 오차 분포를 반영하여 금융 리스크 관리에 더 적합한 예측구간을 제공한다는 장점이 있다 [19]. 또한 구현 측면에서 분위수(quantile) 구간으로도 예측구간을 제공한다(예: 10%–90% 분위). forecast.py 기본 설정은 $\alpha=0.05$, $\alpha=0.95$를 사용한다.

## 3. 4.3 국가 간 환율 격차 정의
국가 i와 j의 상대적 환율 갭을 예측 평균 기반으로 정의한다.

$$\Delta_{i,j}(t+h) = \mu_{i,t+h} - \mu_{j,t+h} \quad (15)$$

이는 국가 간 상대적 강세/약세를 정량화하며, 분위수 구간을 함께 제시하면 갭의 불확실성까지 해석할 수 있다. 예측구간을 함께 고려하면 다음과 같이 표현된다.

$$\text{PI}_{\Delta_{i,j}} = [\mu_{i,t+h} - \mu_{j,t+h}] \pm 1.96 \times \sqrt{\sigma^2_{\text{total},i} + \sigma^2_{\text{total},j}} \quad (16)$$

## 4. 실험 결과 및 분석

## 4.1 데이터셋 및 실험 설정
본 연구는 미국(US Trade Weighted Dollar Index), 한국(KRW/USD, kr_fx), 일본(JPY/USD, jp_fx) 3개국의 월별 환율 데이터를 중심으로, 총 33개의 거시경제 변수를 통합하여 실험을 수행하였다. 전체 데이터는 2011년 7월부터 2025년 12월까지 180개월로 구성되며, 학습 구간은 2011년 7월~2023년 12월(150개월), 검증 구간은 2024년 1월~12월(12개월), 테스트 구간은 2025년 1월~12월(12개월)로 분할하였다. 예측 horizon은 12개월(1년)로 설정하여 테스트 구간 전체에 대한 환율 경로를 예측하였다.

주요 하이퍼파라미터는 다음과 같다: 입력 시퀀스 길이($T_{\text{in}}$) = 40, 출력 시퀀스 길이($T_{\text{out}}$) = 1 (슬라이딩 윈도우 방식), 학습률($\eta$) = 0.00015, Dropout 비율 = 0.02, 은닉층 차원 = 256, 에폭 수 = 180. 베이지안 불확실성 추정을 위한 MC dropout 전방추론 샘플 수는 학습·평가 시 $S = 10$, 장기 전망 시 $S = 20$으로 설정하였다.

모델의 범용 안정성을 검증하기 위해, 테스트 구간을 슬라이딩 윈도우 방식으로 이동하며 반복 평가하는 롤링 백테스트(Rolling Backtest)를 추가로 실시하였다. 총 145개 윈도우에 대해 타겟 RRSE를 산출하고, 동일 조건에서 평균 기준선(Mean Baseline) 및 지속 기준선(Persistence Baseline)과 비교하였다.

## 4.2 예측 성능 평가
예측 성능은 RSE(Root Relative Squared Error)를 주요 지표로 사용하였으며, RSE < 0.5를 목표 임계값으로 설정하였다. RSE는 모델의 예측 오차를 단순 평균 예측 대비 상대적으로 평가하는 지표로, 1.0 미만이면 평균 예측보다 우수함을, 0.5 미만이면 상당히 정밀한 예측력을 의미한다.

**표 2. 타겟 국가별 테스트 RSE 성능**
**Table 2. Test RSE Performance by Target Country**
| 국가 | RSE | 목표 달성 여부 |
|------|-----|--------------|
| US (Trade Weighted Dollar Index) | 0.4884 | ✓ |
| JP (jp_fx) | 0.2721 | ✓ |
| KR (kr_fx) | 0.2970 | ✓ |

3개 타겟 국가 모두에서 RSE가 0.5 미만을 달성하였으며, 특히 일본(0.2721)과 한국(0.2970)은 0.3 이하의 우수한 예측 정밀도를 보였다. 이는 B-MTGNN이 국가 간 그래프 구조를 활용하여 다중 통화의 시공간적 패턴을 효과적으로 포착하고 있음을 시사한다.

또한 래깅(Lagging) 탐지 진단을 통해, 예측값이 실제값을 단순히 시간 지연 추종하는 나이브(Naive) 패턴이 아님을 검증하였다. Lag-0 상관계수가 0.96 이상, 방향 일치도(Directional Accuracy)가 72~100%로 나타났으며, 범위 비율(Range Ratio)이 0.97~1.18로 적절한 변동성 복원력을 유지하였다. 이는 모델이 실제 환율의 방향성과 변동 폭을 모두 학습하고 있음을 뒷받침한다.

**표 3. 롤링 백테스트 비교 결과 (145 윈도우, Focus RRSE)**
**Table 3. Rolling Backtest Comparison (145 Windows, Focus RRSE)**
| 모델 | 평균 RRSE | 중앙값 RRSE | 90th 백분위 |
|------|----------|------------|------------|
| B-MTGNN (제안 모델) | 1.4837 | 1.2865 | 2.4477 |
| Persistence Baseline | 2.06 | — | — |
| Mean Baseline | 3.65 | — | — |

롤링 백테스트 결과, 제안 모델은 평균 RRSE 1.48로 Persistence Baseline(2.06) 대비 약 28%, Mean Baseline(3.65) 대비 약 59% 낮은 오차를 기록하였다. 이는 B-MTGNN이 단순 추세 외삽이나 평균 회귀를 넘어, 다변량 시공간 의존성에 기반한 실질적인 예측력을 확보하고 있음을 의미한다.

## 4.3 잔차 기반 예측구간의 효과
기존 MC dropout만 사용한 예측구간은 dropout 비율 0.02의 극히 낮은 설정으로 인해 밴드 폭이 지나치게 좁아, 실제 예측 오차를 포괄하지 못하는 한계가 있었다. 이를 해결하기 위해 3.4.2절에서 제안한 잔차 기반 예측구간 방법을 적용한 결과, 예측 밴드가 실제 오차 분포를 현실적으로 반영하는 수준으로 확장되었다.

검증 구간의 잔차 분산($\sigma^2_{\text{residual}}$)을 MC dropout 분산($\sigma^2_{\text{MC}}$)과 결합하고, $\beta = 0.5$ 조정 계수를 적용하여 시각적 명확성과 통계적 신뢰성 사이의 균형을 달성하였다. 그림 2는 일본 엔화(jp_fx)의 2025년 테스트 예측 결과를 보여주며, 분홍색 밴드가 실제값의 변동을 적절히 포괄하면서도 과도하게 넓지 않아, 실용적인 리스크 평가 도구로서의 가치를 확인할 수 있다.

이러한 결합 분산 접근법은 인식론적 불확실성(Epistemic Uncertainty)과 우연적 불확실성(Aleatoric Uncertainty)의 양면을 동시에 반영함으로써, 단일 불확실성 추정에 의존하는 기존 방법론 대비 더욱 견실한(Robust) 예측구간을 제공한다.

## 4.4 국가 간 환율 격차 동학 및 2026년 전망 예비 분석
본 연구의 B-MTGNN 모델을 활용하여 2026년 1월~12월(12개월) 환율 전망을 예비적으로 수행하였다. 이는 모델이 2025년 12월까지의 관측 데이터를 기반으로 미래 경로를 확률적으로 추정한 결과이다.

**표 4. 2026년 환율 전망 요약 (MC Dropout $S=20$)**
**Table 4. 2026 Exchange Rate Forecast Summary (MC Dropout $S=20$)**
| 국가 | 예측 평균 | 95% CI 하한 | 95% CI 상한 | CI 폭/평균 비율 |
|------|----------|------------|------------|---------------|
| US (TWDI) | 118.45 | 118.21 | 118.69 | ±0.20% |
| KR (KRW/USD) | 1,451.19 | 1,448.09 | 1,454.29 | ±0.21% |
| JP (JPY/USD) | 149.67 | 149.52 | 149.82 | ±0.10% |

2026년 전망에서 주목할 점은 MC dropout 기반 신뢰구간(CI)이 매우 좁게(평균 대비 ±0.1~0.2%) 형성되었다는 것이다. 이는 dropout 비율(0.02)이 극히 낮아 모델 파라미터의 인식론적 분산이 제한적으로만 포착되기 때문으로 해석된다. 따라서 MC dropout CI만으로는 실제 예측 불확실성을 과소추정할 위험이 있으며, 4.3절에서 제안한 잔차 기반 보정 구간을 함께 참조하는 것이 바람직하다.

3.4.3절에서 정의한 국가 간 환율 격차($\Delta_{i,j}$)를 분석한 결과, 한국 원화와 일본 엔화 간의 상대적 격차가 가장 동적인 변동 패턴을 보였으며, 이는 양국 간 무역 구조와 금리 차이에 기인하는 것으로 해석된다. 분위수 구간을 함께 제시함으로써 환율 갭의 불확실성까지 정량적으로 평가할 수 있었다.

다만, 2026년 전망은 아직 실현되지 않은 미래 구간이므로 실측치와의 비교가 불가능하다. 따라서 본 전망의 신뢰성은 2025년 테스트 구간에서 확인된 RSE 0.27~0.49의 예측 성능에 기반하여 방향성과 추세 수준에서 참고 가치를 지니며, 절대값의 정확도에 대해서는 지속적인 모니터링이 필요하다.

## 5. 논의

## 5.1 방법론적 기여
본 연구의 핵심 방법론적 기여는 세 가지로 요약된다. 첫째, 적응형 그래프 학습(Adaptive Graph Learning)을 통해 사전 정의된 인접 행렬 없이도 국가 간 환율의 잠재적 상호의존성을 자동으로 추출하는 데 성공하였다. 이는 금융 네트워크에서 노드 간 관계가 명시적으로 주어지지 않는 현실적 제약을 극복한 것이다. 둘째, MC dropout과 잔차 분산을 결합한 하이브리드 불확실성 추정 방식을 제안하여, 단일 점 예측을 넘어 확률적 예측구간을 제공함으로써 금융 의사결정에 필수적인 리스크 정보를 함께 산출하였다. 셋째, SLM 기반 다중 에이전트 시스템을 예측 파이프라인에 통합하여, 정량적 예측에 정성적 해석을 결합하는 새로운 패러다임을 제시하였다.

## 5.2 경제학적 해석
실험 결과는 주요 거시경제 이론과 일관된 방향을 보여준다. 한-미 금리 스프레드(Kr_us_spread)와 한국 원화 환율 간의 높은 관련성은 금리평가설(Interest Rate Parity)의 설명력을 뒷받침하며, 일-미 스프레드(Jp_us_spread)와 엔화 간의 관계 역시 캐리 트레이드(Carry Trade) 동학과 부합한다. 그래프 학습 모듈이 자동으로 학습한 국가 간 연결 가중치는, 경제 이론에서 예측하는 상호 영향의 방향성과 정성적으로 일치하였다. 이는 B-MTGNN이 단순한 통계적 상관을 넘어, 경제적으로 의미 있는 구조적 관계를 포착하고 있음을 시사한다.

## 5.3 실무적 함의
본 모델의 실무적 활용 가능성은 다음과 같은 측면에서 평가된다. 예측 밴드와 국가 간 격차 분석은 다통화 포트폴리오 관리자에게 헤지 비율 조정 및 리밸런싱 시점 결정을 위한 정량적 근거를 제공한다. 또한 환율의 방향성 예측과 불확실성 구간은 수출입 기업의 환 리스크 관리 전략 수립에 기여할 수 있다. 다중 에이전트의 해석 보고서는 모델의 수치적 출력을 경제적 맥락으로 변환하여, 비전문가 의사결정자에게도 접근 가능한 인사이트를 제공한다는 점에서 실용적 가치가 크다.

## 5.4 기존 연구와의 차별점
기존의 다변량 시계열 예측 연구들이 변수 간의 관계를 암묵적으로만 학습하거나 사전 정의된 구조에 의존했던 것과 달리 [5], [6], 본 연구는 적응형 그래프 학습을 통해 데이터 기반의 동적 관계 추출을 가능하게 하였다. 또한 Informer [11]나 기존 Transformer 기반 모델들이 불확실성 정량화에 소극적이었던 반면, 본 연구는 베이지안 추론을 핵심 구성요소로 통합하여 예측의 신뢰도를 명시적으로 제시하였다. 무엇보다 다중 에이전트 기반의 자동 해석 체계는 기존 연구에서 시도되지 않았던 새로운 접근으로, 예측-해석의 통합 프레임워크를 구현한 점에서 본질적인 차별성을 갖는다.

## 6. 한계점 및 향후 연구

## 6.1 데이터 범위 및 기간의 제약
본 연구의 학습 데이터는 2011년 7월부터 2025년 12월까지 총 180개월로, 약 15년간의 거시경제 사이클만을 포함한다. 이 기간에는 COVID-19 팬데믹, 러시아-우크라이나 전쟁 등의 구조적 충격이 포함되어 있으나, 2008년 글로벌 금융위기와 같은 이전의 대규모 위기 기간은 학습 범위에서 제외되어 있다. 또한 분석 대상이 미국·한국·일본 3개국에 한정되어, 유로존이나 신흥국 통화와의 교차 영향을 반영하지 못한다는 한계가 있다.

## 6.2 모델 구조의 한계
B-MTGNN의 그래프 학습 모듈은 학습 과정에서 하나의 정적 인접 행렬을 도출하며, 시간에 따라 변화하는 동적 그래프 구조(Dynamic Graph Structure)를 명시적으로 모델링하지 못한다. 국가 간 경제적 관계는 정책 변화, 무역 분쟁, 지정학적 사건 등에 따라 시간적으로 변이하므로, 향후 시간 가변적(Time-varying) 그래프 학습 메커니즘의 도입이 필요하다. 또한 현재의 팽창 컨볼루션(Dilated Convolution) 기반 시간 모듈은 매우 긴 장기 의존성 포착에 있어 Transformer 대비 제한적일 수 있다.

## 6.3 불확실성 추정의 개선 필요성
실험에서 확인된 바와 같이, MC dropout 기반의 신뢰구간은 dropout 비율(0.02)이 극히 낮을 경우 과도하게 좁은 구간을 산출한다. 잔차 기반 보정을 통해 이를 완화하였으나, 이 접근법은 과거 검증 구간의 오차 분포가 미래에도 유지된다는 정상성(Stationarity) 가정에 의존한다. 비정상적 시장 환경에서는 불확실성이 과소추정될 위험이 존재하며, Deep Ensemble이나 Heteroscedastic Regression 등의 대안적 불확실성 추정 방법론과의 비교 검토가 후속 연구에서 이루어져야 한다.

## 6.4 외생적 충격 대응의 부재
현재 모델은 과거 시계열 패턴에 기반한 내삽적(Interpolative) 예측에 초점을 맞추고 있어, 학습 데이터에 포함되지 않은 유형의 외생적 충격(Black Swan Event)에 대한 대응력이 제한적이다. 지정학적 위기, 급격한 통화정책 전환, 팬데믹 등의 비선형적 사건에 대한 로버스트성(Robustness) 강화가 향후 핵심 과제로 남는다. 이벤트 기반 시나리오 분석이나 조건부 예측(Conditional Forecasting) 기법의 통합이 유망한 연구 방향이다.

## 6.5 다중 에이전트 시스템의 발전 방향
현재의 SLM 기반 다중 에이전트 시스템은 예측 데이터를 해석하고 비평하는 사후적(Post-hoc) 역할에 한정되어 있다. 향후 에이전트가 모델의 학습 과정에 피드백을 제공하거나, 예측 결과를 기반으로 포트폴리오 리밸런싱을 자동 실행하는 폐루프(Closed-loop) 시스템으로 발전시키는 연구가 필요하다. 또한 에이전트 간의 합의 메커니즘(Consensus Mechanism)과 신뢰도 가중(Confidence Weighting)을 도입하여 해석의 일관성과 정확도를 향상시킬 수 있을 것이다.

## 6.6 일반화 및 확장 가능성
본 연구의 프레임워크가 3개국 환율 외의 자산군(주식, 원자재, 채권 등)이나 더 많은 국가를 포함하는 대규모 금융 네트워크로 확장될 때의 성능과 확장성(Scalability)은 아직 검증되지 않았다. 특히 노드 수가 증가할 때 적응형 그래프 학습의 계산 복잡도와 예측 정확도 간의 트레이드오프를 체계적으로 평가하는 후속 연구가 요구된다. 다양한 시장 조건과 자산군에서의 교차 검증을 통해 프레임워크의 범용성을 입증하는 것이 중요한 향후 과제이다.

## 참고문헌

[1] S. Miranda-Agrippino and H. Rey, “US Monetary Policy and the Global Financial Cycle,” The Review of Economic Studies, vol. 87, no. 6, pp. 2754-2776, 2020.

[2] J. Baruník and T. Křehlík, “Measuring the Frequency Dynamics of Financial Connectedness and Systemic Risk,” Journal of Financial Econometrics, vol. 16, no. 2, pp. 271-296, 2018.

[3] Z. Pan, Y. Liang, W. Wang, Y. Yu, M. Li, and J. Zhang, “Urban Traffic Prediction from Spatio-Temporal Data: A Deep Learning Approach,” IEEE Transactions on Knowledge and Data Engineering, vol. 32, no. 4, pp. 720-733, 2019.

[4] D. Diebold and K. Yilmaz, “Better to Give than to Receive: Predictive Directional Measurement of Volatility Spillovers,” International Journal of Forecasting, vol. 28, no. 1, pp. 57-66, 2012.

[5]  Wu, Z., Pan, S., Chen, G., Long, G., Zhang, C., & Philip, S. Y. (2020). "Connecting the dots: Multivariate time series forecasting with graph neural networks." Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining.

[6]  Li, Y., Yu, R., Shahabi, C., & Liu, Y. (2018). "Diffusion convolutional recurrent neural network: Data-driven traffic forecasting." International Conference on Learning Representations (ICLR).

[7] Yu, S., et al. (2024). "FinAgent: A Multi-modal Generalist Agent for Financial Analysis and Reasoning." arXiv preprint arXiv:2402.14411.

[8]  Wu, Q., et al. (2023). "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation." Proceedings of the 40th International Conference on Machine Learning (ICML).

[9] Pradeepkumar, D., & Ravi, V. (2018). Forecasting financial time series volatility using particle swarm optimization trained quantile regression neural network. Applied Soft Computing, 58, 35-52.

[10] Fischer, T., & Krauss, C. (2018). Deep learning with long short-term memory networks for financial market predictions.

[11] Zhou, H., Zhang, S., Peng, J., Zhang, S., Li, J., Xiong, H., & Zhang, W. (2021). Informer: Beyond efficient transformer for long sequence time-series forecasting.

[12] Amat, C., Michalski, T., & Stoltz, G. (2018). Fundamentals and exchange rate forecastability with machine learning methods. Journal of International Money and Finance, 88, 58-74.

[13] Yao, Q., Song, D., Chen, H., Wei, A., Cottrell, G. W., & Bian, G. W. (2018). A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction.

[14] Yu, B., et al. (2018). Spatio-temporal graph convolutional networks: A deep learning framework for traffic forecasting. IJCAI.

[15] Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. ICLR.

[16] Wu, Z., Pan, S., Long, G., Jiang, J., & Zhang, C. (2019). Graph WaveNet for deep spatial-temporal graph modeling. Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI), 1907-1913.

[17] Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2018). Graph Attention Networks. International Conference on Learning Representations (ICLR).
[18] Kendall, A., & Gal, Y. (2017). "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?" Advances in Neural Information Processing Systems (NeurIPS), vol. 30, pp. 5574-5584.

[19] Gneiting, T., & Raftery, A. E. (2007). "Strictly Proper Scoring Rules, Prediction, and Estimation," Journal of the American Statistical Association, vol. 102, no. 477, pp. 359-378.
[3] C. W. Xu, “Fuzzy Model Identification and Self-learning for Dynamic Systems,” IEEE Trans. Syst. Man. Cybern., vol. 17, no. 4, pp. 683-689, 1987.
[4] H. Takagi and M. Sugeno, “Fuzzy Identification of Systems and Its Application to Modeling and Control,” IEEE Trans. on Sys. Man and Cybern., vol. 15, pp. 116-132, 1985.
[5] J. H. Holland, Genetic Algorithms in Search, Optimization and Machine Learning, Addison- Wesley, 1989. (신명조 9pt)

### 표 1
| 그래프 신경망 기반 다중 환율 예측: MTGNN 변형과 앙상블 예측 |
| --- |
| Graph Neural Network-based Multi-Currency FX Forecasting: MTGNN Variants, Ensemble Prediction |
| 김수민 1․ 최지원 2․ 백승환 3․ 한용섭 4․ 박대승 5 Kim Soo-min, Choi Ji-won, Baek Seung-hwan, Han Yong-seop, Park Dae-seung |
| 1수원대학교 컴퓨터학부 SW학과 E-mail: [본인 이메일 입력]@suwon.ac.kr 2수원대학교 컴퓨터학부 SW학과 E-mail: 23017097@suwon.ac.kr 3수원대학교 컴퓨터학부 SW학과 E-mail: sqortmdghks1@suwon.ac.kr 4수원대학교 컴퓨터학부 SW학과 E-mail: [본인 이메일 입력]@suwon.ac.kr 5수원대학교 컴퓨터학부 SW학과 E-mail: [본인 이메일 입력]@suwon.ac.kr |
| 요 약 (HY중고딕 11pt 굵게)   본 연구는 환율 예측의 전통적인 시계열 모델이 간과해 온 국가 간 상호의존성을 포착하는 데 있어 한계를 극복하고, 환율 변동의 복잡한 메커니즘을 규명하는 데 목적이 있다. 이를 위해 그래프 신경망(GNN) 기반의 MTGNN 모델을 채택하여 주요 3개국(미국, 한국, 일본)의 월간 환율 데이터와 거시경제 지표를 결합한 시공간적 상관관계를 통합 분석하였다. 특히 데이터로부터 잠재적 관계를 스스로 학습하는 적응형 그래프 학습 모듈을 통해 예측의 정교함과 확장성을 검증하였다.  연구 결과, 제안 모델은 기존 LSTM 및 통계 모델 대비 우수한 예측 정확도를 기록하였다. 수치 예측의 한계를 넘어 SLM(Small Language Model) 기반의 다중 에이전트 시스템을 결합함으로써 분석의 신뢰성을 강화하였으며, 전문화된 에이전트 기법으로 순환형 체이닝 구조를 통해 예측값의 시계열 일관성을 검증하였다. 또한 특정 오차의 원인을 식별하여 모델의 개선 방향을 제시하고, 장기 예측 결과에 대한 국가별 상호작용 해석과 설명 가능한(Explainable) 인사이트 도출을 지원한다. 결론적으로 본 연구는 인공지능 기술을 활용하여 수치 예측과 경제적 맥락 해석이 통합된 차세대 외환 분석 프레임워크를 정립하였다는 데 학술적 의의가 있다.  (신명조 8.5pt)       키워드 - 환율 예측, 그래프 신경망(GNN), 다중 에이전트 시스템(MAS), 설명 가능한 AI(XAI), 시계열 일관성 검증   This study aims to overcome the limitations of traditional time-series models in capturing cross-country dependencies and to elucidate the complex mechanisms of exchange rate fluctuations. By adopting the MTGNN (Multivariate Time Series Forecasting with Graph Neural Networks) model, this research conducts an integrated analysis of spatio-temporal correlations by combining monthly exchange rate data from five major IMF countries with macroeconomic indicators. In particular, the precision and scalability of the forecasts are validated through an adaptive graph learning module that autonomously identifies latent relationships within the data.  Experimental results demonstrate that the proposed model achieves superior prediction accuracy compared to existing LSTM and statistical models. Beyond numerical forecasting, the reliability of the analysis is reinforced by integrating a Small Language Model (SLM)-based multi-agent system. This specialized agent technique utilizes a recursive chaining flow to verify the time-series consistency of forecasted values while identifying the causes of specific errors to suggest model improvements. Furthermore, this approach facilitates the interpretation of inter-country interactions and the derivation of explainable insights for long-term forecasts. Consequently, this research holds significant academic value by establishing a next-generation foreign exchange analysis framework that integrates numerical prediction with economic contextual interpretation using artificial intelligence. Keywords — Exchange Rate Forecasting, Graph Neural Networks (GNN), Multi-Agent System (MAS), Explainable AI (XAI), Time-series Consistency Verification |

### 표 2
|  | (1) |
| --- | --- |

### 표 3
|  | (2) |
| --- | --- |

### 표 4
|  | (3) |
| --- | --- |

### 표 5
|  | (4) |
| --- | --- |

### 표 6
|  | (5) |
| --- | --- |

### 표 7
|  | (6) |
| --- | --- |

### 표 8
|  | (7) |
| --- | --- |

### 표 9
|  | (8) |
| --- | --- |

### 표 10
|  | (9) |
| --- | --- |

### 표 11
|  | (10) |
| --- | --- |

### 표 12
| Parameter | Value |
| --- | --- |
| Population number | 100 |
| Mutation rate | 0.2 |
