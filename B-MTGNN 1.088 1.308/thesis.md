content = """# 3. Methodology (방법론)

본 장에서는 환율 다변량 시계열을 그래프 기반 시계열 예측 문제로 정식화하고, 동적 그래프 학습 및 다중-스텝 예측 구조를 수식 중심으로 기술한다.

### 3.1 문제 정의 (Problem Formulation)

환율 및 보조변수로 구성된 다변량 시계열 데이터를 그래프 기반 예측 문제로 정의합니다.

**입력 데이터 (Input Window)**
$$X = \{x_t, x_{t+1}, \dots, x_{t+P-1}\}, \quad x_t \in \mathbb{R}^N$$
* $X$: 길이 $P$의 입력 윈도우
* $x_t$: 시점 $t$에서의 $N$개 노드(환율 및 보조변수) 관측값 벡터

**예측 모델 및 출력 (Forecasting)**
$$\hat{Y} = f_\\theta(X, \\tilde{A}), \quad \hat{Y} = \{\\hat{y}_{t+1}, \dots, \\hat{y}_{t+H}\}, \quad \\hat{y}_{t+h} \in \mathbb{R}^N$$
* $f_\\theta$: 예측 모델
* $\\tilde{A}$: 동적으로 학습되는 그래프(인접행렬)
* $\\hat{Y}$: $H$-step 예측 결과

### 3.2 동적 그래프 구축 (Dynamic Graph Construction)

고정된 그래프 대신 학습 가능한 노드 임베딩을 통해 인접행렬을 적응적으로 생성합니다.

**노드 임베딩 (Node Embeddings)**
$$E_1 = \\text{Emb}_1(V), \quad E_2 = \\text{Emb}_2(V), \quad E_1, E_2 \in \mathbb{R}^{N \\times d}$$
* $V$: 노드 집합
* $d$: 임베딩 차원

**유사도 및 인접행렬 생성 (Similarity & Adjacency Matrix)**
$$S = \\tanh(\\alpha \\cdot E_1 W_1) \\cdot (\\tanh(\\alpha \\cdot E_2 W_2))^T$$
$$\\tilde{A} = \\text{rowsoftmax}(\\text{ReLU}(S))$$
* $S$: 내적 형태의 유사도 행렬 ($\\alpha$는 스케일 하이퍼파라미터)
* $\\tilde{A}$: ReLU로 음수 연결을 제거하고 정규화한 확률적 인접행렬

**그래프 희소화 (Sparsification)**
$$\\tilde{A}_k = \\text{TopK}(\\tilde{A}, k)$$
* 각 노드 기준 상위 $k$개의 연결만 유지하여 과적합 방지 및 계산 효율화

### 3.3 시간적 모델링 (Temporal Modeling via Dilated Gated Conv)

확장 합성곱(Dilated Convolution)과 게이팅 메커니즘을 통해 시계열의 장기 의존성을 포착합니다.

**초기 변환 및 게이팅 (Transformation & Gating)**
$$Z^{(0)} = \\text{Conv}_{1 \\times 1}(X)$$
$$F^{(l)} = \\tanh(\\text{Conv}_d^{(l)}(Z^{(l-1)})), \quad G^{(l)} = \\sigma(\\text{Conv}_d^{(l)}(Z^{(l-1)}))$$
* $Z^{(0)}$: 1x1 합성곱을 통한 차원 매핑
* $F^{(l)}, G^{(l)}$: $l$번째 블록의 필터 및 게이트 (tanh, sigmoid 사용)

**특징 추출 및 잔차 연결 (Feature Extraction & Residual)**
$$H^{(l)} = F^{(l)} \\odot G^{(l)}$$
$$Z^{(l)} = H^{(l)} + \\text{Align}(Z^{(l-1)}, H^{(l)})$$
* $\\odot$: 원소별 곱 (Element-wise product)
* $\\text{Align}(\\cdot)$: 시간축 길이 정합을 위한 정렬 연산

**Skip 연결 (Skip Connection)**
$$S^{(l)} = \\text{Conv}_{\\text{skip}}^{(l)}(H^{(l)}), \quad \\text{Skip} = \\text{Skip} + S^{(l)}$$
* 각 블록의 정보를 누적하여 다중 스케일 특징 활용

### 3.4 그래프 합성곱 (Graph Convolution via MixProp)

시간 특징 $H$에 대해 그래프 전파를 수행하여 노드 간 상호작용을 반영합니다.

**정규화 전이행렬 (Normalized Transition Matrix)**
$$\\hat{A} = \\tilde{A} + I, \quad D = \\text{diag}(\\hat{A} \\cdot 1), \quad P = D^{-1}\\hat{A}$$

**정보 전파 및 혼합 (Propagation & Mixing)**
$$U_0 = H, \quad U_i = \\beta \\cdot H + (1-\\beta) \\cdot P \\cdot U_{i-1} \quad (i=1..g)$$
* $\\beta$: 원본 입력 유지 비율 (propalpha)
* $g$: 그래프 전파 단계 수

**특징 통합 (Aggregation)**
$$\\text{GCN}(H, \\tilde{A}) = \\text{MLP}([U_0; U_1; \\dots; U_g])$$
$$H' = \\text{GCN}(H, \\tilde{A}) + \\text{GCN}(H, \\tilde{A}^T)$$
* 방향성 그래프를 고려하기 위해 정방향($\\tilde{A}$)과 역방향($\\tilde{A}^T$) 전파를 결합

### 3.5 출력 및 다중-스텝 예측 (Output Head)

누적된 Skip 특징을 변환하여 최종 예측값을 출력합니다.

$$Y = \\text{Conv}_{\\text{end2}}(\\text{ReLU}(\\text{Conv}_{\\text{end1}}(\\text{ReLU}(\\text{Skip}))))$$

### 3.6 학습 목표 및 평가 (Training Objective & Evaluation)

**손실 함수 (Loss Function)**
$$\\mathcal{L}(\\theta) = \\sum_{h=1}^{H} ||\\hat{y}_{t+h} - y_{t+h}||_1$$
* L1 손실을 사용하여 이상치에 대한 민감도 완화

**평가 지표 (Metrics)**
$$\\text{RSE} = \\frac{\\sqrt{\\sum||\\hat{y}-y||_2^2}}{\\sqrt{\\sum||y-\\bar{y}||_2^2}}, \quad \\text{RAE} = \\frac{\\sum||\\hat{y}-y||_1}{\\sum||y-\\bar{y}||_1}$$
* $\\bar{y}$: 관측값의 평균 (Baseline)

### 3.7 불확실성 추정 (Uncertainty Estimation)

Monte-Carlo Dropout을 사용하여 예측의 불확실성을 추정합니다.

**샘플링 및 통계량 (Sampling & Statistics)**
$$\\hat{y}^{(s)} = f_\\theta^{\\text{drop}}(X, \\tilde{A}), \quad s=1..S$$
$$\\mu = \\frac{1}{S} \\sum_{s=1}^{S} \\hat{y}^{(s)}, \quad \\sigma^2 = \\frac{1}{S-1} \\sum_{s=1}^{S} (\\hat{y}^{(s)} - \\mu)^2$$

**신뢰 구간 (Confidence Interval)**
$$[q_{lo}, q_{hi}] = \\text{Quantile}(\\{\\hat{y}^{(s)}\\}, \\tau_{lo}, \\tau_{hi})$$

### 3.8 보조 지표 (Auxiliary Output: Change Index)

해석을 돕기 위한 보조 지표로 변화량을 산출합니다.

$$\\text{GAP}_{t+h} = \\hat{y}_{t+h} - y_t$$
* 마지막 관측값 $y_t$ 대비 예측 변화량
"""

with open('Methodology_Math_Formatted.md', 'w', encoding='utf-8') as f:
    f.write(content)
