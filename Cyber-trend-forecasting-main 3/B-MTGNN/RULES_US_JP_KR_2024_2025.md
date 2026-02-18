# B-MTGNN 운영 규칙 (US/JP/KR, 2024 Validation, 2025 Testing)

이 문서는 환율 실험 운영 시 반드시 적용할 규칙을 정의합니다.

## 1) graph.csv 사용 규칙
- `graph.csv` 원본 파일은 유지한다.
- 학습 시 그래프 로딩 단계에서 **중국/영국 환율 노드(`cn_fx`, `uk_fx`)를 자동 제외**한다.
- 즉, 그래프 연결은 미국/일본/한국 환율 + 보조 데이터 중심으로 구성된다.

적용 코드:
- `util.py`의 `build_predefined_adj()`
- `util.py`의 `is_excluded_fx_node()`

## 2) Validation 기간 고정
- Validation 대상 기간은 **2024-01 ~ 2024-12**로 고정한다.
- `Date` 컬럼이 있으면 해당 연도로 정확히 분할한다.
- `Date` 컬럼이 없으면 fallback으로 마지막 24개월 중 앞 12개월을 Validation으로 사용한다.

## 3) Testing 기간 고정
- Testing 대상 기간은 **2025-01 ~ 2025-12**로 고정한다.
- `Date` 컬럼이 있으면 해당 연도로 정확히 분할한다.
- `Date` 컬럼이 없으면 fallback으로 마지막 12개월을 Testing으로 사용한다.

적용 코드:
- `util.py`의 `DataLoaderS(..., fixed_eval_periods, valid_year, test_year)`
- `util.py`의 `_resolve_split_points()`

## 4) Validation/Testing 예측 규칙
- Validation/Testing 그래프 생성 시, 모델이 실제값을 따라가지 않도록 한다.
- 즉, 평가 시 롤아웃은 **강제 recursive**로 사용한다.
- teacher forcing 기반 평가 갱신은 허용하지 않는다.

적용 코드:
- `train_test.py`의 `evaluate_sliding_window()`
- 인자: `--force_recursive_eval 1`

---

## 권장 실행 예시

### 단일 학습/평가
```bash
python train_test.py \
  --data data/sm_data.csv \
  --target_profile run001_us \
  --fixed_eval_periods 1 \
  --valid_year 2024 \
  --test_year 2025 \
  --force_recursive_eval 1 \
  --rollout_mode recursive
```

### 자동 튜닝
```bash
python auto_tune_rse.py \
  --datasets sm_data.csv \
  --fixed_eval_periods 1 \
  --valid_year 2024 \
  --test_year 2025 \
  --force_recursive_eval 1 \
  --rollout_mode recursive
```

---

## 참고
- `Date` 컬럼이 데이터 파일에 있으면 연도 기준 분할이 정확히 적용된다.
- `Date` 컬럼이 없는 경우, 월별 정렬 데이터라는 가정 하에 fallback 분할이 적용된다.
- 규칙 변경이 필요하면 `train_test.py` 인자와 `util.py` 분할 함수를 함께 수정한다.
