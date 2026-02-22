#!/bin/bash
# ============================================================
# run_reproduce.sh — B-MTGNN 최종 결과 재현 스크립트
# ============================================================
# 이 스크립트는 TRIPLE050 프로파일(기본값)로 학습 + 평가를 수행합니다.
# 동일 환경에서 동일한 결과를 재현하는 것이 목적입니다.
#
# 기준 성능 (2025 테스트 구간):
#   us_Trade Weighted Dollar Index  RSE ≤ 0.40
#   kr_fx                           RSE ≤ 0.29
#   jp_fx                           RSE ≤ 0.27
#   (모두 0.5 이하)
#
# 사용법:
#   bash run_reproduce.sh              # 기본 실행 (전체 학습 + 평가 + 리포트)
#   bash run_reproduce.sh --quick      # 빠른 확인 (plot 없이, autotune_mode)
#   bash run_reproduce.sh --focus-only # focus 노드 그래프만 생성
#   bash run_reproduce.sh --extra "Corr,spread" # 추가 노드 그래프
# ============================================================

set -euo pipefail
cd "$(dirname "$0")"

PYTHON="${PYTHON:-/Users/samrobert/Documents/GitHub/Exchangeprediction/.venv/bin/python}"
LOGFILE="reproduce_$(date +%Y%m%d_%H%M%S).log"

# Parse optional arguments
EXTRA_ARGS=""
for arg in "$@"; do
    case "$arg" in
        --quick)
            EXTRA_ARGS="$EXTRA_ARGS --autotune_mode 1 --plot 0 --generate_final_report 0"
            ;;
        --focus-only)
            EXTRA_ARGS="$EXTRA_ARGS --plot_focus_only 1"
            ;;
        --extra)
            shift
            EXTRA_ARGS="$EXTRA_ARGS --report_extra_nodes $1"
            ;;
    esac
done

echo "=============================="
echo "B-MTGNN Reproduce Run"
echo "Time: $(date)"
echo "Python: $PYTHON"
echo "Log: $LOGFILE"
echo "Extra args: $EXTRA_ARGS"
echo "=============================="

# TRIPLE050 is the default profile, so no extra profile arg needed.
# All settings (seed=1, epochs=180, eval_last, l1 loss, etc.) are baked into the profile.
$PYTHON train_test.py \
    $EXTRA_ARGS \
    2>&1 | tee "$LOGFILE"

echo ""
echo "=============================="
echo "Run complete. Log saved to: $LOGFILE"
echo "Output files:"
echo "  - AXIS/model/Bayesian/Testing/*.png  (per-node plots)"
echo "  - AXIS/model/Bayesian/Testing/*.txt  (per-node metrics)"
echo "  - final_forecast_results.png         (combined 3-target plot)"
echo "  - final_summary_table.png            (summary table)"
echo "=============================="

# Verify expected output files exist
for f in "final_forecast_results.png" "final_summary_table.png"; do
    if [ -f "$f" ]; then
        echo "  [OK] $f exists"
    else
        echo "  [MISSING] $f"
    fi
done
