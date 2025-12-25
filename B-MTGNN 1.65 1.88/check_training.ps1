# 학습 진행 상황 체크 스크립트
param(
    [int]$lines = 30
)

$logfile = "full_training.log"
if (Test-Path $logfile) {
    $content = Get-Content $logfile -Tail $lines
    Write-Host "=== 최근 $lines 줄 ===" -ForegroundColor Green
    $content
    Write-Host ""
    
    # 에포크 수 추출
    $epoch_lines = $content | Select-String "end of epoch" | Select-Object -Last 1
    if ($epoch_lines) {
        Write-Host "상태: $epoch_lines" -ForegroundColor Cyan
    }
    
    Write-Host "[업데이트: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')]" -ForegroundColor Yellow
} else {
    Write-Host "로그 파일을 찾을 수 없습니다." -ForegroundColor Red
}
