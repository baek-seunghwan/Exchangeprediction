"""
train_test_with_reading_log.py

train_test.py를 실행하면서 모든 출력(stdout, stderr)을 
train_test.log 파일에 기록하는 wrapper 스크립트입니다.

사용법:
    python train_test_with_reading_log.py [train_test.py의 모든 인자]
    
예시:
    python train_test_with_reading_log.py --epochs 10 --lr 0.001
"""

import sys
import os
import subprocess
import datetime
import re

def update_integrated_fx_testing():
    """
    Testing 폴더의 개별 FX 파일들을 읽어 평균을 계산하고
    Integrated_FX_Testing.txt를 업데이트합니다.
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        testing_dir = os.path.join(script_dir, '..', 'model', 'Bayesian', 'Testing')
        
        fx_files = {
            'Cn_fx_Testing.txt': 'Cn_fx',
            'Jp_fx_Testing.txt': 'Jp_fx',
            'Kr_fx_Testing.txt': 'Kr_fx',
            'Uk_fx_Testing.txt': 'Uk_fx'
        }
        
        rse_values = []
        rae_values = []
        fx_details = []
        
        for fx_file, fx_name in fx_files.items():
            file_path = os.path.join(testing_dir, fx_file)
            if os.path.exists(file_path):
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    rse_match = re.search(r'rse:([0-9.]+)', content)
                    rae_match = re.search(r'rae:([0-9.]+)', content)
                    
                    if rse_match and rae_match:
                        rse_val = float(rse_match.group(1))
                        rae_val = float(rae_match.group(1))
                        rse_values.append(rse_val)
                        rae_values.append(rae_val)
                        fx_details.append((fx_name, rse_val, rae_val))
        
        if rse_values and rae_values:
            avg_rse = sum(rse_values) / len(rse_values)
            avg_rae = sum(rae_values) / len(rae_values)
            
            integrated_path = os.path.join(testing_dir, 'Average_of_each_call.txt')
            with open(integrated_path, 'w', encoding='utf-8') as f:
                f.write(f"rse:{avg_rse:.3f}\n")
                f.write(f"rae:{avg_rae:.3f}\n")
                f.write("\n")
                for fx_name, rse_val, rae_val in fx_details:
                    f.write(f"{fx_name}: rse:{rse_val:.3f}, rae:{rae_val:.3f}\n")
            
            print(f"[UPDATE] Integrated_FX_Testing.txt 업데이트: rse={avg_rse:.3f}, rae={avg_rae:.3f}")
    except Exception as e:
        print(f"[WARNING] Integrated_FX_Testing.txt 업데이트 실패: {e}")

def run_with_logging(run_count=1):
    """
    train_test.py를 실행하고 출력을 로그 파일에 기록합니다.
    run_count: 실행 횟수 (기본값: 1, 50으로 설정하면 50번 실행)
    """
    # 현재 스크립트의 디렉토리
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # train_test.py 경로
    train_test_path = os.path.join(script_dir, 'train_test_v1.01.py')
    
    # 로그 파일 경로 (train_test.log)
    log_file_path = os.path.join(script_dir, 'train_test.log')
    
    # 첫 번째 실행 시 로그 파일 초기화
    with open(log_file_path, 'w', encoding='utf-8') as f:
        f.write(f"로그 시작 시간: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")
    
    # train_test.py의 모든 인자 전달 (--runs과 그 값은 제외)
    remaining_args = []
    skip_next = False
    for i, arg in enumerate(sys.argv[1:]):
        if skip_next:
            skip_next = False
            continue
        if arg.startswith('--runs'):
            # --runs=10 형식 또는 --runs 10 형식 모두 처리
            if '=' not in arg:
                skip_next = True  # 다음 인자도 skip
            continue
        remaining_args.append(arg)
    
    print(f"총 {run_count}번 실행됩니다.")
    print(f"로그 파일: {log_file_path}")
    print("=" * 80)
    
    total_runs = run_count
    successful_runs = 0
    failed_runs = 0
    
    for run_num in range(1, run_count + 1):
        print(f"\n[{run_num}/{total_runs}] 실행 중...")
        
        cmd = [sys.executable, train_test_path] + remaining_args
        
        # 로그 파일 헤더 작성
        with open(log_file_path, 'a', encoding='utf-8') as log_file:
            log_file.write(f"\n{'='*80}\n")
            log_file.write(f"[{run_num}/{total_runs}] 시작 시간: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            log_file.write(f"명령어: {' '.join(cmd)}\n")
            log_file.write(f"{'='*80}\n\n")
        
        try:
            # train_test.py 실행 및 출력 기록
            with open(log_file_path, 'a', encoding='utf-8') as log_file:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=script_dir,
                    bufsize=1,
                    universal_newlines=True
                )
                
                # 실시간으로 출력하면서 로그에도 기록
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log_file.write(line)
                    log_file.flush()
                
                # 프로세스 종료 대기
                return_code = process.wait()
            
            # 로그 파일 종료 문구
            with open(log_file_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"\n{'='*80}\n")
                log_file.write(f"[{run_num}/{total_runs}] 종료 시간: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f"종료 코드: {return_code}\n")
                log_file.write(f"{'='*80}\n\n")
            
            if return_code == 0:
                successful_runs += 1
                print(f"[{run_num}/{total_runs}] 완료 (성공)")
                # 성공 시 Integrated_FX_Testing.txt 업데이트
                update_integrated_fx_testing()
            else:
                failed_runs += 1
                print(f"[{run_num}/{total_runs}] 완료 (실패: 종료 코드 {return_code})")
        
        except KeyboardInterrupt:
            print(f"\n[{run_num}/{total_runs}] 사용자가 중단했습니다.")
            with open(log_file_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"\n[중단됨] {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            return 1
        
        except Exception as e:
            failed_runs += 1
            print(f"[{run_num}/{total_runs}] 오류 발생: {e}")
            with open(log_file_path, 'a', encoding='utf-8') as log_file:
                log_file.write(f"\n[오류] {str(e)}\n")
    
    # 최종 결과
    print("\n" + "=" * 80)
    print(f"최종 결과: 총 {total_runs}회 (성공: {successful_runs}, 실패: {failed_runs})")
    print(f"로그 파일: {log_file_path}")
    print("=" * 80)
    
    return 0 if failed_runs == 0 else 1
    
    # 최종 결과
    print("\n" + "=" * 80)
    print(f"최종 결과: 총 {total_runs}회 (성공: {successful_runs}, 실패: {failed_runs})")
    print(f"로그 파일: {log_file_path}")
    print("=" * 80)
    
    return 0 if failed_runs == 0 else 1

if __name__ == "__main__":
    # --runs 옵션으로 실행 횟수 지정 (기본값: 1)
    run_count = 1
    
    for i, arg in enumerate(sys.argv[1:]):
        if arg.startswith('--runs'):
            # --runs=10 형식
            if '=' in arg:
                try:
                    run_count = int(arg.split('=')[1])
                except (ValueError, IndexError):
                    print(f"경고: --runs={arg.split('=')[1]} 값이 올바르지 않습니다. 기본값 1을 사용합니다.")
                    run_count = 1
            # --runs 10 형식
            elif i + 1 < len(sys.argv[1:]):
                try:
                    run_count = int(sys.argv[i + 2])
                except (ValueError, IndexError):
                    print(f"경고: --runs 다음의 값이 숫자가 아닙니다. 기본값 1을 사용합니다.")
                    run_count = 1
            break
    
    exit_code = run_with_logging(run_count)
    sys.exit(exit_code)