import time
import os
import re
from datetime import datetime

log_file = "full_training.log"
last_size = 0
last_epoch = 0

print("=" * 70)
print("📊 Training Progress Monitor")
print("=" * 70)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Log file: {log_file}\n")

while True:
    if os.path.exists(log_file):
        current_size = os.path.getsize(log_file)
        
        if current_size > last_size:
            # 파일이 업데이트됨 - 마지막 내용 읽기
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            # 에포크 정보 추출
            for line in reversed(lines):
                if "end of epoch" in line:
                    # Parse epoch info
                    match = re.search(r'end of epoch\s+(\d+)', line)
                    if match:
                        epoch = int(match.group(1))
                        if epoch > last_epoch:
                            last_epoch = epoch
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] Epoch {epoch}/200 completed")
                            
                            # Extract metrics
                            if "valid rse" in line:
                                print(f"  → {line.strip()}")
                    break
            
            last_size = current_size
    
    # Check if training is complete
    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    if "Training complete" in content or last_epoch >= 200:
        print("\n" + "=" * 70)
        print("✅ Training Complete!")
        print("=" * 70)
        break
    
    time.sleep(30)  # Check every 30 seconds

