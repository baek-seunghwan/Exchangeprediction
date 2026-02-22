import re

# Parse jp_fx sweep results
results = []
current_config = None

with open('jp_fx_sweep_results.txt', 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    line = line.strip()
    
    # Detect run start
    if line.startswith('>>> jp_fx'):
        current_config = line
        continue
    
    # Detect eval_last_epoch result
    if 'final test rse' in line and current_config:
        m = re.search(r'final test rse\s+([\d.]+)', line)
        if m:
            final_rse = float(m.group(1))
            results.append({
                'config': current_config,
                'final_test_rse': final_rse,
                'best_focus_rrse': None,
                'best_focus_epoch': None
            })
    
    # Detect focus_rrse in test output
    if 'test focus_rrse' in line and current_config:
        m = re.search(r'test focus_rrse\s+([\d.]+)', line)
        if m:
            frrse = float(m.group(1))
            # Find which epoch this is
            epoch_match = None
            for j in range(max(0, i-5), i):
                em = re.search(r'end of epoch\s+(\d+)', lines[j])
                if em:
                    epoch_match = int(em.group(1))
            
            if results and results[-1]['config'] == current_config:
                if results[-1]['best_focus_rrse'] is None or frrse < results[-1]['best_focus_rrse']:
                    results[-1]['best_focus_rrse'] = frrse
                    results[-1]['best_focus_epoch'] = epoch_match
            else:
                results.append({
                    'config': current_config,
                    'final_test_rse': None,
                    'best_focus_rrse': frrse,
                    'best_focus_epoch': epoch_match
                })

# Combine entries for same config
combined = {}
for r in results:
    cfg = r['config']
    if cfg not in combined:
        combined[cfg] = r
    else:
        if r['final_test_rse'] is not None:
            combined[cfg]['final_test_rse'] = r['final_test_rse']
        if r['best_focus_rrse'] is not None:
            if combined[cfg]['best_focus_rrse'] is None or r['best_focus_rrse'] < combined[cfg]['best_focus_rrse']:
                combined[cfg]['best_focus_rrse'] = r['best_focus_rrse']
                combined[cfg]['best_focus_epoch'] = r['best_focus_epoch']

# Sort by best focus_rrse
sorted_results = sorted(combined.values(), key=lambda x: x['best_focus_rrse'] if x['best_focus_rrse'] else 999)

print("=" * 100)
print("TOP 30 by best focus_rrse (MAX of target RSEs during training epochs)")
print("=" * 100)
print(f"{'Config':<65} {'focus_rrse':>10} {'ep':>4} {'final_rse':>10}")
print("-" * 95)
for r in sorted_results[:30]:
    cfg = r['config'][4:65]  # trim ">>> "
    frrse = f"{r['best_focus_rrse']:.4f}" if r['best_focus_rrse'] else "N/A"
    ep = str(r['best_focus_epoch']) if r['best_focus_epoch'] else "?"
    final = f"{r['final_test_rse']:.4f}" if r['final_test_rse'] else "N/A"
    print(f"{cfg:<65} {frrse:>10} {ep:>4} {final:>10}")

# Sub-0.5 results
sub05 = [r for r in sorted_results if r['best_focus_rrse'] and r['best_focus_rrse'] < 0.5]
print(f"\n\n=== RESULTS WITH focus_rrse < 0.5 ({len(sub05)} found) ===")
for r in sub05:
    print(f"  {r['config']}")
    print(f"    focus_rrse={r['best_focus_rrse']:.4f} at ep={r['best_focus_epoch']}, final_test_rse={r['final_test_rse']:.4f if r['final_test_rse'] else 'N/A'}")
