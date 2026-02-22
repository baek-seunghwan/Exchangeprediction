import pandas as pd

df = pd.read_csv('data/sm_data.csv')
n = len(df)
train_end = int(0.70 * n)
valid_end = int(0.85 * n)
input_start = max(0, valid_end - 24)

print(f'Total rows: {n}')
print(f'Train: 0-{train_end-1} ({df.iloc[0,0]} ~ {df.iloc[train_end-1,0]})')
print(f'Valid: {train_end}-{valid_end-1} ({df.iloc[train_end,0]} ~ {df.iloc[valid_end-1,0]})')
print(f'Test window input: {input_start}-{input_start+23} ({df.iloc[input_start,0]} ~ {df.iloc[input_start+23,0]})')
print(f'Test forecast: {input_start+24}-{min(input_start+35,n-1)} ({df.iloc[input_start+24,0]} ~ {df.iloc[min(input_start+35,n-1),0]})')
print()

cols = ['us_Trade Weighted Dollar Index', 'jp_fx', 'kr_fx']
forecast_rows = df.iloc[input_start+24:input_start+36]
for c in cols:
    vals = forecast_rows[c].values
    print(f'{c}:')
    print(f'  Values: {[round(float(v),1) for v in vals]}')
    print(f'  Range: {vals.min():.1f} ~ {vals.max():.1f} (span={float(vals.max()-vals.min()):.1f})')
    print(f'  Direction: {float(vals[-1]-vals[0]):+.1f}')
    ss_total = sum((float(v) - float(vals.mean()))**2 for v in vals)
    print(f'  SS_total: {ss_total:.1f}')
    print()
