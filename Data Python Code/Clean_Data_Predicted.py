import pandas as pd
import numpy as np
import sys
sys.stdout.reconfigure(encoding='utf-8')

df = pd.read_csv('CSV Files/master_data_3years_SST_DHW_RGBNIR.csv')
df['date'] = pd.to_datetime(df['date'])

print(f"Original: {len(df)} rows")

# ─── 1. ลบ Outliers (reflectance > 0.3 = เมฆ/sun glint) ───
has_bands = df['blue'].notna()
df = df[~(has_bands & (df['blue'] > 0.3))]
print(f"After outlier removal: {len(df)} rows")

# ─── 2. SST: ทุกวันมี 2 แถว (2 SST sources) → เก็บ max DHW, mean SST ───
agg_sst = df.groupby('date').agg(
    sst_celsius=('sst_celsius', 'mean'),
    dhw=('dhw', 'max')
).reset_index()

# ─── 3. Sentinel: เก็บแถวที่ cloud น้อยที่สุด (ต่อวัน) ───
s2_rows = df[df['blue'].notna()].copy()
s2_clean = (s2_rows
    .sort_values('blue')  # proxy: ค่า blue ต่ำ = น้ำใส = cloud น้อย
    .drop_duplicates(subset='date', keep='first')
    [['date', 'blue', 'green', 'red', 'nir']])

# ─── 4. Merge กลับ ───
df_clean = pd.merge(agg_sst, s2_clean, on='date', how='left')
df_clean = df_clean.sort_values('date').reset_index(drop=True)

# ─── 5. คำนวณ Indices ───
df_clean['ndvi'] = ((df_clean['nir'] - df_clean['red']) /
                    (df_clean['nir'] + df_clean['red'])).round(4)

df_clean['ndwi'] = ((df_clean['green'] - df_clean['nir']) /
                    (df_clean['green'] + df_clean['nir'])).round(4)

# ─── 6. สรุปผล ───
total = len(df_clean)
has_s2 = df_clean['blue'].notna().sum()
print(f"\nAfter cleaning: {total} days total")
print(f"Days with Sentinel-2: {has_s2} ({has_s2/total*100:.1f}%)")
print(f"Days without Sentinel-2 (NaN): {total - has_s2}")
print(f"\nSST range: {df_clean['sst_celsius'].min():.1f} – {df_clean['sst_celsius'].max():.1f} °C")
print(f"DHW range: {df_clean['dhw'].min():.1f} – {df_clean['dhw'].max():.1f}")
print(f"\nSample:")
print(df_clean[df_clean['blue'].notna()].head(5).to_string())

df_clean.to_csv('CSV Files/master_clean.csv', index=False)
print(f"\nSaved to CSV Files/master_clean.csv")