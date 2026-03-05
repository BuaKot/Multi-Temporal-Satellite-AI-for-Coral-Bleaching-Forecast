import ee
import pandas as pd
import numpy as np
import os

try:
    ee.Initialize(project='macbf-project2')
except:
    ee.Authenticate()
    ee.Initialize(project='macbf-project2')

poi = ee.Geometry.Point([101.65, 12.60])

# ─────────────────────────────────────────
# 1. FETCH SST + ANOM (OISST)
# ─────────────────────────────────────────
print("Fetching SST data...")

def get_oisst(image):
    stats = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=poi,
        scale=27830
    )
    return ee.Feature(None, {
        'date': image.date().format('yyyy-MM-dd'),
        'sst':  stats.get('sst'),
        'anom': stats.get('anom'),
    })

oisst_col = ee.ImageCollection('NOAA/CDR/OISST/V2_1') \
              .filterDate('2022-01-01', '2023-01-01')

df_sst = pd.DataFrame([
    f['properties']
    for f in oisst_col.map(get_oisst).getInfo()['features']
])
df_sst['date'] = pd.to_datetime(df_sst['date'])
df_sst = df_sst.sort_values('date').drop_duplicates('date').reset_index(drop=True)

# Fix: anom เป็น °C × 100 → หาร 100
df_sst['sst']  = df_sst['sst']  / 100.0
df_sst['anom'] = df_sst['anom'] / 100.0

# คำนวณ DHW ที่ถูกต้อง
df_sst['hotspot'] = (df_sst['anom'] - 1.0).clip(lower=0)
df_sst['dhw']     = df_sst['hotspot'].rolling(window=84, min_periods=1).sum() / 7.0
df_sst = df_sst.drop(columns='hotspot')

print(f"SST: {len(df_sst)} days | "
      f"SST {df_sst['sst'].min():.1f}-{df_sst['sst'].max():.1f}C | "
      f"DHW {df_sst['dhw'].min():.2f}-{df_sst['dhw'].max():.2f}")

# ─────────────────────────────────────────
# 2. FETCH SENTINEL-2
# ─────────────────────────────────────────
print("Fetching Sentinel-2 data...")

def get_s2(image):
    stats = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=poi,
        scale=10
    )
    return ee.Feature(None, {
        'date':  image.date().format('yyyy-MM-dd'),
        'blue':  stats.get('B2'),
        'green': stats.get('B3'),
        'red':   stats.get('B4'),
        'nir':   stats.get('B8'),
        'cloud': image.get('CLOUDY_PIXEL_PERCENTAGE')
    })

s2_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(poi)
          .filterDate('2022-01-01', '2023-01-01')
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)))

df_s2 = pd.DataFrame([
    f['properties']
    for f in s2_col.map(get_s2).getInfo()['features']
])
df_s2['date'] = pd.to_datetime(df_s2['date'])

# Scale + clean
for col in ['blue', 'green', 'red', 'nir']:
    df_s2[col] = df_s2[col] / 10000.0

# ลบ outlier (เมฆ/glint)
df_s2 = df_s2[df_s2['blue'] < 0.3]

# Dedup: เก็บแถว cloud น้อยสุดต่อวัน
df_s2 = (df_s2.sort_values('cloud')
              .drop_duplicates('date', keep='first')
              .reset_index(drop=True))

print(f"S2: {len(df_s2)} days with valid imagery")

# ─────────────────────────────────────────
# 3. MERGE
# ─────────────────────────────────────────
df_final = pd.merge(
    df_sst,
    df_s2[['date', 'blue', 'green', 'red', 'nir']],
    on='date',
    how='left'
)

# ─────────────────────────────────────────
# 4. SAVE
# ─────────────────────────────────────────
if not os.path.exists('CSV Files'):
    os.makedirs('CSV Files')

df_final.to_csv('CSV Files/master_data_2022.csv', index=False)

print(f"\nSaved: {len(df_final)} rows")
print(f"Days with S2: {df_final['blue'].notna().sum()}")
print(df_final[['date','sst','anom','dhw','blue','green']].head(10).to_string())