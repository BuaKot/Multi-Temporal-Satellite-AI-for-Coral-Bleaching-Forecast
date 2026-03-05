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
start_date = '2026-01-01'
end_date = '2026-02-01'

print("Fetching 2022 SST data from NOAA OISST...")

def get_oisst_data(image):
    stats = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=poi,
        scale=27830
    )
    return ee.Feature(None, {
        'date': image.date().format('yyyy-MM-dd'),
        'sst': stats.get('sst'),
        'anom': stats.get('anom'),   # SST anomaly (ใช้คำนวณ DHW)
    })

col = ee.ImageCollection('NOAA/CDR/OISST/V2_1') \
        .filterDate(start_date, end_date)

count = col.size().getInfo()
print(f"Found {count} images")

nested_list = col.map(get_oisst_data).getInfo()
data_list = [f['properties'] for f in nested_list['features']]
df = pd.DataFrame(data_list)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

# ── คำนวณ DHW จาก anomaly ──────────────────────────────
# DHW = สะสม anomaly ที่ > 1°C ย้อนหลัง 84 วัน (12 สัปดาห์)
# standard ของ NOAA: Hotspot = max(anom - 1, 0), DHW = sum(Hotspot)/7
df['hotspot'] = df['anom'].clip(lower=0).subtract(1) if hasattr(df['anom'], 'subtract') else (df['anom'] - 1).clip(lower=0)
df['dhw'] = df['hotspot'].rolling(window=84, min_periods=1).sum() / 7

# ── SST unit check ─────────────────────────────────────
# OISST เก็บเป็น °C * 100 ในบางเวอร์ชัน
if df['sst'].max() > 100:
    df['sst'] = df['sst'] / 100.0

print(f"\nSST range: {df['sst'].min():.2f} - {df['sst'].max():.2f} degC")
print(f"DHW range: {df['dhw'].min():.2f} - {df['dhw'].max():.2f}")

if not os.path.exists('CSV Files'):
    os.makedirs('CSV Files')

output_file = 'CSV Files/sst_dhw_2022.csv'
df[['date', 'sst', 'anom', 'dhw']].to_csv(output_file, index=False)

print(f"\nSaved {len(df)} rows to {output_file}")
print(df[['date', 'sst', 'anom', 'dhw']].head(10).to_string())