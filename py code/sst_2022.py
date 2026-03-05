import ee
import pandas as pd
import os

# 1. Initialize
try:
    ee.Initialize(project='macbf-project2')
except:
    ee.Authenticate()
    ee.Initialize(project='macbf-project2')

poi = ee.Geometry.Point([101.65, 12.60])
start_date = '2022-01-01'
end_date = '2023-01-01'

print("Fetching 2022 Ocean Data (Global 5km Dataset)...")

# --- PLAN B: ใช้ ID ที่เสถียรที่สุดใน Catalog ของ GEE ---
dataset_id = 'NOAA/CORALREEFWATCH/V31/5KM_DAILY' 

# หาก ID ข้างบนยังไม่ได้ ให้ลองเปลี่ยนเป็น ID ของ NOAA โดยตรงด้านล่างนี้:
# dataset_id = 'NASA/OCEANDATA/MODIS-Aqua/L3SMI' (กรณีใช้ MODIS แทน)

def get_noaa_data(image):
    stats = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=poi,
        scale=5000
    )
    # เราใช้ชื่อ band ที่เป็นสากล: 'sea_surface_temperature' และ 'degree_heating_weeks'
    return ee.Feature(None, {
        'date': image.date().format('yyyy-MM-dd'),
        'sst': stats.get('sea_surface_temperature'),
        'dhw': stats.get('degree_heating_weeks')
    })

try:
    # ดึงข้อมูล
    noaa_col = ee.ImageCollection(dataset_id).filterBounds(poi).filterDate(start_date, end_date)
    
    # เช็คจำนวนรูปก่อน (ถ้า 0 แสดงว่าเข้าถึงไม่ได้จริงๆ)
    count = noaa_col.size().getInfo()
    if count == 0:
        print("ยังไม่พบข้อมูลจาก NOAA V31... กำลังเปลี่ยนไปใช้ข้อมูล MODIS (NASA) เพื่อให้ได้ค่า SST แทน...")
        # แผนสำรองสุดท้าย: ใช้ MODIS สำหรับ SST
        noaa_col = ee.ImageCollection('NASA/OCEANDATA/MODIS-Aqua/L3SMI').filterDate(start_date, end_date).select('sst')
        
        def get_modis_sst(image):
            stats = image.reduceRegion(reducer=ee.Reducer.mean(), geometry=poi, scale=4638)
            return ee.Feature(None, {
                'date': image.date().format('yyyy-MM-dd'),
                'sst': stats.get('sst'),
                'dhw': 0 # MODIS ไม่มีค่า DHW ให้ใส่ 0 ไว้ก่อน
            })
        nested_list = noaa_col.map(get_modis_sst).getInfo()
    else:
        nested_list = noaa_col.map(get_noaa_data).getInfo()

    # 5. แปลงและบันทึก
    data_list = [feature['properties'] for feature in nested_list['features']]
    df_ocean = pd.DataFrame(data_list)
    
    if not os.path.exists('CSV Files'): os.makedirs('CSV Files')
    output_file = 'CSV Files/sst_dhw_2022.csv'
    df_ocean.to_csv(output_file, index=False)

    print(f"Successfully saved 2022 data! (Total {len(df_ocean)} days)")
    print(df_ocean.head())

except Exception as e:
    print(f"Error: {e}")
    print("แนะนำ: เข้าไปที่ https://code.earthengine.google.com/ แล้วค้นหา 'NOAA Coral Reef Watch' ")
    print("เพื่อดูว่าใน Account ของคุณเห็นชื่อ Asset เป็นอะไรครับ")