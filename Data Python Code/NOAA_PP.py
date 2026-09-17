import ee
import pandas as pd

# Connect System
try:
    ee.Initialize(project='macbf-project2')
    print("System connected...")
except Exception as e:
    print("Connection Error:", e)

# Coordinates: Koh Phi Phi (เกาะพีพี จังหวัดกระบี่)
target_lat = 7.7401
target_lon = 98.7784
poi = ee.Geometry.Point(target_lon, target_lat)

# Date Range: Year 2023-2026
start_date = '2023-01-01'
end_date = '2026-12-31'

print(f"Fetching SST & Anomaly for Koh Phi Phi: {target_lat}, {target_lon}")
print("Dataset: NOAA OISST V2.1")

# Function to extract SST and Anomaly
def get_sst_anomaly(image):
    stats = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=poi,
        scale=27830,  # ปรับ Scale ให้ตรงกับ OISST Resolution (0.25 deg)
        bestEffort=True
    )
    return image.set('date', image.date().format('YYYY-MM-dd')) \
                .set('sst', stats.get('sst')) \
                .set('anom', stats.get('anom'))

# Load Data from Earth Engine
dataset = ee.ImageCollection('NOAA/CDR/OISST/V2_1') \
            .filterDate(start_date, end_date) \
            .map(get_sst_anomaly)

# Process Data
try:
    data_list = dataset.reduceColumns(
        ee.Reducer.toList(3), ['date', 'sst', 'anom']
    ).values().get(0).getInfo()

    df = pd.DataFrame(data_list, columns=['Date', 'SST_Raw', 'Anom_Raw'])

    if df.empty or df['SST_Raw'].isnull().all():
        print("WARNING: No data found for the selected area or date range.")
    else:
        df = df.dropna()

        # เช็ค Scale Factor: หากค่าเกิน 100 แสดงว่าต้องคูณ 0.01 ปรับหน่วยเป็น C
        if df['SST_Raw'].iloc[0] > 100:
            df['SST_Celsius'] = df['SST_Raw'] * 0.01
            df['SST_Anomaly'] = df['Anom_Raw'] * 0.01
        else:
            df['SST_Celsius'] = df['SST_Raw']
            df['SST_Anomaly'] = df['Anom_Raw']

        # Show Result
        print("\n--- Result for Koh Phi Phi (First 5 days) ---")
        print(df[['Date', 'SST_Celsius', 'SST_Anomaly']].head())
        print(f"\nTotal days collected: {len(df)}")
        
        max_anom = df['SST_Anomaly'].max()
        print(f"Max Anomaly: +{max_anom:.2f} deg C")

        # Save to CSV
        filename = 'sst_anomaly_koh_phi_phi_2023_2026.csv'
        df[['Date', 'SST_Celsius', 'SST_Anomaly']].to_csv(filename, index=False) 
        print(f"SUCCESS: File saved as {filename}")

except Exception as e:
    print("Error processing data:", e)