import ee
import pandas as pd

# 1. Connect System
try:
    ee.Initialize(project='macbf-project2')
    print("System connected...")
except Exception as e:
    print("Connection Error:", e)

# 2. Coordinates: Koh Man Nai
target_lat = 12.614167
target_lon = 101.683611
poi = ee.Geometry.Point(target_lon, target_lat)

# 3. Date Range: Year 2023
start_date = '2023-01-01'
end_date = '2023-12-31'

print(f"Fetching SST & Anomaly for: {target_lat}, {target_lon}")
print("Dataset: NOAA OISST V2.1")

# 4. Function
def get_sst_anomaly(image):
    stats = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=poi,
        scale=5000, 
        bestEffort=True
    )
    return image.set('date', image.date().format('YYYY-MM-dd')) \
                .set('sst', stats.get('sst')) \
                .set('anom', stats.get('anom'))

# 5. Load Data
dataset = ee.ImageCollection('NOAA/CDR/OISST/V2_1') \
            .filterDate(start_date, end_date) \
            .map(get_sst_anomaly)

# 6. Process
try:
    data_list = dataset.reduceColumns(
        ee.Reducer.toList(3), ['date', 'sst', 'anom']
    ).values().get(0).getInfo()

    df = pd.DataFrame(data_list, columns=['Date', 'SST_Raw', 'Anom_Raw'])

    if df['SST_Raw'].isnull().all():
        print("WARNING: No data found.")
    else:
        df = df.dropna()

        # Convert Unit (x 0.01)
        df['SST_Celsius'] = df['SST_Raw'] * 0.01
        df['SST_Anomaly'] = df['Anom_Raw'] * 0.01

        # Show Result
        print("\n--- Result (First 5 days) ---")
        print(df[['Date', 'SST_Celsius', 'SST_Anomaly']].head())
        print(f"\nTotal days collected: {len(df)}")
        
        # Safe Print (No degree symbol)
        max_anom = df['SST_Anomaly'].max()
        print(f"Max Anomaly: +{max_anom:.2f} deg C") # Changed symbol to text

        # Save to CSV
        filename = 'sst_anomaly_koh_man_nai.csv'
        df[['Date', 'SST_Celsius', 'SST_Anomaly']].to_csv(filename, index=False)
        print(f"SUCCESS: File saved as {filename}")

except Exception as e:
    print("Error:", e)