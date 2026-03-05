import ee
import pandas as pd
import os

# 1. Initialize Google Earth Engine
try:
    ee.Initialize(project='macbf-project2')
except Exception as e:
    ee.Authenticate()
    ee.Initialize(project='macbf-project2')

# 2. Define Point of Interest (Koh Man Nai) - [Long, Lat]
poi = ee.Geometry.Point([101.65, 12.60]) 

# 3. Function to extract Mean Reflectance
def get_s2_data(image):
    stats = image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=poi,
        scale=10
    )
    return ee.Feature(None, {
        'date': image.date().format('yyyy-MM-dd'),
        'blue': stats.get('B2'),
        'green': stats.get('B3'),
        'red': stats.get('B4'),
        'nir': stats.get('B8'),
        'cloud': image.get('CLOUDY_PIXEL_PERCENTAGE')
    })

# 4. Fetch 3 years of data (2023 - 2025)
print("Fetching data from Google Earth Engine (2023-2025)...")
s2_col = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
          .filterBounds(poi)
          .filterDate('2022-01-01', '2022-12-31')
          .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)))

nested_list = s2_col.map(get_s2_data).getInfo()

# 5. Process Sentinel-2 DataFrame
data_list = [feature['properties'] for feature in nested_list['features']]
df_s2 = pd.DataFrame(data_list)

# Scale values to 0-1 range
cols_to_scale = ['blue', 'green', 'red', 'nir']
df_s2[cols_to_scale] = df_s2[cols_to_scale] / 10000.0

# 6. Save Raw Sentinel-2 Data
if not os.path.exists('CSV Files'):
    os.makedirs('CSV Files')

s2_output_path = 'CSV Files/sentinel2_data_with_nir.csv'
df_s2.to_csv(s2_output_path, index=False)
print(f"Sentinel-2 data saved to: {s2_output_path}")

# --- Data Merging Section ---

print("Starting Data Integration (Merging)...")
try:
    # Load Ocean data (SST + DHW)
    ocean_path = 'CSV Files/sst_dhw_2022.csv'
    df_ocean = pd.read_csv(ocean_path)
    
    # Standardize Column Names (Removes spaces and converts to lowercase)
    df_ocean.columns = df_ocean.columns.str.strip().str.lower()
    df_s2.columns = df_s2.columns.str.strip().str.lower()

    # Ensure 'date' column exists after cleaning
    if 'date' in df_ocean.columns:
        # Standardize Date Format to YYYY-MM-DD for perfect matching
        df_ocean['date'] = pd.to_datetime(df_ocean['date']).dt.strftime('%Y-%m-%d')
        df_s2['date'] = pd.to_datetime(df_s2['date']).dt.strftime('%Y-%m-%d')

        # Perform Left Join (Keep all days from Ocean data)
        cols_to_merge = ['date', 'blue', 'green', 'red', 'nir']
        df_final = pd.merge(df_ocean, df_s2[cols_to_merge], on='date', how='left')

        # Save Final Master Table
        master_path = 'CSV Files/master_data_2022_SST_DHW_RGBNIR.csv'
        df_final.to_csv(master_path, index=False)
        
        print(f"Success! Master table created: {master_path}")
        print("-" * 50)
        print(df_final.head())
    else:
        print(f"Error: 'date' column not found. Available columns: {df_ocean.columns.tolist()}")

except FileNotFoundError:
    print(f"Error: Could not find {ocean_path}. Please check the filename.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")