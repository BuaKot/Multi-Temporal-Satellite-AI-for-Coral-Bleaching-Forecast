import pandas as pd

# 1. Read the CSV file
filename = 'sst_anomaly_koh_man_nai.csv'

try:
    df = pd.read_csv(filename)
    df['Date'] = pd.to_datetime(df['Date']) 
    
    print(f"Data loaded successfully: {len(df)} days")

    # ==========================================
    # 2. Parameters
    # ==========================================
    # MMM (Maximum Monthly Mean) for Koh Man Nai
    MMM = 29.5 # degrees Celsius
    
    # ==========================================
    # 3. Calculate DHW
    # ==========================================
    
    # 3.1 Calculate HotSpot
    df['HotSpot'] = df['SST_Celsius'].apply(lambda x: x - MMM if x > MMM else 0)

    # 3.2 Filter HotSpot (Only count if HotSpot >= 1.0 deg C)
    df['HotSpot_Filtered'] = df['HotSpot'].apply(lambda x: x if x >= 1.0 else 0)

    # 3.3 Calculate DHW (Rolling Sum 84 days / 7)
    df['DHW'] = df['HotSpot_Filtered'].rolling(window=84, min_periods=1).sum() / 7

    # ==========================================
    # 4. Result Summary
    # ==========================================
    max_dhw = df['DHW'].max()
    print("-" * 30)
    print(f"Summary for 2023:")
    print(f"Max DHW: {max_dhw:.2f} deg C-weeks")
    
    if max_dhw >= 8:
        print("Status: ALERT LEVEL 2 (Severe Bleaching / Mortality Risk)")
    elif max_dhw >= 4:
        print("Status: ALERT LEVEL 1 (Bleaching Risk)")
    else:
        print("Status: NO STRESS / WATCH")
    print("-" * 30)
    
    # Save the file
    output_filename = 'sst_dhw_calculated.csv'
    df.to_csv(output_filename, index=False)
    print(f"SUCCESS: File saved as {output_filename}")

except FileNotFoundError:
    print(f"Error: File {filename} not found.")
except Exception as e:
    print(f"Error: {e}")