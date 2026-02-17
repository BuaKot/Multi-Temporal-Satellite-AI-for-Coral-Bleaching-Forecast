import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # ใช้โหมดนี้เพื่อป้องกัน Error เรื่องไอคอนหาย
import matplotlib.pyplot as plt

# ==========================================
# PART 1: ฟังก์ชันเตรียมข้อมูล (Functions)
# ==========================================

def create_mock_2024():
    print("[INFO] Creating mock data for 2025...")
    # 1. สร้างช่วงเวลาปี 2025 (Leap Year - 366 วัน)
    dates = pd.date_range(start='2025-01-01', end='2025-12-31')
    day_of_year = dates.dayofyear

    # 2. จำลองอุณหภูมิ (SST) ปี 2024
    sst_base = 29.0 + 1.5 * np.sin((day_of_year - 100) * 2 * np.pi / 366)
    
    # Anomaly: ร้อนไวช่วงต้นปี + พีคช่วงเมษา
    early_year_heat = 0.8 * np.exp(-0.5 * ((day_of_year - 30) / 40) ** 2) 
    peak_heat = 1.0 * np.exp(-0.5 * ((day_of_year - 120) / 20) ** 2)      
    
    sst_2024 = sst_base + early_year_heat + peak_heat + np.random.normal(0, 0.1, len(dates))

    # 3. สร้าง DataFrame
    df = pd.DataFrame({'Date': dates, 'SST_Celsius': sst_2024})
    filename = 'sst_2024.csv'
    df.to_csv(filename, index=False)
    print(f"[OK] File '{filename}' created successfully!")
    return filename

def merge_and_clean_data(file1, file2):
    print(f"[INFO] Merging {file1} and {file2}...")
    try:
        df1 = pd.read_csv(file1)
        df2 = pd.read_csv(file2)
        
        # แปลงเป็น Datetime
        df1['Date'] = pd.to_datetime(df1['Date'])
        df2['Date'] = pd.to_datetime(df2['Date'])
        
        # รวมไฟล์
        df_merged = pd.concat([df1, df2], ignore_index=True)
        
        # จัดการข้อมูล: เรียงเวลา, ลบซ้ำ, ถมช่องว่าง
        df_merged = df_merged.drop_duplicates(subset=['Date'])
        df_merged = df_merged.sort_values(by='Date').set_index('Date')
        
        # Interpolate (ถมวันที่หายไป)
        df_full = df_merged.resample('D').mean()
        df_full['SST_Celsius'] = df_full['SST_Celsius'].interpolate(method='linear')
        df_full = df_full.reset_index()
        
        return df_full
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return None

def calculate_continuous_dhw(df):
    print("[INFO] Calculating Continuous DHW (Multi-year)...")
    MMM = 29.5
    dhw_list = []
    current_dhw = 0
    
    for temp in df['SST_Celsius']:
        if pd.isna(temp):
            dhw_list.append(current_dhw)
            continue
            
        diff = temp - MMM
        if diff > 1.0: 
            current_dhw += (diff / 7) # สะสมความร้อน
        else:
            current_dhw = max(0, current_dhw - 0.5) # คลายความร้อน
            
        dhw_list.append(current_dhw)
    
    df['DHW'] = dhw_list
    return df

# ==========================================
# PART 2: เริ่มทำงาน (Main Execution)
# ==========================================

# 1. สร้างไฟล์ปี 2025 ขึ้นมาก่อน
file_2025 = create_mock_2024()
file_2023 = 'sst_dhw_calculated.csv' # ไฟล์เดิมจากขั้นตอนที่แล้ว

# 2. รวมไฟล์ 2023 + 2025
df_combined = merge_and_clean_data(file_2023, file_2025)

if df_combined is not None:
    # 3. คำนวณ DHW ใหม่แบบต่อเนื่อง
    df_final = calculate_continuous_dhw(df_combined)
    
    # 4. บันทึกผลลัพธ์รวม
    output_csv = 'sst_dhw_2023_2024_2025_combined.csv'
    df_final.to_csv(output_csv, index=False)
    print(f"[OK] Saved combined data to: {output_csv}")
    
    # 5. วาดกราฟเปรียบเทียบ 2 ปี
    print("[INFO] Plotting graph...")
    fig, ax1 = plt.subplots(figsize=(14, 6))
    
    # Plot SST
    color_sst = '#d62728'
    ax1.plot(df_final['Date'], df_final['SST_Celsius'], color=color_sst, linewidth=1, label='SST')
    ax1.set_ylabel(r'SST ($^\circ$C)', color=color_sst)
    ax1.tick_params(axis='y', labelcolor=color_sst)
    
    # Reference Lines
    ax1.axhline(y=29.5, color='green', linestyle='--', alpha=0.5, label='MMM')
    ax1.axhline(y=30.5, color='orange', linestyle='--', alpha=0.5, label='Bleaching Threshold')
    
    # Plot DHW
    ax2 = ax1.twinx()
    color_dhw = '#404040'
    ax2.fill_between(df_final['Date'], df_final['DHW'], color='gray', alpha=0.3, label='DHW')
    ax2.plot(df_final['Date'], df_final['DHW'], color=color_dhw, linewidth=1)
    ax2.set_ylabel('DHW', color=color_dhw)
    ax2.tick_params(axis='y', labelcolor=color_dhw)
    
    # Alert Zones
    ax2.axhspan(4, 8, color='yellow', alpha=0.1)
    ax2.axhspan(8, 20, color='red', alpha=0.1)

    plt.title('Coral Bleaching Risk Analysis: 2023 - 2025 (Continuous)')
    plt.tight_layout()
    
    plot_filename = 'bleaching_risk_2023_2024_2025.png'
    plt.savefig(plot_filename, dpi=300)
    print(f"[SUCCESS] Graph saved as '{plot_filename}'")