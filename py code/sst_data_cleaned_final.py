import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

print("[INFO] Starting Data Cleaning Pipeline (Fixed)...")

# ==========================================
# STEP 1: โหลด/สร้างข้อมูลดิบ (Raw Data)
# ==========================================
def generate_raw_data():
    dates = pd.date_range(start='2023-01-01', end='2025-12-31')
    
    # [FIXED] แปลงเป็น numpy array ทันทีเพื่อให้แก้ไขค่าได้
    day_of_year = dates.dayofyear.to_numpy() 
    n_days = len(dates)
    
    # SST พื้นฐาน + Noise เยอะๆ
    sst_base = 29.0 + 1.5 * np.sin((day_of_year - 100) * 2 * np.pi / 365)
    noise = np.random.normal(0, 0.3, n_days)
    
    # สร้างข้อมูลดิบ
    sst_raw = sst_base + noise
    
    # ใส่ค่า NaN (จำลองข้อมูลหาย) -- ตอนนี้ทำได้แล้วเพราะเป็น array
    random_indices = np.random.choice(n_days, 20, replace=False)
    sst_raw[random_indices] = np.nan 
    
    return pd.DataFrame({'Date': dates, 'SST_Raw': sst_raw})

# สร้างข้อมูล
df = generate_raw_data()
print(f"[INFO] Raw data generated. Rows: {len(df)}")
print(f"       Missing values found: {df['SST_Raw'].isna().sum()}")

# ==========================================
# STEP 2: กระบวนการทำความสะอาด (Cleaning Process)
# ==========================================

# 2.1 Interpolation (ถมช่องว่าง)
df['SST_Cleaned'] = df['SST_Raw'].interpolate(method='linear')

# 2.2 Smoothing (ลดสัญญาณรบกวนด้วย Moving Average 7 วัน)
df['SST_Smoothed'] = df['SST_Cleaned'].rolling(window=7, center=True, min_periods=1).mean()

# 2.3 Rounding (ปัดเศษให้สวยงาม)
df['SST_Smoothed'] = df['SST_Smoothed'].round(2)
df['SST_Raw'] = df['SST_Raw'].round(2)

# ==========================================
# STEP 3: บันทึกและแสดงผล (Save & Plot)
# ==========================================

# Save CSV
final_df = df[['Date', 'SST_Smoothed']].rename(columns={'SST_Smoothed': 'SST_Celsius'})
filename_csv = 'sst_data_cleaned_final.csv'
final_df.to_csv(filename_csv, index=False)
print(f"[SUCCESS] Cleaned data saved to '{filename_csv}'")

# Plot Comparison
plt.figure(figsize=(12, 6))

# ข้อมูลดิบ (สีเทาจางๆ)
plt.plot(df['Date'], df['SST_Raw'], color='gray', alpha=0.3, linewidth=1, label='Raw Data (Noisy & Gaps)')

# ข้อมูลที่คลีนแล้ว (สีน้ำเงินเข้ม)
plt.plot(df['Date'], df['SST_Smoothed'], color='#0052cc', linewidth=2, label='Cleaned & Smoothed (7-day MA)')

plt.title('Data Cleaning Results: Raw vs. Cleaned SST (2023-2025)')
plt.ylabel('SST (Celsius)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

filename_img = 'data_cleaning_comparison.png'
plt.savefig(filename_img, dpi=150)
print(f"[SUCCESS] Comparison graph saved as '{filename_img}'")