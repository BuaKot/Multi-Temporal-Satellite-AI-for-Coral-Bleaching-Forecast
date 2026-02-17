import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

print("[INFO] Starting Fresh Generation (2023-2024)...")

# Mock Data Generation Function
def generate_mock_year(year, start_temp_anomaly=0.0):
    # สร้างวันที่ 1 ม.ค. - 31 ธ.ค.
    dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31')
    day_of_year = dates.dayofyear
    days_in_year = len(dates)

    # Base Temperature (SST พื้นฐานตามฤดูกาล)
    sst_base = 29.0 + 1.5 * np.sin((day_of_year - 100) * 2 * np.pi / days_in_year)
    
    # Noise & Variation
    random_noise = np.random.normal(0, 0.15, days_in_year)
    
    # Anomaly เฉพาะปี (จำลองความร้อน)
    if year == 2023:
        # ปี 2023: ร้อนช่วงกลางปี (El Nino เริ่ม)
        heat_wave = 1.2 * np.exp(-0.5 * ((day_of_year - 150) / 30) ** 2)
    elif year == 2024:
        # ปี 2024: ร้อนสะสมต่อเนื่องจากต้นปี
        heat_wave = 1.0 * np.exp(-0.5 * ((day_of_year - 100) / 40) ** 2)
        # เพิ่ม Offset ให้ร้อนกว่าปกตินิดหน่อย
        start_temp_anomaly = 0.2 
    else:
        heat_wave = 0

    sst_final = sst_base + heat_wave + random_noise + start_temp_anomaly
    
    return pd.DataFrame({'Date': dates, 'SST_Celsius': sst_final})

# Generate Data for 2023 and 2024
print("[INFO] Generating 2023 data...")
df_2023 = generate_mock_year(2023)

print("[INFO] Generating 2024 data...")
df_2024 = generate_mock_year(2024)

print("[INFO] Merging datasets...")
df_combined = pd.concat([df_2023, df_2024], ignore_index=True)
df_combined['Date'] = pd.to_datetime(df_combined['Date'])
df_combined = df_combined.sort_values('Date')

# Calculate DHW
print("[INFO] Calculating DHW...")
MMM = 29.5
dhw_list = []
current_dhw = 0

for temp in df_combined['SST_Celsius']:
    diff = temp - MMM
    if diff > 1.0:
        current_dhw += (diff / 7) # สะสมความร้อน
    else:
        current_dhw = max(0, current_dhw - 0.5) # คลายความร้อน
    dhw_list.append(current_dhw)

df_combined['DHW'] = dhw_list

# Save CSV
output_csv = 'sst_dhw_FINAL_2023_2024.csv'
df_combined.to_csv(output_csv, index=False)
print(f"[OK] Data saved to {output_csv}")

# Plotting
print("[INFO] Plotting final graph...")
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot SST (สีแดง)
color_sst = '#d62728'
ax1.plot(df_combined['Date'], df_combined['SST_Celsius'], color=color_sst, linewidth=1, label='SST')
ax1.set_ylabel('SST (Celsius)', color=color_sst)
ax1.tick_params(axis='y', labelcolor=color_sst)
ax1.set_ylim(27, 32) # ล็อคแกน Y ให้อ่านง่าย

# Threshold Lines
ax1.axhline(y=MMM, color='green', linestyle='--', alpha=0.6, label='MMM (29.5 C)')
ax1.axhline(y=MMM+1, color='orange', linestyle='--', alpha=0.6, label='Bleaching Threshold')

# Plot DHW สีเทา
ax2 = ax1.twinx()
color_dhw = '#404040'
ax2.fill_between(df_combined['Date'], df_combined['DHW'], color='gray', alpha=0.3, label='DHW')
ax2.plot(df_combined['Date'], df_combined['DHW'], color=color_dhw, linewidth=0.5)
ax2.set_ylabel('DHW (Degree Heating Weeks)', color=color_dhw)
ax2.set_ylim(0, 16) # ล็อคแกน DHW ไม่ให้สูงเกินไป

# Alert Level Shading
ax2.axhspan(4, 8, color='yellow', alpha=0.15) # Alert Level 1
ax2.axhspan(8, 16, color='red', alpha=0.15)   # Alert Level 2

plt.title('Corrected Coral Bleaching Risk (2023-2024)')
plt.tight_layout()

filename_plot = 'corrected_bleaching_risk.png'
plt.savefig(filename_plot, dpi=150)
print(f"[SUCCESS] Graph saved as '{filename_plot}'. No flat lines!")