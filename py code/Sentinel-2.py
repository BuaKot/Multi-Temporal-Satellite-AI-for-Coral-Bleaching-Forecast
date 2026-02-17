import pandas as pd
import numpy as np
import matplotlib

# ใช้โหมด 'Agg' เพื่อแก้ปัญหาหน้าต่างกราฟค้าง/ไอคอนหาย
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ==========================================
# 1. สร้างข้อมูลจำลอง (Mock Data)
# ==========================================
print("Creating sample data for 2023...")
dates = pd.date_range(start='2023-01-01', end='2023-12-31')
day_of_year = dates.dayofyear

# จำลองอุณหภูมิ
sst_simulated = 29.0 + 1.5 * np.sin((day_of_year - 100) * 2 * np.pi / 365) 
heatwave = 1.2 * np.exp(-0.5 * ((day_of_year - 130) / 20) ** 2) 
sst_final = sst_simulated + heatwave + np.random.normal(0, 0.1, len(dates))

# --- แก้ไข: ประกาศตัวแปรให้เป็นตัวพิมพ์ใหญ่ (MMM) ให้ตรงกับตอนกราฟ ---
MMM = 29.5
Bleaching_Threshold = MMM + 1.0
# -----------------------------------------------------------

dhw_list = []
current_dhw = 0

for temp in sst_final:
    diff = temp - MMM  # แก้ชื่อตัวแปรตรงนี้ด้วย
    if diff > 1.0: 
        current_dhw += (diff / 7)
    else:
        current_dhw = max(0, current_dhw - 0.5) 
    dhw_list.append(current_dhw)

df = pd.DataFrame({'Date': dates, 'SST_Celsius': sst_final, 'DHW': dhw_list})
filename = 'sst_dhw_calculated.csv'
df.to_csv(filename, index=False)
print(f"File '{filename}' created successfully!")

# ==========================================
# 2. วาดกราฟ (Plotting)
# ==========================================
print("Plotting graph (Headless mode)...")

try:
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # --- Plot 1: SST ---
    color_sst = '#d62728'
    ax1.set_xlabel('Date (Year 2023)')
    ax1.set_ylabel(r'Sea Surface Temperature ($^\circ$C)', color=color_sst)
    ax1.plot(df['Date'], df['SST_Celsius'], color=color_sst, linewidth=1.5, label='SST')
    ax1.tick_params(axis='y', labelcolor=color_sst)

    # Reference Lines (เรียกใช้ MMM ตัวพิมพ์ใหญ่ได้แล้ว)
    ax1.axhline(y=MMM, color='green', linestyle='--', alpha=0.7, label=r'MMM (29.5$^\circ$C)')
    ax1.axhline(y=Bleaching_Threshold, color='orange', linestyle='--', alpha=0.7, label='Bleaching Threshold')

    # --- Plot 2: DHW ---
    ax2 = ax1.twinx()
    color_dhw = '#404040'
    ax2.set_ylabel(r'Degree Heating Weeks ($^\circ$C-weeks)', color=color_dhw)
    
    ax2.fill_between(df['Date'], df['DHW'], color='gray', alpha=0.3, label='DHW')
    ax2.plot(df['Date'], df['DHW'], color=color_dhw, linewidth=1, alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color_dhw)
    ax2.set_ylim(0, max(df['DHW'].max() + 2, 10))

    # Alert Levels
    ax2.axhspan(4, 8, color='yellow', alpha=0.1)
    ax2.text(df['Date'].iloc[0], 4.2, ' Alert Level 1 ', color='orange', fontsize=8, fontweight='bold')
    ax2.axhspan(8, 20, color='red', alpha=0.1)
    ax2.text(df['Date'].iloc[0], 8.2, ' Alert Level 2 (Severe) ', color='red', fontsize=8, fontweight='bold')

    plt.title('Coral Bleaching Risk Analysis 2023: Koh Man Nai', fontsize=14, fontweight='bold')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    
    # Save Image
    output_img = 'final_bleaching_chart_2023.png'
    plt.savefig(output_img, dpi=300)
    print(f"SUCCESS: Chart saved as {output_img}")
    print("คุณสามารถเปิดไฟล์รูปภาพจากโฟลเดอร์ได้เลยครับ")

except Exception as e:
    print(f"Error: {e}")