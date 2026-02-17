import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# โหลดข้อมูล
df = pd.read_csv("C:\\Users\\buako\\OneDrive\\เดสก์ท็อป\\Multi-Temporal-Satellite-AI-for-Coral-Bleaching-Forecast\\CSV Files\\sst_trend_output_MILD.csv")
df['Date'] = pd.to_datetime(df['Date'])

# แยกข้อมูล History และ Forecast
history = df[df['Status'] == 'History']
forecast = df[df['Status'] == 'Forecast']

# ตั้งค่ากราฟ
plt.figure(figsize=(12, 6), dpi=100)

# 1. วาดข้อมูลจริง (Actual Data)
plt.scatter(history['Date'], history['SST_Celsius'], 
            color='#1f77b4', alpha=0.3, s=10, label='Actual SST (History)')

# 2. วาดเส้น Trend ช่วงอดีต (Trend Line)
plt.plot(history['Date'], history['Linear_Trend'], 
         color='red', linewidth=2, label='Linear Trend')

# 3. วาดเส้นพยากรณ์ (Forecast Line)
plt.plot(forecast['Date'], forecast['Linear_Trend'], 
         color='orange', linestyle='--', linewidth=2.5, label='Forecast (Next 90 Days)')

# ตกแต่งกราฟ
plt.title('SST Trend Analysis & Forecast (Linear Regression)', fontsize=14)
plt.ylabel('Sea Surface Temperature (°C)', fontsize=12)
plt.xlabel('Date', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

# จัด Format วันที่แกน X
plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gcf().autofmt_xdate()

# แสดงผล
plt.tight_layout()
plt.savefig('sst_trend_graph_from_csv.png')
plt.show()