import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
import datetime as dt

# ==========================================
# 1. โหลดข้อมูล
# ==========================================
file_path = 'sst_data_cleaned_final.csv'  
print(f"[INFO] Reading file: {file_path} ...")

try:
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
except FileNotFoundError:
    print(f"[ERROR] File '{file_path}' not found! Please check the file name.")
    exit()

# ==========================================
# 2. เตรียมข้อมูล (Data Preparation)
# ==========================================
df['Date_Ordinal'] = df['Date'].map(dt.datetime.toordinal)

X = df[['Date_Ordinal']]
y = df['SST_Celsius']

# ==========================================
# 3. สร้างและ Train Model
# ==========================================
model = LinearRegression()
model.fit(X, y)

slope = model.coef_[0]

# [FIXED] ใช้ภาษาอังกฤษเพื่อป้องกัน Unicode Error
print("-" * 30)
print(f"Slope: {slope:.5f} Celsius / Day") 
if slope > 0:
    print("Trend: WARMING (Going Up)")
else:
    print("Trend: COOLING (Going Down)")
print("-" * 30)

# ==========================================
# 4. ทำนายผล (Prediction & Forecast)
# ==========================================
df['Linear_Trend'] = model.predict(X)
df['Status'] = 'History'

# พยากรณ์ล่วงหน้า 90 วัน
days_to_predict = 90  
last_date = df['Date'].max()

future_dates = [last_date + dt.timedelta(days=x) for x in range(1, days_to_predict + 1)]
future_ordinal = [[d.toordinal()] for d in future_dates]
future_trend = model.predict(future_ordinal)

future_df = pd.DataFrame({
    'Date': future_dates,
    'SST_Celsius': [np.nan] * days_to_predict,
    'Date_Ordinal': [d[0] for d in future_ordinal],
    'Linear_Trend': future_trend,
    'Status': 'Forecast'
})

final_df = pd.concat([df, future_df], ignore_index=True)

# บันทึกไฟล์ CSV
output_csv = 'sst_trend_output_MILD.csv'
final_df[['Date', 'Status', 'SST_Celsius', 'Linear_Trend']].to_csv(output_csv, index=False)
print(f"[SUCCESS] Data saved to: {output_csv}")

# ==========================================
# 5. วาดกราฟ (Visualization)
# ==========================================
plt.figure(figsize=(12, 6))

# จุดข้อมูลจริง
plt.scatter(df['Date'], df['SST_Celsius'], color='#1f77b4', alpha=0.3, s=10, label='Actual SST')

# เส้น Trend
plt.plot(df['Date'], df['Linear_Trend'], color='red', linewidth=2, label='Linear Trend')

# เส้น Forecast
plt.plot(future_df['Date'], future_df['Linear_Trend'], color='orange', linestyle='--', linewidth=2, label=f'Forecast ({days_to_predict} Days)')

plt.title('SST Linear Regression Trend Analysis')
plt.xlabel('Date')
plt.ylabel('SST (Celsius)') # แก้แกน Y เป็นภาษาอังกฤษ
plt.legend()
plt.grid(True, alpha=0.3)

plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gcf().autofmt_xdate()

# [ADDED] บันทึกรูปกราฟ
output_img = 'sst_trend_mild_graph.png'
plt.tight_layout()
plt.savefig(output_img) 
print(f"[SUCCESS] Graph saved as: {output_img}")

plt.show()