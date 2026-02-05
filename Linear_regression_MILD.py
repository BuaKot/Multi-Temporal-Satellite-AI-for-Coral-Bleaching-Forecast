import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.linear_model import LinearRegression
import datetime as dt

# ==========================================
# 1. โหลดข้อมูล
# ==========================================
file_path = 'sst_data_cleaned_final.csv'  # ตรวจสอบชื่อไฟล์ให้ตรง
print(f"กำลังอ่านไฟล์: {file_path} ...")

try:
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
except FileNotFoundError:
    print(f"Error: หาไฟล์ {file_path} ไม่เจอ! ช่วยเช็คว่าไฟล์อยู่ในโฟลเดอร์เดียวกันหรือยังครับ")
    exit()

# ==========================================
# 2. เตรียมข้อมูล (Data Preparation)
# ==========================================
# แปลงวันที่เป็นตัวเลข (Ordinal) เพื่อให้คอมพิวเตอร์คำนวณสมการเส้นตรงได้
df['Date_Ordinal'] = df['Date'].map(dt.datetime.toordinal)

X = df[['Date_Ordinal']]  # ตัวแปรต้น (เวลา)
y = df['SST_Celsius']     # ตัวแปรตาม (อุณหภูมิ)

# ==========================================
# 3. สร้างและ Train Model (Linear Regression)
# ==========================================
model = LinearRegression()
model.fit(X, y)

# ดูความชัน (Slope)
slope = model.coef_[0]
print(f"ความชัน (Slope): {slope:.5f} °C ต่อวัน")
if slope > 0:
    print("แนวโน้ม: อุณหภูมิ 'สูงขึ้น' (Warming Trend)")
else:
    print("แนวโน้ม: อุณหภูมิ 'ลดลง' (Cooling Trend)")

# ==========================================
# 4. ทำนายผล (Prediction & Forecast)
# ==========================================

# 4.1 คำนวณเส้น Trend ในอดีต
df['Linear_Trend'] = model.predict(X)
df['Status'] = 'History'

# 4.2 พยากรณ์อนาคต (แก้จำนวนวันตรงนี้ได้)
days_to_predict = 90  
last_date = df['Date'].max()

# สร้างวันที่ในอนาคต
future_dates = [last_date + dt.timedelta(days=x) for x in range(1, days_to_predict + 1)]
future_ordinal = [[d.toordinal()] for d in future_dates]

# ให้ Model ทำนายค่า
future_trend = model.predict(future_ordinal)

# สร้างตารางข้อมูลอนาคต
future_df = pd.DataFrame({
    'Date': future_dates,
    'SST_Celsius': [np.nan] * days_to_predict, # อนาคตไม่มีค่าจริง ใส่ NaN
    'Date_Ordinal': [d[0] for d in future_ordinal],
    'Linear_Trend': future_trend,
    'Status': 'Forecast'
})

# รวมข้อมูลเก่า + ใหม่
final_df = pd.concat([df, future_df], ignore_index=True)

# บันทึกเป็น CSV (เอาไปใช้ต่อได้เลย)
output_csv = 'sst_trend_output_MILD.csv'
final_df[['Date', 'Status', 'SST_Celsius', 'Linear_Trend']].to_csv(output_csv, index=False)
print(f"\nบันทึกไฟล์สำเร็จ: {output_csv}")

# ==========================================
# 5. วาดกราฟ (Visualization)
# ==========================================
plt.figure(figsize=(12, 6))

# จุดข้อมูลจริง (สีฟ้าจางๆ)
plt.scatter(df['Date'], df['SST_Celsius'], color='#1f77b4', alpha=0.3, s=10, label='Actual SST')

# เส้น Trend (สีแดง)
plt.plot(df['Date'], df['Linear_Trend'], color='red', linewidth=2, label='Linear Trend')

# เส้น Forecast (สีส้มประ)
plt.plot(future_df['Date'], future_df['Linear_Trend'], color='orange', linestyle='--', linewidth=2, label=f'Forecast ({days_to_predict} Days)')

plt.title('SST Linear Regression Trend Analysis')
plt.xlabel('Date')
plt.ylabel('Sea Surface Temperature (°C)')
plt.legend()
plt.grid(True, alpha=0.3)

# จัดรูปแบบแกนวันที่ให้สวยงาม
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
plt.gcf().autofmt_xdate()

# แสดงกราฟ
plt.tight_layout()
plt.show() # ถ้ากราฟไม่ขึ้น ให้เช็คว่าลง matplotlib หรือยัง