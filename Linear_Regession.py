import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Safe mode for plotting
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import datetime as dt

print("[INFO] Starting Process: Generate Data + Linear Regression...")

# ==========================================
# PART 1: สร้างข้อมูลจำลอง 3 ปี (2023-2025)
# ==========================================
def generate_mock_data():
    data_frames = []
    print("[INFO] Generating mock data...")
    
    for year in [2023, 2024, 2025]:
        dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31')
        day_of_year = dates.dayofyear
        n_days = len(dates)
        
        # Base Temperature
        sst_base = 29.0 + 1.5 * np.sin((day_of_year - 100) * 2 * np.pi / n_days)
        noise = np.random.normal(0, 0.15, n_days)
        
        # Trend / Anomaly
        if year == 2023:
            anomaly = 1.0 * np.exp(-0.5 * ((day_of_year - 140) / 30) ** 2)
        elif year == 2024:
            # Heatwave year + Slight warming trend
            anomaly = 1.4 * np.exp(-0.5 * ((day_of_year - 120) / 35) ** 2) + 0.2
        elif year == 2025:
            # Recovery but base is slightly higher (Global Warming effect)
            anomaly = -0.3 * np.exp(-0.5 * ((day_of_year - 200) / 60) ** 2) + 0.05
        else:
            anomaly = 0
            
        sst_final = sst_base + anomaly + noise
        data_frames.append(pd.DataFrame({'Date': dates, 'SST_Celsius': sst_final}))

    return pd.concat(data_frames, ignore_index=True)

# สร้างข้อมูลขึ้นมาใหม่เลย (ไม่ต้องง้อไฟล์เก่า)
df = generate_mock_data()

# ==========================================
# PART 2: วิเคราะห์ Linear Regression
# ==========================================
print("[INFO] Fitting Linear Regression Model...")

# แปลงวันที่เป็นตัวเลข (Ordinal) เพื่อคำนวณ
df['Date_Ordinal'] = df['Date'].map(dt.datetime.toordinal)

X = df['Date_Ordinal'].values.reshape(-1, 1)
y = df['SST_Celsius'].values

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# คำนวณค่าสถิติ
slope = model.coef_[0]
intercept = model.intercept_
r_squared = model.score(X, y)
warming_per_year = slope * 365

# ==========================================
# PART 3: แสดงผลและวาดกราฟ
# ==========================================
print("-" * 40)
print("REGRESSION RESULTS")
print(f"Slope (Daily change): {slope:.5f}")
print(f"Trend (Yearly change): {warming_per_year:.4f} Celsius/Year")
print(f"R-squared: {r_squared:.4f}")

if slope > 0:
    print("CONCLUSION: Trend is WARMING (Positive)")
    trend_color = 'red'
else:
    print("CONCLUSION: Trend is COOLING (Negative)")
    trend_color = 'blue'
print("-" * 40)

# Plot Graph
plt.figure(figsize=(12, 6))
plt.scatter(df['Date'], df['SST_Celsius'], color='gray', alpha=0.3, s=5, label='Actual Data')
plt.plot(df['Date'], y_pred, color=trend_color, linewidth=2, label=f'Linear Trend ({warming_per_year:.2f} C/Year)')

plt.title(f'SST Trend Analysis (2023-2025)\nWarming Rate: {warming_per_year:.4f} Celsius per Year')
plt.ylabel('SST (Celsius)')
plt.xlabel('Date')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()

filename = 'sst_linear_regression_final.png'
plt.savefig(filename, dpi=150)
print(f"[SUCCESS] Graph saved as '{filename}'")