import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg') # Use 'Agg' backend for non-GUI environments
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression # Import Linear Regression model
import datetime as dt

print("[INFO] Starting Prediction Process (2023-2027)...")

# Data Generation Function from 2023 to 2025
def generate_history_data():
    data_frames = []
    for year in [2023, 2024, 2025]:
        dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31')
        day_of_year = dates.dayofyear
        n_days = len(dates)
        
        # Base Data
        sst_base = 29.0 + 1.5 * np.sin((day_of_year - 100) * 2 * np.pi / n_days)
        noise = np.random.normal(0, 0.15, n_days)
        
        if year == 2023: anomaly = 1.0 * np.exp(-0.5 * ((day_of_year - 140) / 30) ** 2)
        elif year == 2024: anomaly = 1.4 * np.exp(-0.5 * ((day_of_year - 120) / 35) ** 2) + 0.2
        elif year == 2025: anomaly = -0.3 * np.exp(-0.5 * ((day_of_year - 200) / 60) ** 2) + 0.05
        else: anomaly = 0
            
        data_frames.append(pd.DataFrame({'Date': dates, 'SST_Celsius': sst_base + anomaly + noise}))
    return pd.concat(data_frames, ignore_index=True)

df_history = generate_history_data()

# Model Trainning 2023-2025
print("[INFO] Training Model on 2023-2025 data...")

#train data
df_history['Date_Ordinal'] = df_history['Date'].map(dt.datetime.toordinal) # Convert date to ordinal for regression
X_train = df_history['Date_Ordinal'].values.reshape(-1, 1) #Date as feature
y_train = df_history['SST_Celsius'].values #SST

# Create and train model
model = LinearRegression()
model.fit(X_train, y_train)

#Predict future data
print("[INFO] Forecasting for 2026-2027...")
future_dates = pd.date_range(start='2026-01-01', end='2027-12-31')
future_dates_ordinal = future_dates.map(dt.datetime.toordinal).values.reshape(-1, 1)
####
# Model Prediction
future_pred = model.predict(future_dates_ordinal)

# Cobine Historical and Future Data for Plotting
all_dates = pd.concat([df_history['Date'], pd.Series(future_dates)])
all_preds = model.predict(all_dates.map(dt.datetime.toordinal).values.reshape(-1, 1))

# Calculate and report trend
slope = model.coef_[0]
trend_per_year = slope * 365

print("-" * 40)
print(f"FORECAST REPORT (5-Year View)")
print(f"Trend Rate: {trend_per_year:.4f} Celsius/Year")
if slope > 0: print("Direction: WARMING UP")
else: print("Direction: COOLING DOWN (Recovery Phase)")
print("-" * 40)

# Plotting
plt.figure(figsize=(14, 7))

# Data Points 2023-2025
plt.scatter(df_history['Date'], df_history['SST_Celsius'], color='gray', alpha=0.4, s=5, label='Historical Data (2023-2025)') # Scatter plot for historical data

# Data Points 2026-2027
plt.plot(all_dates, all_preds, color='red', linewidth=2.5, linestyle='--', label='Trend Forecast (Extended)') # Line plot for trend

# Highlight Forecast Period
plt.axvspan(pd.to_datetime('2026-01-01'), pd.to_datetime('2027-12-31'), color='blue', alpha=0.05, label='Forecast Period (2026-2027)') # Highlight area

# Decorations
plt.title(f'SST Forecast 5-Year Horizon (2023-2027)\nBased on Linear Trend: {trend_per_year:.4f} C/Year', fontsize=14)
plt.ylabel('SST (Celsius)')
plt.xlabel('Date')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)

# Formatting Dates
plt.gca().xaxis.set_major_locator(matplotlib.dates.YearLocator())
plt.gca().xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%Y'))

plt.tight_layout()
filename = 'sst_forecast_2027.png'
plt.savefig(filename, dpi=150)
print(f"[SUCCESS] Forecast graph saved as '{filename}'")