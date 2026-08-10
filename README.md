# Multi-Temporal Satellite AI for Coral Bleaching Forecast

> **An AI-powered environmental intelligence platform** for monitoring, analyzing, and forecasting coral bleaching risks using multi-temporal satellite data and machine learning algorithms.

---

## Executive Summary

Coral bleaching is primarily triggered by prolonged thermal stress resulting from elevated **Sea Surface Temperature (SST)**. This project leverages multi-temporal satellite remote sensing data combined with Artificial Intelligence to provide predictive insights, enabling marine conservationists and decision-makers to anticipate and mitigate ecological risks before severe degradation occurs.

---

##Core Methodology & Working Principles
1. **Multi-Temporal Data Acquisition**: Extracts continuous temporal sequences of Sea Surface Temperature (SST) and oceanographic indicators from satellite sensors.
2. **Data Pipeline & Preprocessing**:
   * Cleans missing spatial/temporal entries.
   * Calculates Degree Heating Weeks (DHW) and thermal anomaly thresholds.
   * Normalizes time-series features for high-dimensional model ingestion.
3. **AI Sequence Modeling**: Uses deep temporal networks (such as LSTM/GRU or Transformer architectures) to capture seasonal trends and non-linear heat-accumulation dynamics over time.
4. **Predictive Analytics & Forecasting**: Outputs risk probability scores and spatial heatmaps to project bleaching severity levels ahead of time.

---

## Repository Structure

```text
├── 📁 CSV Files/                # Raw and processed tabular datasets
├── 📁 Data Python Code/         # Data ingestion and processing scripts
├── 📁 PNG/                      # Visualizations, charts, and spatial maps
├── 📁 venv/                     # Python virtual environment setup
├── 📜 .gitignore               # Ignored files (builds, virtualenvs, cache)
├── 📜 sst_2024.csv              # Sea Surface Temperature data for 2024
├── 📜 sst_data_cleaned_final.csv# Fully preprocessed dataset ready for training
└── 📜 README.md                # Project documentation
# Clone the repository
git clone [https://github.com/BuaKot/Multi-Temporal-Satellite-AI-for-Coral-Bleaching-Forecast.git](https://github.com/BuaKot/Multi-Temporal-Satellite-AI-for-Coral-Bleaching-Forecast.git)
cd Multi-Temporal-Satellite-AI-for-Coral-Bleaching-Forecast

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install required dependencies
pip install -r requirements.txt
python "Data Python Code/preprocessing.py"
python "Data Python Code/train_forecast.py"
