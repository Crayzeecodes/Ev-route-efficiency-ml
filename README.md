# EV Route Efficiency ML 🚗⚡

Predict battery consumption for electric vehicles (EVs) along a planned route to help users optimize energy efficiency and avoid unexpected mid-route recharging.

---

## 📝 Problem Statement

EV owners often face uncertainty in estimating battery consumption for a planned trip. The challenge is to **predict the battery SOC (State of Charge) drop** along a route, considering factors like:

- Speed  
- Elevation  
- Distance  
- Temperature  
- Battery Voltage & Current  

This allows EV users to choose **battery-efficient routes** and plan charging stops proactively.

---

## 🛠️ Approach / Solution

We approached the problem in the following steps:

### 1️⃣ Data Collection & Cleaning
- Collected EV trip data including speed, elevation, distance, battery parameters, temperature, and other environmental features.  
- Removed missing values and outliers to prepare a **clean dataset** (`final_dataset_ev.csv`).

### 2️⃣ Exploratory Data Analysis (EDA)
- Visualized feature distributions and correlations with battery SOC.  
- Identified key drivers of energy consumption.

### 3️⃣ Model Selection
- Compared multiple regression models: **Linear Regression**, **Decision Tree**, and **Random Forest**.  
- **Random Forest Regressor** outperformed others based on accuracy metrics.

### 4️⃣ Model Training & Evaluation
- Split dataset: 80% training, 20% testing.  
- Evaluated using **R² Score, MAE, and RMSE**.  
- Saved trained model for future route simulations.

---

## 📊 Dataset

**Features Used:**

- Speed (km/h)  
- Elevation (m)  
- Distance (m)  
- Temperature (°C)  
- Battery Voltage (V)  
- Battery Current (A)  
- Battery SOC (%)

**Dataset Link:** [EVED Dataset](https://bitbucket.org/datarepo/eved-dataset/src/main/data/eVED.zip)

---

## 💡 Model & Performance

**Model Chosen:** Random Forest Regressor

| Metric                     | Value   |
|-----------------------------|---------|
| R² Score                    | 0.9756  |
| Mean Absolute Error (MAE)   | 2.7179  |
| Root Mean Squared Error (RMSE) | 5.5094 |

> ⚡ Random Forest significantly outperforms Linear Regression (R² ≈ 0.5157), making it highly reliable for SOC predictions along routes.

---

## ✨ Improvisations / Enhancements

- **NaN Handling & Dataset Cleaning:** Removed missing values and outliers for improved model performance.  
- **Feature Engineering:** Added battery parameters and environmental factors (Temperature, Slope, Humidity) for realistic predictions.  
- **Model Upgrade:** Replaced simple Linear Regression with **Random Forest**, boosting accuracy dramatically.  

---

