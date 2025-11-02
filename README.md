# Ev-route-efficiency-ml
EV Route Optimization using Machine Learning
Problem Statement

Electric vehicle (EV) owners often face uncertainty in estimating battery consumption for a planned route. The challenge is to predict the battery SOC (State of Charge) drop along a route, considering factors like speed, elevation, distance, temperature, battery voltage, and current. This allows EV users to choose routes that optimize energy efficiency and avoid unexpected mid-route recharging.

Approach / Solution

We approached the problem in the following steps:

Data Collection & Cleaning
Collected EV trip data including speed, elevation, distance, battery parameters, temperature, and other environmental features. Removed missing values and outliers to prepare a clean dataset.

Exploratory Data Analysis (EDA)
Visualized feature distributions and relationships with battery SOC to understand key drivers of energy consumption.

Model Selection
Compared multiple regression models: Linear Regression, Decision Tree, and Random Forest.
Based on performance metrics, Random Forest Regressor was chosen as the best model.

Model Training & Evaluation
Trained the Random Forest model on 80% of the dataset and tested on 20%. Evaluated performance using R² Score, MAE, and RMSE.

Route SOC Prediction
Developed a route simulation that predicts SOC drop along a given route using the trained model, taking into account user-provided battery SOC, temperature, and capacity.

Dataset

The dataset used is the cleaned EV trip dataset, containing features like:

Speed (km/h)

Elevation (m)

Distance (m)

Temperature (°C)

Battery Voltage (V)

Battery Current (A)

Battery SOC (%)

Dataset link: [Add your dataset link here]

Model & Performance

Model Chosen: Random Forest Regressor

Metric	Value
R² Score	0.9756
Mean Absolute Error (MAE)	2.7179
Root Mean Squared Error (RMSE)	5.5094

The Random Forest model significantly outperforms Linear Regression (R² ≈ 0.5157), making it highly reliable for predicting SOC drop along routes.
