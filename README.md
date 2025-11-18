

# GROUTE: Intelligent EV Route Energy Assistant 🚗⚡

**GROUTE** is a smart, chat-based navigation assistant designed to help electric vehicle (EV) drivers **predict energy consumption**, plan charging stops, and select the most **battery-efficient routes**. By combining machine learning with Google Maps APIs, GROUTE tackles range anxiety and makes EV trip planning data-driven and reliable.

---

## 📝 Problem Statement

Electric vehicle drivers lack reliable insights into how different routes impact energy consumption. Existing navigation systems treat EVs like conventional vehicles and ignore critical factors such as:

* Road gradients
* Temperature
* Traffic conditions
* Driving speed

This leads to **uncertainty in trip planning**, inaccurate range expectations, and increased **range anxiety**, as drivers are unsure whether their battery will last the journey or if charging stops will be needed.

---

## 🎯 Learning Objectives

* Understand the challenges of EV route planning, including range anxiety, charging availability, and energy variability.
* Analyze real-world route parameters (distance, slope, speed, traffic, temperature) and their impact on EV energy consumption.
* Explore supervised machine learning techniques for predicting energy usage based on vehicle and environmental features.
* Integrate multiple Google Maps APIs (Directions, Geocoding, Places) to fetch real-time routing and charging station data.
* Develop a multi-step energy estimation pipeline, including polyline decoding, coordinate processing, and SOC tracking.
* Design a user-friendly chat-based interface to collect trip details and deliver intelligent route and charging suggestions.
* Visualize multi-route comparisons using interactive maps and geospatial tools.
* Build an end-to-end intelligent EV assistant capable of recommending the most energy-efficient route.

---

## 🛠️ Tools & Technology Used

**Machine Learning & Data:**

* Python — core backend development
* Pandas, NumPy — data cleaning & feature engineering
* Scikit-Learn — ML model training for energy prediction
* Joblib — saving and loading the trained model

**Routing & Maps:**

* Google Maps Directions API — multi-route data
* Google Maps Geocoding API — coordinate retrieval
* Google Places API — EV charging station lookup
* Polyline — decoding encoded routes
* Haversine — accurate step-wise distance calculation

**Frontend / UI:**

* Streamlit — chat-driven interface
* Streamlit-Folium — interactive map rendering
* Folium — route visualization

**Code Management & Deployment:**

* GitHub — version control & project hosting
* dotenv — environment variable & API key management
* Virtual Environment (venv) — dependency isolation

---

## 🛠️ Methodology

**Step 1: Problem Identification**

* EV drivers lack route-wise energy insights, causing range anxiety and inefficient trips.

**Step 2: Data Collection & Feature Engineering**

* Extracted route distances, elevation, temperature, traffic, slope, and speed.
* Engineered features: energy consumption per km, SOC drop, traffic factor.

**Step 3: Model Training**

* Trained an ML regression model (Random Forest) to estimate energy consumption per route segment.
* Saved the model for real-time predictions using Joblib.

**Step 4: Route Evaluation Engine**

* Fetched multiple route options using Google Directions API.
* Calculated step-wise energy consumption using the trained model.

**Step 5: Best Route Selection**

* Compared all possible routes based on energy usage.
* Selected the most energy-efficient route.

**Step 6: Charging Station Logic**

* Tracked SOC throughout the journey.
* When SOC predicted to drop below 15%, located nearest EV charging station and marked it on the map.

**Step 7: Frontend Integration (GROUTE Chatbot)**

* Created a chat-driven assistant using Streamlit.
* Users provide start point, destination, SOC, and battery details.
* Chatbot displays maps, tables, efficiency metrics, and insights.

---

## 📊 Dataset & Features

**Target Variable:** `Energy_Consumption_kWh`

**Numeric Features:**
Speed, Acceleration, Battery State, Battery Temperature, Distance Travelled, Slope, Temperature, Humidity, Wind Speed, Vehicle Weight, Tire Pressure

**Categorical Features:**
Driving Mode, Road Type, Traffic Condition, Weather Condition

---

## 💡 Model & Performance

**Model Chosen:** Random Forest Regressor

| Metric                         | Value     |
| ------------------------------ | --------- |
| R² Score                       | 0.914     |
| Mean Absolute Error (MAE)      | 0.515 kWh |
| Root Mean Squared Error (RMSE) | 0.647 kWh |

---

## 📈 Visualizations

* **Predicted vs Actual Energy Consumption** — shows high correlation.
* **Residuals Histogram** — confirms low bias and small prediction errors.
* **Feature Importance Chart** — highlights factors affecting EV energy consumption.

!(images1.png)

---

## 💡 Solution: GROUTE

**GROUTE** is a smart, chat-based EV navigation assistant that:

1. Analyzes all available routes between two locations.
2. Predicts energy consumption using a trained ML model.
3. Estimates final State of Charge (SOC).
4. Identifies when the battery will drop below a safe threshold.
5. Automatically suggests the nearest EV charging station.

**Displays:**

* Most efficient route map
* All routes comparison map
* Traffic factor
* Energy consumed
* Final SOC
* Expected route duration

GROUTE provides EV-aware, battery-intelligent navigation, filling the gap left by conventional navigation apps.

---

## 🖼️ Output / Screenshots

> *(Attach screenshots of Predicted vs Actual Energy Consumption, Residuals Histogram, Feature Importance, and the chat-based interface here.)*

---

## 🏁 Conclusion

GROUTE successfully demonstrates how **machine learning + maps APIs** can remove range anxiety for EV users.

* Provides **energy-aware routing** and real-time charging insights.
* Accurately estimates energy consumption along routes.
* Empowers drivers to select routes based on **efficiency**, not just distance or time.
* Scalable to different EV models, real-time weather, elevation & traffic updates, and fleet optimization.

GROUTE builds a strong foundation for **next-generation intelligent EV navigation**.

---



Do you want me to do that?
