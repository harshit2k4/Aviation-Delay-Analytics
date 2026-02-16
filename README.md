# Aviation: Flight Delay Prediction System

## Overview

This project builds a machine learning binary classification model to predict flight arrival delays (≥15 minutes) using operational flight features.  
A Random Forest classifier achieved 78.8% accuracy with strong precision (0.8361) and is deployed as an interactive Streamlit web application.

The model predicts whether a flight will be:

- On-Time (Arrival delay less than 15 minutes)  
- Delayed (Arrival delay equal to or greater than 15 minutes)  

---

## Live Application

🔗 https://flight-delay-analytics.streamlit.app/

---

## Problem Statement

Flight delays affect airline efficiency and passenger satisfaction.

Target Variable: **is_delayed**

- is_delayed = 1 → Arrival delay ≥ 15 minutes  
- is_delayed = 0 → Arrival delay < 15 minutes  

This project uses **Binary Classification** to predict the value of `is_delayed`.

---


## Project Structure

```text
.
├── app.py                      # Streamlit application
├── data
│   ├── airlines.csv
│   └── airports.csv
├── models
│   ├── best_model.pkl          # Trained Random Forest model
│   └── scaler.pkl              # StandardScaler parameters
└── notebooks
    └── Aviation_Delay_Analytics.ipynb
```

---

## Features Used

The model uses 10 operational features:

- TAXI_OUT  
- WHEELS_OFF  
- DEPARTURE_TIME  
- TAXI_IN  
- SCHEDULED_DEPARTURE  
- SCHEDULED_ARRIVAL  
- WHEELS_ON  
- ARRIVAL_TIME  
- ELAPSED_TIME  
- MONTH  

Ground movement times and schedule timings are the strongest predictors.

---

## Methodology

### Data Preprocessing

Aviation features exist on different scales (for example, Month = 1–12 vs Time = 0000–2359).  
To normalize feature influence, **Z-Score Standardization** was applied:

$$
z_i = \frac{x_i - \mu_i}{\sigma_i}
$$

Where:

- $z_i$ = standardized value of feature $i$  
- $x_i$ = original value of feature $i$  
- $\mu_i$ = mean of feature $i$ in the training dataset  
- $\sigma_i$ = standard deviation of feature $i$ in the training dataset

This ensures all features contribute proportionally to the model.

---

## Model Comparison

| Model               | Accuracy | Precision | Recall  | F1-Score |
|--------------------|----------|-----------|---------|----------|
| Naive Bayes        | 0.6264   | 0.6615    | 0.5197  | 0.5821   |
| Logistic Regression| 0.7038   | 0.7330    | 0.6424  | 0.6847   |
| KNN                | 0.6604   | 0.6806    | 0.6062  | 0.6412   |
| SVM                | 0.6659   | 0.6959    | 0.5909  | 0.6391   |
| Random Forest      | 0.7882   | 0.8361    | 0.7176  | 0.7723   |

### Selected Model: Random Forest

- Strong overall performance  
- High precision (0.8361), which minimizes false delay alerts  
- Robust ensemble-based learning  

---

## Running the Streamlit Application

### 1️⃣ Create a Python Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate       # Mac/Linux
venv\Scripts\activate          # Windows
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Launch App

```bash
streamlit run app.py
```

The app will open automatically in your web browser.

---

## Test Cases

### Test Case 1 — On-Time Scenario

Represents a standard, efficient flight operation.

| Field | Input Value |
|-------|------------|
| Operating Month | May |
| Carrier | Delta Air Lines Inc. |
| Origin Airport | Atlanta International |
| Scheduled Departure | 10:00 (1000) |
| Actual Departure | 10:00 (1000) |
| Taxi Out | 15 |
| Wheels Off | 10:15 (1015) |
| Wheels On | 12:00 (1200) |
| Actual Arrival | 12:05 (1205) |
| Taxi In | 5 |
| Total Elapsed Time | 125 |
| Scheduled Arrival | 12:10 (1210) |

**Expected Result:**  
ON-TIME

---

### Test Case 2 — High-Probability Delay Scenario

High taxi-out time and significant gap between scheduled and actual departure.

| Field | Input Value |
|-------|------------|
| Operating Month | December |
| Carrier | United Air Lines Inc. |
| Origin Airport | Chicago O'Hare |
| Scheduled Departure | 08:00 (0800) |
| Actual Departure | 09:30 (0930) |
| Taxi Out | 45 |
| Wheels Off | 10:15 (1015) |
| Wheels On | 13:00 (1300) |
| Actual Arrival | 13:20 (1320) |
| Taxi In | 20 |
| Total Elapsed Time | 230 |
| Scheduled Arrival | 11:30 (1130) |

**Expected Result:**  
DELAY EXPECTED

---

## Dataset

Dataset sourced from:

**U.S. Department of Transportation Flight Delays Dataset (Kaggle)**  
https://www.kaggle.com/datasets/usdot/flight-delays

---

## Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-Learn  
- Streamlit  

---

## License

For academic and portfolio demonstration purposes.

---

## Contribution

If you find this project useful, feel free to **star**, **fork**, and **contribute**.  
Pull requests and suggestions for improvement are welcome.