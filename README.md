# 🚦 RoadCast AI

## AI-Powered Traffic Congestion Nowcasting & Prediction

RoadCast AI is an AI-powered traffic intelligence platform designed to predict urban traffic congestion before it becomes severe.

Instead of only showing the current traffic situation, RoadCast AI uses machine learning and traffic data to forecast congestion for the next **15, 30, 45, and 60 minutes**.

The goal is simple:

> **Move traffic management from reactive to predictive.**

---

## 🎯 The Problem

Urban traffic congestion is dynamic and difficult to predict.

Most traffic systems tell us:

**"Where is traffic right now?"**

But traffic management needs to answer a more important question:

**"Where is traffic going to become congested next?"**

RoadCast AI addresses this challenge through short-term traffic nowcasting and congestion prediction.

---

## 💡 My Solution

RoadCast AI combines:
`
- 📊 Historical traffic patterns
- 🚗 Traffic speed observations
- 🕒 Time-based features
- 📈 Recent traffic trends
- 🤖 Machine learning
- 🗺️ Interactive traffic visualization

The system generates traffic predictions for:

| Horizon | Purpose |
|---|---|
| 🟢 15 Minutes | Immediate traffic changes |
| 🟡 30 Minutes | Near-term congestion |
| 🟠 45 Minutes | Medium-term traffic conditions |
| 🔴 60 Minutes | Extended traffic outlook |

---

# 🧠 Machine Learning

RoadCast AI currently uses **XGBoost regression models** for traffic forecasting.

A separate model is used for each prediction horizon:

- 15-minute model
- 30-minute model
- 45-minute model
- 60-minute model

### Model Performance

| Prediction Horizon | MAE | RMSE | R² |
|---|---:|---:|---:|
| 15 Minutes | 4.23 | 9.37 | 0.830 |
| 30 Minutes | 5.76 | 11.87 | 0.727 |
| 45 Minutes | 6.88 | 13.36 | 0.654 |
| 60 Minutes | 7.73 | 14.43 | 0.596 |

The model performance naturally decreases as the prediction horizon increases, reflecting the increasing uncertainty of longer-term traffic forecasting.

---

# 🚦 Congestion Intelligence

RoadCast AI converts predicted traffic speed into a normalized congestion score between **0 and 1**.

The score is then converted into an easy-to-understand traffic severity level.

| Congestion | Severity |
|---|---|
| 🟢 0–25% | LOW |
| 🟡 25–50% | MODERATE |
| 🟠 50–75% | HIGH |
| 🔴 75–100% | SEVERE |

This allows complex machine learning predictions to be understood quickly by traffic operators and users.

---

# 🗺️ Interactive Traffic Dashboard

The RoadCast AI dashboard provides a real-time traffic intelligence interface with:

- 🗺️ Interactive traffic map
- 🚦 Congestion visualization
- 📍 Location-based traffic monitoring
- 📊 15/30/45/60-minute predictions
- 📈 Congestion trajectory
- 🚗 Average traffic speed
- 🔥 Most congested corridors
- 🤖 AI-generated traffic insights
- 🎯 Traffic severity classification
- 📌 Interactive traffic markers
- 📉 Forecast trends

The dashboard is designed as a **traffic command center**, allowing users to understand both current and upcoming congestion.

---

# 📈 Congestion Forecast

RoadCast AI visualizes the expected congestion trajectory:

| Time | Congestion | Predicted Speed | Severity |
|---|---:|---:|---|
| NOW | 68% | 31 km/h | HIGH |
| 15 MIN | 71% | 29 km/h | HIGH |
| 30 MIN | 75% | 26 km/h | SEVERE |
| 45 MIN | 78% | 24 km/h | SEVERE |
| 60 MIN | 81% | 22 km/h | SEVERE |

This allows users to understand how traffic conditions may evolve over the next hour.

---

# 🏗️ System Architecture

```text
                  ┌──────────────────────┐
                  │    Traffic Data      │
                  │                      │
                  │ Historical + Live    │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Data Preprocessing   │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Feature Engineering  │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ XGBoost Prediction   │
                  └──────────┬───────────┘
                             │
             ┌───────────────┼───────────────┐
             ▼               ▼               ▼
          15 MIN          30 MIN          45/60 MIN
             │               │               │
             └───────────────┼───────────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Congestion Scoring   │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │    FastAPI Backend   │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ RoadCast AI Dashboard│
                  └──────────────────────┘

🛠️ Technology Stack

    Backend
    Python
    FastAPI
    XGBoost
    Pandas
    NumPy
    SQLite
    Frontend
    React
    JavaScript
    HTML5
    CSS3
    Leaflet
    Interactive Charts
    Machine Learning
    XGBoost Regression
    Feature Engineering
    Time-Series Forecasting
    Traffic Congestion Scoring
    Data
    METR-LA Traffic Dataset
    TomTom Traffic Observations

📂 Project Structure
    RoadCast-AI/
    │
    ├── backend/
    │   ├── data/
    │   ├── models/
    │   ├── preprocessing.py
    │   ├── nowcasting.py
    │   ├── training.py
    │   ├── main.py
    │   ├── requirements.txt
    │   └── ...
    │
    ├── frontend/
    │   ├── src/
    │   ├── public/
    │   ├── package.json
    │   └── ...
    │
    ├── .gitignore
    └── README.md

⚙️ Installation

    1. Clone the Repository
    git clone https://github.com/Vedant-npc/RoadCast-AI.git
    cd RoadCast-AI

🐍 Backend Setup

    Navigate to the backend:

        cd backend

    Create a virtual environment:

        python -m venv .venv

        Activate the environment on Windows:

        .venv\Scripts\activate

        Install dependencies:

        pip install -r requirements.txt

    Start the backend:

        uvicorn main:app --reload

    The backend will run at:

        http://localhost:8000

    FastAPI documentation:

        http://localhost:8000/docs

💻 Frontend Setup

    Open a new terminal and navigate to the frontend:

    cd frontend

    Install dependencies:

    npm install

    Start the development server:

    npm run dev

    Open the local URL shown in the terminal.

🔮 Prediction Pipeline

    RoadCast AI follows the following prediction pipeline:

    Traffic Data
        ↓
    Data Cleaning
        ↓
    Feature Engineering
        ↓
    Historical Traffic Features
        ↓
    Time Features
        ↓
    Trend Features
        ↓
    XGBoost Model
        ↓
    Predicted Speed
        ↓
    Congestion Score
        ↓
    Severity Classification
        ↓
    Dashboard

📊 Features Used

    The prediction system uses a combination of:

    Historical Traffic
    Previous traffic observations
    Recent speed values
    Historical traffic patterns
    Time Features
    Hour
    Day of week
    Weekend indicator
    Minute of hour
    Trend Features
    Short-term traffic trends
    Rolling averages
    Recent traffic changes

    These features allow the model to understand both the current traffic state and its recent direction.

🚀 Deployment


    Frontend

        Vercel
        Set the `VITE_API_BASE` environment variable to the deployed backend
        URL followed by `/api` (for example, `https://<your-render-service>.onrender.com/api`).

    Backend

        Render 
        Deploy with the included `render.yaml` blueprint. The frontend calls
        `/api` by default during local development; Vite proxies those requests
        to `http://127.0.0.1:8000`.

The backend allows cross-origin requests because the dashboard uses a public,
unauthenticated API. Do not send credentials or private data through these endpoints.


🔬 Future Improvements

    RoadCast AI can be further enhanced with:

    🌧️ Weather-aware traffic prediction
    🚨 Accident detection
    🚧 Road closure detection
    🎉 Event-based traffic prediction
    🧠 Graph Neural Networks
    🤖 Transformer-based traffic forecasting
    🛣️ Dynamic route recommendation
    📱 Mobile application
    🔔 Real-time congestion alerts
    🌍 Multi-city deployment
    📊 Advanced historical traffic analytics
    🚦 Adaptive traffic signal optimization

🎯 Vision

    RoadCast AI aims to transform urban traffic management from:

    REACTIVE
    ↓
    MONITOR
    ↓
    ANALYZE
    ↓
    PREDICT
    ↓
    ACT
    ↓
    PREDICTIVE TRAFFIC MANAGEMENT


🏆 Project Highlights

    🤖 AI-powered traffic prediction
    🚦 Four prediction horizons
    🗺️ Interactive traffic intelligence map
    📈 Congestion trajectory visualization
    📊 Machine learning-based forecasting
    ⚡ FastAPI prediction backend
    📍 Location-based congestion monitoring
    🎯 Traffic severity classification
    🌐 Deployable web architecture

👨‍💻 Created By
    Vedant Mande
    AI & Data Science