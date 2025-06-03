# 🔮 AI Stock Price Predictor (Desktop App)

A Python desktop app that uses **machine learning (AI)** to predict future stock prices based on historical data.  
Built with **PyQt5**, **yfinance**, and **scikit-learn**, this app allows users to analyze stocks, compare model performance, and generate 7-day price forecasts with visual charts.

---

## 🚀 Features

- 📈 Fetches real-time stock data from Yahoo Finance using yfinance
- 🧠 Trains 3 machine learning models:
  - Linear Regression
  - Random Forest Regressor
  - K-Nearest Neighbors (KNN)
- ⚙️ Applies feature engineering:
  - Moving averages (5, 10, 20-day)
  - Daily returns, price changes, volume trends
- 🎨 Full GUI built with PyQt5 and custom QSS styles
- 📊 Visualizes:
  - Historical stock prices
  - Model predictions vs actual prices
  - Future 7-day price forecasts
- ⚡ Uses multithreading to keep the UI responsive

---

## 📦 Installation

### Requirements

- Python 3.x
- `pip install -r requirements.txt`

**Dependencies include:**

- PyQt5  
- yfinance  
- scikit-learn  
- matplotlib  
- seaborn  
- pandas  
- numpy


✨ Example Use
Enter a stock ticker (e.g., AAPL, TSLA, MSFT)
Click Analyze
App will:
  Fetch live data
  Train models
  Show model accuracy
  Forecast 7-day stock prices
  Display charts in the GUI


📊 Model Evaluation
Each model is evaluated using:
  R² Score – how well predictions match actual values
  Mean Squared Error (MSE) – how far off the predictions are
  The best-performing model is used for forecasting

🛠️Future Improvements
  Add support for LSTM or other deep learning models (AI upgrade)
  Allow saving prediction history
  Add confidence intervals to forecasts
  Export reports as PDF or CSV