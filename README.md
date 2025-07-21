# 💰 Medical Insurance Cost Prediction

## 📝 Description

This project aims to predict the medical insurance cost of individuals based on demographic and lifestyle attributes like age, BMI, smoking status, and region using machine learning regression models. A web app is also created using Flask for live predictions.

## 🛠️ Tech Stack

- Python
- Pandas, NumPy, Matplotlib, Seaborn
- Scikit-learn
- XGBoost
- Flask (for deployment)
- HTML (Flask Templates)

## ✅ Features

- Data preprocessing and EDA (Exploratory Data Analysis)
- Feature encoding and scaling
- Model training with:
  - Linear Regression
  - Ridge Regression
  - Support Vector Regression (SVR)
  - Random Forest Regressor
- Hyperparameter tuning with GridSearchCV
- Evaluation using R² Score, RMSE, and Cross-Validation
- Model comparison visualization
- Flask-based web interface for real-time predictions
- Model saved using `pickle` and reused for deployment

## 📸 Screenshots (Optional)

*Include here:*
- Heatmap of correlation
- Model comparison bar chart
- Web interface (Flask form and output)

## 🚀 How to Run

### 1. 📊 Train the Model
Run the notebook/script to:
- Load and preprocess the `insurance.csv` dataset
- Train and evaluate different regression models
- Save the best model (`rf_tuned.pkl`)

### 2. 🌐 Run the Flask App

#### Folder Structure:
