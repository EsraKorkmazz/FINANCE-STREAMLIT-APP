# Portfolio Optimization and Credit Score Forecasting App
This repository contains the code for a Streamlit app that combines portfolio optimization and credit score prediction.
The app is designed to help users make smarter financial decisions by analyzing their investment portfolios and predicting credit scores using machine learning models.

## Project Overview

The app integrates two key financial tools:

Portfolio Optimization: Allows users to input stock data and calculate the optimal portfolio allocation using mean-variance optimization.
Credit Score Prediction: A machine learning model predicts the credit score category (Poor, Regular, Good) based on various user inputs such as age, income, credit utilization, and payment history.

##Key Features

Portfolio Optimization: Real-time analysis of stock data and portfolio recommendations.
Credit Score Prediction: Uses a trained Random Forest model to predict credit score categories.
User-Friendly Interface: Built with Streamlit for easy interaction.
Data Visualization: Provides graphs and charts to help users visualize their portfolio and credit data.

##Technologies Used

Streamlit for the web interface.
Pandas, NumPy for data manipulation.
Scikit-learn, PyCaret, Optuna for machine learning and optimization.
Seaborn, Matplotlib, Plotly for data visualization.

##Modeling Approach

Portfolio Optimization: Uses mean-variance optimization to balance risk and return.
Credit Score Prediction: A Random Forest Classifier is trained on a dataset of financial behavior to predict whether a user's credit score will be Poor, Regular, or Good. Hyperparameters were optimized using Optuna for better accuracy.
