# Store Sales - Time Series Forecasting

![LightGBM](https://img.shields.io/badge/LightGBM-v3.3.2-blue)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Overview

This project predicts future store sales using time series forecasting techniques. The model is trained using LightGBM.

## Dataset

The dataset contains historical sales data from a store, including features such as:
- Date
- Store ID
- Product ID
- Sales Quantity
- Promotional Events

You can download the dataset from [Kaggle](https://www.kaggle.com/competitions/store-sales-time-series-forecasting).

## Project Structure

```plaintext
├── data
│   ├── raw
│   └── processed
├── notebooks
│   ├── EDA.ipynb
│   └── Model_Training.ipynb
├── src
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── evaluation.py
├── README.md
└── requirements.txt
