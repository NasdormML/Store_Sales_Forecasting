## Store Sales - Time Series Forecasting

![Python](https://img.shields.io/badge/Python-3.11+-brightgreen)
![Seaborn](https://img.shields.io/badge/Seaborn-v0.13.2-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.5.1-yellow)
![XGBoost](https://img.shields.io/badge/XGBoost-v2.1.0-red)

## Store Sales - Time Series Forecasting

This project aims to predict future store sales using time series forecasting techniques.
The dataset consists of sales data for multiple stores over a period of time. Key features include:

Store ID: Identification of different stores.
Date: The time component of the series.
Sales: The target variable we are predicting.
More detailed EDA and preprocessing steps are included in the notebook.

# Project Structure

- `store-sales-time-series-forecasting/`: Folder containing the dataset.
- `Fmodel.ipynb`: Main notebook with EDA, preprocessing and first run XGB model(without param upgrade).
- `README.md`: Project overview and instructions.

# Modeling

The primary model used in this project is XGBoost, chosen for its high performance and CUDA support. The model was trained with the following default hyperparameters.

# Evaluation

The model was evaluated using the following metrics:
- Root Mean Squared Error (RMSE)

Results:
- RMSE: 317.70

## Getting Started

# Prerequisites

Make sure you have the following libraries installed:
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn

You can install them using:
```bash
pip install pandas numpy scikit-learn lightgbm matplotlib seaborn
```

# Installation:
Clone the repository:
```bash
git clone https://github.com/NasdormML/Time_Series.git
cd Time_Series
```

# Install the required packages:
```bash
pip install -r requirements.txt
```
