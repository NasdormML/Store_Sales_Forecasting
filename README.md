## Store Sales - Time Series Forecasting

![Python](https://img.shields.io/badge/Python-3.11+-brightgreen)
![Seaborn](https://img.shields.io/badge/Seaborn-v0.13.2-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.5.1-yellow)
![XGBoost](https://img.shields.io/badge/XGBoost-v2.1.0-red)
![Optuna](https://img.shields.io/badge/Optuna-v3.0.0-orange)

This project aims to predict future store sales using time series forecasting techniques. The dataset consists of sales data from multiple stores over a period of time, allowing for advanced forecasting and trend analysis.

### Key Features:
- **Store ID:** Identification for each store.
- **Date:** The time component, critical for time series forecasting.
- **Sales:** The target variable representing the store’s sales, which we aim to predict.
  
More detailed exploratory data analysis (EDA) and preprocessing steps are included in the notebook.

---

## Project Structure

- `store-sales-time-series-forecasting/`: Contains the dataset and additional files.
- `Fmodel.ipynb`: Main notebook with EDA, preprocessing, and the first run of the XGBoost model (without hyperparameter tuning).
- `README.md`: Project overview, setup instructions, and details.

---

## Data Preparation and Feature Engineering

The project involves several steps of data preprocessing and feature extraction to prepare the data for modeling:

1. **Data Loading**: The following datasets are loaded:
   - `holidays_events.csv` (holiday information)
   - `oil.csv` (oil price data)
   - `stores.csv` (store information)
   - `transactions.csv` (store transaction counts)
   - `train.csv` and `test.csv` (training and testing sales data)

   ```python
   df_holidays = pd.read_csv(comp_dir / "holidays_events.csv", ...)
   df_oil = pd.read_csv(comp_dir / "oil.csv", parse_dates=['date'])
   df_train = pd.read_csv(comp_dir / 'train.csv', parse_dates=['date'])
   ```

2. **Exploratory Data Analysis (EDA)**:
   - **Oil Prices**: The impact of oil price fluctuations on sales was visualized using line plots.
   - **Sales Trends**: Monthly and weekly sales trends were analyzed, including:
     - Bar plots of average sales per month.
     - Weekly sales distribution with `day_of_week` features.
   - **Boxplots**: Sales distributions by month were visualized using box plots to detect potential outliers or seasonal patterns.

   ```python
   sns.lineplot(data=df_oil, x='date', y='dcoilwtico')
   plt.title('Oil Prices Over Time')
   plt.show()

   train['month'] = train['date'].dt.month
   monthly_sales = train.groupby('month')['sales'].mean()
   monthly_sales.plot(kind='bar', color='orange')
   plt.title('Average Sales Per Month')
   plt.show()
   ```

3. **Feature Engineering**:
   - Extracted key temporal features such as `year`, `month`, `day_of_week`, `week_of_year`, and a binary indicator `is_weekend`.
   - Created lag features (`lag_7_sales`) and rolling averages (`rolling_mean_7`) to capture temporal dependencies.

   ```python
   train['lag_7_sales'] = train['sales'].shift(7)
   train['rolling_mean_7'] = train['sales'].shift(1).rolling(window=7).mean()
   ```

4. **Categorical and Numerical Feature Processing**:
   - Applied one-hot encoding to categorical features (e.g., holiday types, store information).
   - Imputed missing values and scaled numerical features using StandardScaler.

   ```python
   categorical_transformer = Pipeline(steps=[
       ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
       ('encoder', OneHotEncoder(handle_unknown='ignore'))
   ])
   ```

---

## Modeling

The main model used in this project is **XGBoost**, selected for its robustness and support for CUDA, enabling GPU acceleration. The pipeline includes preprocessing steps and the model training process.

- **TimeSeriesSplit**: Used to split the data into sequential training and testing sets, preserving the temporal order.
- **XGBoost Model**: The model was trained with default hyperparameters initially and further optimized through `optuna`.

```python
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', XGBRegressor(n_estimators=500, colsample_bytree=0.5, subsample=0.7, 
                           min_child_weight=3, learning_rate=0.05, max_depth=6, reg_lambda=0.5, reg_alpha=0.5,
                           device = "cuda", tree_method='hist', objective='reg:squarederror', random_state=42,))
])
```

---

## Hyperparameter Optimization with Optuna

We integrated **Optuna** for automatic hyperparameter tuning to further improve the model's performance. 

### Key Hyperparameters Optimized:
- `n_estimators`: Number of trees.
- `max_depth`: Maximum depth of the trees.
- `learning_rate`: Step size for each boosting iteration.
- `subsample`: Fraction of samples used to train each tree.
- `colsample_bytree`: Fraction of features used to train each tree.
- `min_child_weight`: Minimum sum of instance weights for child nodes.
- `reg_alpha`: L1 regularization term.
- `reg_lambda`: L2 regularization term.

The best hyperparameters found by Optuna after 20 trials were:

```bash
Best RMSE: 413.95

Best hyperparameters:
{
    'n_estimators': 992,
    'max_depth': 12,
    'learning_rate': 0.0083,
    'subsample': 0.7746,
    'colsample_bytree': 0.8047,
    'min_child_weight': 6,
    'reg_alpha': 3.62,
    'reg_lambda': 4.25
}
```

### Visualization of the Optimization Process

Optuna provides useful visualizations to track the optimization process:

#### 1. Optimization History
This plot shows how the RMSE changed during the optimization process:
```python
vis.plot_optimization_history(study)
```

#### 2. Hyperparameter Importance
This plot shows which hyperparameters had the most influence on the model's performance:
```python
vis.plot_param_importances(study)
```

---

## Getting Started

### Prerequisites

Ensure that the following libraries are installed:
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `optuna`
- `matplotlib`
- `seaborn`

You can install them with the following command:
```bash
pip install pandas numpy scikit-learn xgboost matplotlib seaborn optuna
```

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/NasdormML/Time_Series.git
cd Time_Series
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```
