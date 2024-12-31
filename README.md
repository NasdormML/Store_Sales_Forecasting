# Store Sales - Time Series Forecasting  

![Python](https://img.shields.io/badge/Python-3.11-brightgreen)
![Seaborn](https://img.shields.io/badge/Seaborn-v0.13.2-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.5.1-yellow)
![XGBoost](https://img.shields.io/badge/XGBoost-v2.1.0-red)
![Optuna](https://img.shields.io/badge/Optuna-v3.0.0-orange)
[![CI/CD](https://github.com/NasdormML/Store_Sales_Forecasting/actions/workflows/ci.yml/badge.svg)](https://github.com/NasdormML/Store_Sales_Forecasting/actions/workflows/ci.yml)

---

## Overview  
This end-to-end project aims to forecast retail store sales based on historical data, helping businesses optimize inventory management, reduce waste, and improve revenue planning. It includes data preprocessing, feature engineering, model training, and deployment-ready code.  

## Key Features  
- **Data Preparation**: Addressed missing values, normalized time series, and created aggregated features for store and category levels.  
- **Feature Engineering**: Generated lag features, rolling averages, seasonal indicators, and holiday-based features to capture temporal patterns.  
- **Model Optimization**: Applied Optuna for hyperparameter tuning of XGBoost, reducing RMSLE by 15%.  
- **Validation and Testing**: Used TimeSeriesSplit for proper evaluation of sequential data and implemented Pytest for functional testing of core modules.  
- **Deployment-Ready**: Integrated CI/CD pipelines and containerized the project using Docker.  

## Results  
- Achieved an RMSLE of **0.75094**, outperforming baseline methods such as moving averages and linear regression by 15%.  
- Forecasting accuracy provides actionable insights for inventory and demand planning.  

## Business Value  
- **Inventory Optimization**: Accurate sales forecasts reduce overstock and stockouts, minimizing storage costs and lost revenue.  
- **Revenue Planning**: Helps align inventory and workforce with expected sales patterns.  
- **Strategic Insights**: Enables better decision-making for promotions, pricing, and holiday planning.  

## Tools & Technologies  
- **Programming Language**: Python  
- **Libraries**: pandas, numpy, XGBoost, Optuna, Scikit-learn, Matplotlib, Seaborn  
- **Tools**: Jupyter Notebook, Pytest, Docker, CI/CD  

## How to Run  
### Option 1: Running Locally  
1. Clone the repository:  
   ```bash  
   git clone https://github.com/NasdormML/Store_Sales_Forecasting.git  
   cd Store_Sales_Forecasting  
   ```  
2. Install the required dependencies:  
   ```bash  
   pip install -r requirements.txt  
   ```  
3. Run the main script:  
   ```bash  
   python main.py  
   ```  

### Option 2: Using Docker  
1. Build the Docker image:  
   ```bash  
   docker build -t store-sales .  
   ```  
2. Run the Docker container:  
   ```bash  
   docker run -p 8080:8080 store-sales  
   ```  

## Data  
- **Source**: The dataset is publicly available on [Kaggle](#). It includes historical sales data, product categories, holidays, and other relevant features.  
- **Preprocessing**:  
  - Cleaned missing values using forward filling and interpolation methods.  
  - Created aggregated features at category and store levels.  
  - Removed outliers and addressed data leakage risks.  

## Project Structure  
```
time_series_project/  
├── .github/  
│   └── workflows/  
│       └── ci.yml               # CI/CD configuration file  
├── data/  
│   ├── processed/               # Preprocessed data ready for modeling  
│   ├── raw/                     # Raw input data  
├── models/                      # Saved trained models  
├── notebooks/                   # Jupyter notebooks for exploratory analysis  
│   ├── EDA.ipynb                # Exploratory Data Analysis  
│   └── store_sales_kaggle.ipynb # Additional exploratory analysis  
├── src/                         # Source code of the project  
│   ├── data_preparation.py      # Code for data preprocessing  
│   ├── model_prediction.py      # Code for generating predictions  
│   ├── model_training.py        # Code for training the model  
├── tests/                       # Unit and integration tests  
├── dockerfile                   # Dockerfile for containerizing the project  
├── main.py                      # Entry point for running the project  
├── README.md                    # Project description and documentation  
└── requirements.txt             # Python dependencies  
```  

## Contact  
If you have any questions or suggestions, feel free to reach out:  
- **Email**: nasdorm.ml@inbox.ru  

## License  
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
