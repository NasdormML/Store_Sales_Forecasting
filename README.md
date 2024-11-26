# Store Sales - Time Series Forecasting

![Python](https://img.shields.io/badge/Python-3.11-brightgreen)
![Seaborn](https://img.shields.io/badge/Seaborn-v0.13.2-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-v1.5.1-yellow)
![XGBoost](https://img.shields.io/badge/XGBoost-v2.1.0-red)
![Optuna](https://img.shields.io/badge/Optuna-v3.0.0-orange)

---

## Overview
This project aims to forecast retail store sales based on historical data, helping businesses optimize inventory management, reduce waste, and improve revenue planning. The model leverages advanced machine learning techniques and modern tools for development, validation, and deployment.

## Key Features
- **Time Series Modeling**: Utilized lag features, rolling averages, and other temporal features.
- **Hyperparameter Optimization**: Implemented Optuna to fine-tune model parameters, significantly improving accuracy.
- **Validation**: Applied TimeSeriesSplit to ensure proper evaluation of sequential data.
- **Exploratory Data Analysis (EDA)**: Analyzed trends, seasonality, and key factors influencing sales.

## Results
- Achieved an RMSLE of **0.75094**, improving forecasting accuracy by 15% over baseline methods.
- The model provides actionable insights for inventory optimization and demand planning.

## Tools & Technologies
- **Programming Language**: Python
- **Libraries**: pandas, numpy, XGBoost, Optuna, Scikit-learn, Matplotlib, Seaborn
- **Tools**: Jupyter Notebook, Docker (optional for deployment)

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
3. Open the Jupyter Notebooks:
   - `EDA.ipynb` for exploratory data analysis.
   - `store_sales_kaggle.ipynb` for model training and evaluation.

### Option 2: Using Docker (Optional)
1. Build the Docker image:
   ```bash
   docker build -t store-sales .
   ```
2. Run the Docker container:
   ```bash
   docker run -p 8080:8080 store-sales
   ```
3. (Optional) Deploy a web interface or interact with the model via the command line.

---

## Data
- **Source**: The dataset is publicly available on [Kaggle](#). It includes historical sales data, product categories, holidays, and other relevant features.
- **Key Features**:
  - Temporal features like date, seasonality, and holidays.
  - Store-specific information to model localized trends.
  - Sales data aggregated by category and date.

## Project Structure
```
store-sales-time-series-forecasting/    # Datasets used in the project
notebooks/
│   ├── EDA.ipynb                      # Jupyter notebook for EDA
│   └── store_sales_kaggle.ipynb       # Jupyter notebook for modeling
README.md                               # Project overview and setup
requirements.txt                        # Dependencies and libraries
```

## Business Value
- **Inventory Management**: Helps reduce overstock and stockouts by accurately predicting demand.
- **Revenue Optimization**: Aligns supply with expected sales, improving profit margins.
- **Marketing Insights**: Assists in planning promotions by identifying demand patterns.

## Contact
If you have any questions or would like to discuss this project further, feel free to reach out:
- **Email**: nasdorm.ml@inbox.ru

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
