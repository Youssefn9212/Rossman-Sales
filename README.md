# Rossman Sales Forecasting System

**Author**: Youssef Nakhla  
**Objective**: Develop and deploy a robust, production-ready pipeline for daily sales forecasting across a national retail chain.

---

## Setup and Installation

### Prerequisites
- Python 3.11 or higher
- Git
- Virtual environment (recommended)

### Required Data Files
Due to size limitations, download the following files from [Google Drive](https://drive.google.com/drive/folders/1h2ce97-ukhYEUFwg-E19DCd7Ior-DORS?usp=sharing):
- `test_processed1.csv` (2.3 MB)
- `train_final1.csv` (510.8 MB)
- `train_processed1.csv` (69.8 MB)

Place these files in their respective directories after cloning the repository.

Note: The repository includes `final_submission.csv` in the `Processed Data` directory, which contains the predictions for the test set using our best performing model (XGB + CNN-LSTM ensemble with RMPSE of 0.0259).

### Installation Steps

1. Clone the repository:
```bash
git clone https://github.com/Youssefn9212/Rossman-Sales.git
cd Rossman-Sales
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
cd Deployment
pip install -r requirements.txt
```

4. Download the required data files from the Google Drive link and place them in the `Processed Data/` directory.

### Running the Application

1. Start the Flask application:
```bash
cd Deployment
python app.py
```

2. Open your web browser and navigate to:
```
http://127.0.0.1:5000
```

### Project Structure
```
Rossman-Sales/
├── Dataset/                  # Original competition datasets
├── Deployment/              # Flask web application
│   ├── app.py              # Main application file
│   ├── models/             # Trained models
│   ├── static/             # CSS, JS files
│   ├── templates/          # HTML templates
│   └── requirements.txt    # Python dependencies
├── Notebooks/              # Jupyter notebooks for analysis
│   ├── Preprocessing.ipynb
│   ├── Feature Engineering.ipynb
│   ├── Modelling.ipynb
│   └── Forecasting.ipynb
└── Processed Data/         # Processed datasets
```

### Usage Guide

1. **Sales Forecasting**:
   - Navigate to the prediction page
   - Enter store details and date
   - Click "Generate Forecast"

2. **Analytics Dashboard**:
   - Navigate to the analysis page
   - Use the tabs to explore different aspects:
     - Overview
     - Time Trends
     - Store Performance
     - Promotions & Holidays
     - Seasonality

---

## 1. Introduction

This project focuses on building and deploying a daily sales forecasting model to support key operational decisions including inventory management, staffing, and promotional planning. The final deliverable includes a high-performing predictive pipeline and a real-time API served through a Flask web application.

---

## 2. Preprocessing

- Merged core datasets using the `Store` key to enrich records with store-level metadata.
- Extracted time-based features such as `Year`, `Month`, `Week`, `Day`, and `DayOfWeek`.
- Applied imputation strategies using sentinel values for missing data in key numerical fields.
- Encoded categorical features using label encoding and one-hot encoding where applicable.
- Converted `PromoInterval` into quarterly numeric values and handled missing values accordingly.
- Excluded days when stores were closed (`Open == 0`) from training, as they distort the sales signal.

---

## 3. Feature Engineering

- Created features capturing temporal patterns, store characteristics, and event-related flags.
- Modeled interactions between promotional periods and calendar time.
- Initially incorporated lag features and rolling statistics to capture recent trends, along with seasonal averages by store type and assortment.
- These features were later removed due to their reliance on future or unavailable sales data, which would introduce data leakage in a real-time prediction setting.
- All models were retrained after removing these features to ensure valid evaluation and deployability.

---

## 4. Models Used

The modeling phase employed both traditional and deep learning techniques:

- Tree-based models: XGBoost, LightGBM, CatBoost, and Random Forest for structured data modeling.
- Linear baseline: Ridge Regression for interpretability and benchmarking.
- Deep learning: CNN–LSTM hybrid to capture short-term dependencies and temporal sequences.
- Ensemble strategies: Included simple and weighted averaging of model predictions.
- A stacking model was tested but excluded due to overfitting concerns.
- Month-filtering strategy was used to restrict training data to calendar-aligned months, inspired by high-performing Kaggle solutions.

---

## 5. Results

Model performance was evaluated using Root Mean Percentage Squared Error (RMPSE), aligned with the Kaggle competition metric. Results are as follows:

| Model                             | RMPSE   |
|----------------------------------|---------|
| XGBoost Regressor                | 0.1283  |
| LightGBM Regressor               | 0.1554  |
| HistGradientBoosting Regressor  | 0.1429  |
| CatBoost Regressor               | 0.1429  |
| Random Forest Regressor          | 0.1428  |
| CNN–LSTM                         | 0.0419  |
| XGB + LGB + CAT (ensemble)       | 0.1136  |
| XGB + RF + HistGBR (ensemble)    | 0.1357  |
| XGB + CNN–LSTM (ensemble)        | 0.0259  |
| Kaggle Competition Winner        | 0.1002  |

The XGB + CNN–LSTM ensemble model outperformed the state-of-the-art Kaggle solution based on internal validation. While promising, these results should be interpreted cautiously until validated on live production data.

---

## 6. Forecasting

The complete pipeline was applied to the test dataset in a manner consistent with the training process. All data transformations were carefully aligned, excluding any features that rely on future or unavailable sales data. The resulting predictions were submitted in the required format. This marks the transition from development to deployment, with the pipeline now ready for integration into operational systems.

---

## 7. Flask Application

A web-based interface was developed to provide business users with both predictive functionality and analytics insights.

### Sales Forecasting Interface

- Users can input store and calendar features through a web form.
- Input fields adapt based on selections (e.g., Promo2 visibility).
- The forecast is generated using the deployed XGB + CNN–LSTM ensemble model.

![Sales Forecasting Interface](Flask%20App%20Demos/new_sales.png)
![Sales Form](Flask%20App%20Demos/new_sales1.png)

### Prediction History

- All forecasts are stored in memory and displayed in a tabular format.
- Includes store number, forecast date, predicted sales, and timestamp.

![Prediction History](Flask%20App%20Demos/history.png)

### Sales Analysis Dashboard

Includes the following interactive visualizations using Plotly:

- Overview: Aggregate metrics and trend plots.
![Overview Dashboard](Flask%20App%20Demos/overview.png)

- Time Trends: Sales by day of week and month with rolling averages.
![Time Trends Analysis](Flask%20App%20Demos/trends.png)

- Store Performance: Top stores and daily sales visualized with boxplots and line charts.
![Store Performance Analysis](Flask%20App%20Demos/store_pref.png)

- Promotions & Holidays: Comparative analysis across different types of events.
![Promotions Analysis](Flask%20App%20Demos/promotions.png)

- Seasonality by Store: Decomposition of sales trends per store.
![Seasonality Analysis](Flask%20App%20Demos/seasonality.png)

### Data Handling

- Dataset is loaded and cached on application startup for optimal performance.
- Clear error messages are shown in the event of missing data or failed processing.

---

## 8. Conclusion

This project successfully delivered a robust and accurate forecasting system for daily sales at the store level. The final model—a hybrid ensemble of XGBoost and CNN–LSTM—achieved an RMPSE of 0.0259, indicating that predictions are within 2.6% of actual sales on average. This level of precision allows for confident, data-driven decision-making in key retail operations.

### Business Implications

- **Inventory Optimization**: Improved stock planning reduces costs and customer dissatisfaction.
- **Workforce Scheduling**: Better alignment of staffing with expected demand.
- **Promotional Planning**: Enhanced ability to evaluate and plan effective campaigns.
- **Store Strategy**: Localized forecasts enable tailored strategies by store.
- **Real-Time Integration**: The system is designed for deployment via API, supporting real-time dashboards and automation.

### Recommendation

A pilot rollout across a subset of stores is recommended to evaluate live performance. Post-validation, a full-scale deployment can form the foundation of an intelligent retail operations platform.

---


